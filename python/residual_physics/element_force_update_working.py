from typing import Optional
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch import Tensor
from einops.layers.torch import Rearrange
import numpy as np

def init_weight(m: nn.Module) -> None:
    if hasattr(m, 'init_weight'):
        m.init_weight()
        return
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_nonlinearity(nonlinearity: Optional[str], **kwargs) -> Optional[nn.Module]:

    if nonlinearity is None:
        return nn.Identity()
    elif nonlinearity.casefold() == 'relu':
        return nn.ReLU(inplace=True, **kwargs)
    elif nonlinearity.casefold() == 'tanh':
        return nn.Tanh(**kwargs)
    elif nonlinearity.casefold() in ['silu', 'swish']:
        return nn.SiLU(**kwargs)
    elif nonlinearity.casefold() == 'gelu':
        return nn.GELU(**kwargs)
    elif nonlinearity.casefold() == 'elu':
        return nn.ELU(**kwargs)
    else:
        raise ValueError('unexpected nonlinearity: {}'.format(nonlinearity))

def get_norm(kind, num_features, dim=1, affine=True):
    if kind is None or kind == 'wn':   # weight norm handled on the Linear itself
        return nn.Identity()
    if kind.lower() in ('bn','batchnorm','batch_norm'):
        return nn.BatchNorm1d(num_features, affine=affine)
    if kind.lower() in ('ln','layernorm','layer_norm'):
        return nn.LayerNorm(num_features, elementwise_affine=affine)
    raise ValueError(f"Unknown norm kind: {kind}")

class MLPBlock(nn.Module):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            no_bias: bool,
            norm: Optional[str],
            nonlinearity: Optional[str]) -> None:

        super().__init__()
        if norm == 'wn':
            self.fc = nn.utils.weight_norm(nn.Linear(in_planes, out_planes, not no_bias))
        else:
            self.fc = nn.Linear(in_planes, out_planes, bias=not no_bias and norm is None, dtype=torch.float64)
        self.norm = get_norm(norm, out_planes, dim=1, affine=not no_bias)
        self.nonlinearity = get_nonlinearity(nonlinearity)

    def forward(self, x: Tensor) -> Tensor:

        x = self.fc(x)
        x = self.norm(x)
        x = self.nonlinearity(x)
        return x

def _make_mlp(in_dim, hidden_dims, out_dim, nonlinearity='gelu', no_bias=False, norm=None):
    act = get_nonlinearity(nonlinearity)

    layers = []
    d = in_dim
    for h in hidden_dims:
        layers.append(MLPBlock(d, h, no_bias, norm, nonlinearity))
        d = h
    layers.append(MLPBlock(d, out_dim, no_bias, None, None))
    return nn.Sequential(*layers)

class DiagonalScale(nn.Module):
    """
    Learnable diagonal scaling layer: y = diag(w) * x
    - Supports any input shape (..., N): scales the last dimension.
    - No bias; just element-wise multiplicative parameters.
    """

    def __init__(self, n_features: int, init_scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((n_features,), init_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight  # broadcast over leading dims

class VarNorm1d(nn.Module):
    """
    Per-channel variance normalization (no mean subtraction).
    Guarantees 0 -> 0. Works for (B,C) or (B,C,L).
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features, dtype=torch.float64))
            self.bias = None  # keep bias None to preserve 0->0
        else:
            self.register_parameter('weight', None)
            self.bias = None

        if track_running_stats:
            self.register_buffer('running_var', torch.ones(num_features, dtype=torch.float64))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

    def eval(self):
        super().eval()
        # temporarily ignore running stats
        self.track_running_stats = False
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "Expected (B,C)"
        dims = (0,)

        # print(self.training or not self.track_running_stats)
        if self.training or not self.track_running_stats:
            var = x.var(dims, keepdim=False, unbiased=False)  # (C,)
            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    m = self.momentum
                    self.running_var.lerp_(var, m)
        else:
            var = self.running_var

        inv_std = torch.rsqrt(var + self.eps)                 # (C,)
        shape = (1, -1) 
        y = x * inv_std.view(*shape)                          # no mean subtraction, no bias
        if self.affine:
            y = y * self.weight.view(*shape)
        return y


class Unmodelled_Acceleration(nn.Module):
    def __init__(self, actuated, hidden_dims, nonlinearity, no_bias=True, normalize_inputs=True):
        super().__init__()
        self.actuated = actuated
        self.input_dim = 41 if self.actuated else 38
        self.normalize_inputs = normalize_inputs
        if self.normalize_inputs:
            #self.normalize = VarNorm1d(self.input_dim)
            self.normalize = torch.nn.BatchNorm1d(self.input_dim, dtype=torch.float64)

        self.net = _make_mlp(in_dim=self.input_dim, hidden_dims=hidden_dims, out_dim=3,
                             nonlinearity=nonlinearity, no_bias=no_bias)
    
    def input_features(self, feat_deformation, feat_strainRate, feat_spin, feat_rigidMotion, feat_force):

        feat = torch.cat([feat_deformation,
                            feat_strainRate,
                            feat_spin,
                            feat_rigidMotion
                            ], dim=-1)
        
        if self.actuated:
            feat = torch.cat([feat,
                                feat_force
                                ], dim=-1)
        

        return feat

    def forward(self, feat_deformation, feat_strainRate, feat_spin, feat_rigidMotion, feat_force):  

        
        feat = self.input_features(feat_deformation, 
                                    feat_strainRate, 
                                    feat_spin, 
                                    feat_rigidMotion,
                                    feat_force)
        orig_shape = feat.shape[:-1]                            # (...,input_dim)
        feat = feat.reshape(-1,self.input_dim)                  # (N,input_dim)

        # feat = torch.cat([feat[:,:-14], self.normalize(feat[:,-14:-11]), feat[:,-11:]], dim=-1)

        if self.normalize_inputs:
            norm_feat = self.normalize(feat)
            a = self.net(norm_feat).view(*orig_shape,3)
        else:
            a = self.net(feat).view(*orig_shape,3)
        
        # a = self.net(feat).view(*orig_shape,3)

        return a


class ElementResidual(nn.Module):
    def __init__(self, dof, elements, faces, mu, lam, rho, X_e, dx, hidden_size=128, num_hidden_layer=4, nonlinearity='elu', no_bias=True, actuated=False, gravity=True, scale=1, changing_boundary_indices=None):
        super().__init__()
        
        self.dof = dof                                      # num_vertices * 3
        self.register_buffer('elements', elements.long())   # element vertex index mapping  
        self.register_buffer('faces', faces.long())   # element vertex index mapping  
        self.register_buffer('mu', mu)                      # material Lame parameter mu
        self.register_buffer('lam', lam)                    # material Lame parameter lambda
        self.register_buffer('rho', rho)                    # material density
        self._precompute_quadrature(X_e, dx)                # shape functions for FEM            

        self.actuated = actuated
        if self.actuated:
            if gravity:
                self.register_buffer('g', scale * torch.tensor([0, 0, -9.80709], dtype=torch.float64))
            else:
                self.register_buffer('g', torch.tensor([0, 0, 0], dtype=torch.float64))

        # features -> acceleration density
        self.unmodelled_nn = Unmodelled_Acceleration(actuated=self.actuated, 
                                                        hidden_dims=num_hidden_layer*[hidden_size],
                                                        nonlinearity=nonlinearity,
                                                        no_bias=no_bias)   
        # self.unmodelled_nn = Unmodelled_Stress(actuated=self.actuated, 
        #                                                 hidden_dims=num_hidden_layer*[hidden_size],
        #                                                 nonlinearity=nonlinearity,
        #                                                 no_bias=no_bias)  
        self.changing_boundary_indices = changing_boundary_indices

    def _precompute_quadrature(self, X_e, dx):
        E, Ne = self.elements.shape

        # Store reference positions q0 (flattened 3*V)
        self.register_buffer('q0', torch.tensor(X_e, dtype=torch.float64))

        if Ne == 8:

            Q = 8

            print("Hex")

            self._mesh_type = 'hex'

            # ---------------------------------------------------------
            # Interpret dx: scalar or 3-vector [hx, hy, hz]
            # ---------------------------------------------------------
            dx_t = torch.as_tensor(dx, dtype=torch.float64)
            if dx_t.numel() == 1:
                dx_t = dx_t.repeat(3)  # isotropic: (dx, dx, dx)
            assert dx_t.numel() == 3, "dx must be a scalar or length-3 (hx, hy, hz)."


            hx, hy, hz = dx_t
            inv_dx_x = 1.0 / hx
            inv_dx_y = 1.0 / hy
            inv_dx_z = 1.0 / hz

            # ---------------------------------------------------------
            # Reference Gauss points in [0,1]^3 (independent of size)
            # Node order: [(0,0,0), (0,0,1), (0,1,0), (0,1,1),
            #             (1,0,0), (1,0,1), (1,1,0), (1,1,1)]
            # ---------------------------------------------------------
            samples = torch.tensor(
                [
                    [0., 0., 0.],
                    [0., 0., 1.],
                    [0., 1., 0.],
                    [0., 1., 1.],
                    [1., 0., 0.],
                    [1., 0., 1.],
                    [1., 1., 0.],
                    [1., 1., 1.],
                ],
                dtype=torch.float64,
            )
            samples -= 0.5
            samples /= np.sqrt(3)
            samples += 0.5
            # scale to physical coordinates with hx, hy, hz
            samples = samples * dx_t  # broadcast: (8,3) * (3,) -> (8,3)

            # Precompute N and gradN for these samples (same for all hex elements)
            N_q_e = []
            gradN_q_e = []

            for s in range(8):
                x, y, z = samples[s, 0], samples[s, 1], samples[s, 2]
                nx, ny, nz = x * inv_dx_x, y * inv_dx_y, z * inv_dx_z
                cnx, cny, cnz = 1.0 - nx, 1.0 - ny, 1.0 - nz

                # Shape functions N at sample s
                N_q_e.append(
                    [
                        cnx * cny * cnz,   # N000
                        cnx * cny * nz,    # N001
                        cnx * ny * cnz,    # N010
                        cnx * ny * nz,     # N011
                        nx * cny * cnz,    # N100
                        nx * cny * nz,     # N101
                        nx * ny * cnz,     # N110
                        nx * ny * nz,      # N111
                    ]
                )

                # Gradients in physical coordinates:
                # d/dx = (d/dnx) * (1/hx), etc.
                gradN_q_e.append(
                    [
                        # N000
                        [-inv_dx_x * cny * cnz,  cnx * -inv_dx_y * cnz,  cnx * cny * -inv_dx_z],
                        # N001
                        [-inv_dx_x * cny * nz,   cnx * -inv_dx_y * nz,   cnx * cny *  inv_dx_z],
                        # N010
                        [-inv_dx_x * ny * cnz,   cnx *  inv_dx_y * cnz,  cnx * ny  * -inv_dx_z],
                        # N011
                        [-inv_dx_x * ny * nz,    cnx *  inv_dx_y * nz,   cnx * ny  *  inv_dx_z],
                        # N100
                        [ inv_dx_x * cny * cnz,  nx  * -inv_dx_y * cnz,  nx  * cny * -inv_dx_z],
                        # N101
                        [ inv_dx_x * cny * nz,   nx  * -inv_dx_y * nz,   nx  * cny *  inv_dx_z],
                        # N110
                        [ inv_dx_x * ny * cnz,   nx  *  inv_dx_y * cnz,  nx  * ny  * -inv_dx_z],
                        # N111
                        [ inv_dx_x * ny * nz,    nx  *  inv_dx_y * nz,   nx  * ny  *  inv_dx_z],
                    ]
                )

            N_q_e = torch.tensor(N_q_e, dtype=torch.float64)          # (Q=8, Ne=8)
            gradN_q_e = torch.tensor(gradN_q_e, dtype=torch.float64)  # (Q=8, Ne=8, 3)

            # Element sample volume (same for all elements if dx is global)
            vol_e = hx * hy * hz
            sample_vol = vol_e / 8.0
            element_sample_volume_e = torch.full((8, 1), sample_vol, dtype=torch.float64)

            # Replicate for all elements
            N_q = [N_q_e for _ in range(E)]
            gradN_q = [gradN_q_e for _ in range(E)]
            element_sample_volume = [element_sample_volume_e for _ in range(E)]

            self.register_buffer('N_q', torch.stack(N_q, dim=0))                     # (E, Q=8, Ne=8)
            self.register_buffer('gradN_q', torch.stack(gradN_q, dim=0))             # (E, Q=8, Ne=8, 3)
            self.register_buffer('element_sample_volume',
                                torch.stack(element_sample_volume, dim=0))          # (E, Q=8, 1)

        elif Ne == 4:

            Q = 1

            print("Tet")

            self._mesh_type = 'tet'

            N_q = []                    # Shape function N
            gradN_q = []                # Shape function gradient ∇N
            element_sample_volume = []  # Element sample volume
            for e in range(E):

                N_q.append(torch.tensor(4 * [0.25], dtype=torch.float64))  # (Ne)

                X0 = X_e[3 * self.elements[e][0]:3 * self.elements[e][0] + 3]
                X1 = X_e[3 * self.elements[e][1]:3 * self.elements[e][1] + 3]
                X2 = X_e[3 * self.elements[e][2]:3 * self.elements[e][2] + 3]
                X3 = X_e[3 * self.elements[e][3]:3 * self.elements[e][3] + 3]

                X0 = torch.as_tensor(X0, dtype=torch.float64)
                X1 = torch.as_tensor(X1, dtype=torch.float64)
                X2 = torch.as_tensor(X2, dtype=torch.float64)
                X3 = torch.as_tensor(X3, dtype=torch.float64)

                J = torch.stack([X1 - X0, X2 - X0, X3 - X0], dim=1)              # (3,3)
                B = torch.linalg.inv(J)                                          # (3,3)
                gradN_q.append(torch.stack(
                    [-B[:, 0] - B[:, 1] - B[:, 2], B[:, 0], B[:, 1], B[:, 2]]
                ))                                                                # (Ne,3)

                vol_e = torch.abs(torch.det(J)) / 6.0                            # scalar
                element_sample_volume.append(vol_e.reshape(1))                   # (1)

            self.register_buffer('N_q', torch.stack(N_q, dim=0).unsqueeze(1))                 # (E, Q=1, Ne)
            self.register_buffer('gradN_q', torch.stack(gradN_q, dim=0).unsqueeze(1))         # (E, Q=1, Ne,3)
            self.register_buffer('element_sample_volume',
                                torch.stack(element_sample_volume, dim=0).unsqueeze(1))      # (E, Q=1, 1)

        else:
            # Current implementation only supports hex and tet mesh
            exit()
        

        
        q0 = self.q0.reshape(-1,3)
        Qf = q0[self.faces, :]
        face_positions = q0[self.faces, :].mean(1)
        element_positions = torch.einsum('eni,eqn->eqi', q0[self.elements, :], self.N_q) 

        F = self.faces.shape[0]
        k = 3

        if self._mesh_type == 'hex':
            # Collect Triangles
            tris = [(0,1,2), (0,2,3), (0,1,3), (3,1,2)]
            unnormalized_n = torch.zeros((F, 3), dtype=q0.dtype, device=q0.device)     # (F,3)
            A_sum = torch.zeros((F, 1), dtype=q0.dtype, device=q0.device)              # (F,1)
            for a, b, c in tris:
                p0, p1, p2 = Qf[:, a, :], Qf[:, b, :], Qf[:, c, :]             # (F,3)
                cp = torch.cross(p1 - p0, p2 - p1, dim=-1)                              # (F,3)
                unnormalized_n += cp                                                    # (F,3)
                A_sum += cp.norm(dim=-1, keepdim=True)                                  # (F,1)

            # Average area and norm over triangles
            A_f = A_sum / 4.0                                                           # (F,1)
            n_f = unnormalized_n / unnormalized_n.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # (F,3)


        dists = torch.cdist(element_positions.reshape(-1,3), face_positions, p=2)                 # (E*Q,F)

        alignment = torch.sum(n_f.reshape(1,F,3) * (face_positions.reshape(1,F,3) - element_positions.reshape(E*Q,1,3)), dim=-1) # (E*Q,F)

        aligned_dists = torch.where(alignment > 0, dists, float('inf')) # (E*Q,F)

        min_dist_face, min_dist_face_idx = torch.topk(aligned_dists, largest=False, k =k, dim=1)        # (E*Q,k)                          # (B,E*Q)

        idx = min_dist_face_idx.reshape(E*Q*k,1).expand(-1, 3)                                   # (E*Q*k,3)

        target_face_normals = torch.gather(n_f, dim=0, index=idx)               # (E*Q*k,3)
        target_face_positions = torch.gather(face_positions, dim=0, index=idx)  # (E*Q*k,3)

        diff_to_face = target_face_positions.reshape(E*Q,k,3) - element_positions.reshape(-1,1,3)                # (E*Q,k,3)
        dist_to_face = torch.sum(diff_to_face * target_face_normals.reshape(E*Q,k,3), dim =-1)       # (E*Q,k)

        mean_edge = (Qf[:, [0,1,2,3], :] - Qf[:, [1,2,3,0], :]).norm(dim=-1).mean() # (B,)
        tau = 0.5 * mean_edge.clamp_min(1e-12)

        w = torch.softmax(dist_to_face / tau, dim=1)             # (E*Q,k)

        # dir_to_face = (w.unsqueeze(-1) * target_face_normals.reshape(E*Q,k,3)).sum(dim=1)        # (E*Q,3)
        # dir_to_face = dir_to_face / dir_to_face.norm(dim=-1, keepdim=True).clamp_min(1e-12) 
        # dist_to_face = (w * dist_to_face).sum(dim=1, keepdim=True) # (E*Q,1)

        self.register_buffer('element_q0', element_positions)
        # self.register_buffer('dist2face', dist_to_face.reshape(E,Q,1))
        # self.register_buffer('dir2face', dir_to_face.reshape(E,Q,3))

        face2face_offset = face_positions.reshape(1,F,3) - face_positions.reshape(F,1,3) #(F,F,3)
        face2face_dist = torch.sum(-n_f.reshape(F,1,3) * face2face_offset, dim=-1)  #(F,F)
        face2face_offsetPerp = torch.linalg.norm(face2face_offset + face2face_dist.reshape(F,F,1) * n_f.reshape(F,1,3), dim=-1)  #(F,F)
        
        forward = face2face_dist > 0

        lateral_ok = face2face_offsetPerp <= torch.sqrt(A_f).max(dim=0, keepdim=True)[0]

        facing = (-n_f.reshape(F,1,3) * n_f.reshape(1,F,3)).sum(dim=-1) > 0.5 #(F,F)

        not_self = ~(torch.eye(F, dtype=torch.bool, device=q0.device)) #(F,F)

        mask =  forward & lateral_ok & facing & not_self #(F,F)

        valid_face2face_dist = torch.where(mask, face2face_dist, torch.full_like(face2face_dist, float("inf"))) #(F,F)

        max_depth_per_face = valid_face2face_dist.min(dim=-1)[0] #(F)

        self.register_buffer('max_depth', max_depth_per_face)

        target_max_depth = torch.gather(max_depth_per_face, dim=0, index=min_dist_face_idx.reshape(E*Q*k)).reshape(E*Q,k)
    
        dist_to_face = torch.sum(diff_to_face * target_face_normals.reshape(E*Q,k,3), dim =-1)       # (E*Q,k)
        depth_from_face = dist_to_face / target_max_depth #(E*Q,k)

        filtered_depth_from_face = torch.where(torch.abs(depth_from_face) <= 0.5, depth_from_face, torch.abs(depth_from_face - 1))

        # offsets_to_faces = filtered_depth_from_face.reshape(E*Q,k,1) * target_face_normals.reshape(E*Q,k,3)
        # offset_to_face = torch.gather(offsets_to_faces, dim=1, index=offsets_to_faces.abs().argmax(dim=1, keepdim=True)).squeeze(1)
        

        filtered_depth_from_face = torch.where(torch.abs(depth_from_face) <= 0.5, depth_from_face, torch.abs(1 - depth_from_face))
        filtered_target_face_normals = torch.where(torch.abs(depth_from_face.unsqueeze(-1)) <= 0.5, target_face_normals.reshape(E*Q,k,3), -target_face_normals.reshape(E*Q,k,3))
        filtered_dist_to_face = torch.where(torch.abs(depth_from_face) <= 0.5, dist_to_face, target_max_depth - dist_to_face)

        mean_edge = (Qf[:, [0,1,2,3], :] - Qf[:, [1,2,3,0], :]).norm(dim=-1).mean() # (B,)
        tau = 0.5 * mean_edge.clamp_min(1e-12)

        #w = torch.softmax(filtered_dist_to_face / tau, dim=1)             # (E*Q,k)
        w = torch.softmax(-filtered_depth_from_face / 0.1, dim=1)              # (E*Q,k)

        # print(w.reshape(E,Q,k))
        # exit()
        dir_to_face = (w.unsqueeze(-1) * filtered_target_face_normals.reshape(E*Q,k,3)).sum(dim=1)        # (E*Q,3)
        dir_to_face = dir_to_face / dir_to_face.norm(dim=-1, keepdim=True).clamp_min(1e-12) 
        # dist_to_face = (w * filtered_dist_to_face).sum(dim=1, keepdim=True) # (E*Q,1)
        # max_dist_to_face = (w * target_max_depth).sum(dim=1, keepdim=True) # (E*Q,1)
        norm_dist_to_face = (w.unsqueeze(-1) * filtered_depth_from_face.unsqueeze(-1)).sum(dim=1)        # (E*Q,1)
        #norm_dist_to_face = dist_to_face / max_dist_to_face        # (E*Q,1)

        # # print(depth_from_face.reshape(E,Q,k))
        # # print(target_face_normals.reshape(E*Q,k,3))
        # # print(dist_to_face.reshape(E,Q))
        # # print(norm_dist_to_face.reshape(E,Q))
        # # print(norm_dist_to_face.max())
        # # print(norm_dist_to_face.min())
        offset_to_face = norm_dist_to_face * dir_to_face # (E*Q,3)
        #self.register_buffer('dist2face', dist_to_face.reshape(E,Q,1))
        # self.register_buffer('dir2face', dir_to_face.reshape(E,Q,3))
        self.register_buffer('offset2face', offset_to_face.reshape(E,Q,3))

        # print(self.offset2face.reshape(E,Q,3))
        print(offset_to_face.max(dim=0)[0])
        print(offset_to_face.min(dim=0)[0])
        # print(offset_to_face[torch.where(torch.abs(offset_to_face).sum(dim=1) == 0)[0]])
        # print(dist_to_face[torch.where(torch.abs(offset_to_face).sum(dim=1) == 0)[0],:])
        # print(target_max_depth[torch.where(torch.abs(offset_to_face).sum(dim=1) == 0)[0],:])
        # print(depth_from_face[torch.where(torch.abs(offset_to_face).sum(dim=1) == 0)[0],:])
        # print(target_face_normals[torch.where(torch.abs(offset_to_face).sum(dim=1) == 0)[0],:])
        # print(offsets_to_faces[torch.where(torch.abs(offset_to_face).sum(dim=1) == 0)[0],:])
        #exit()


    def to_elements(self, q):
        # q: (B,V,3) -> (B,E,Ne,3)
        return q[:, self.elements, :]
    
    def to_faces(self, q):
        # q: (B,V,3) -> (B,F,Nf,3)
        return q[:, self.faces, :]

    def element_deformationGrad(self, q):
        q_e = self.to_elements(q)                                               # (B,E,Ne,3)
        
        # q_e: (B,E,Ne,3), gradN_q: (E,Q,Ne,3) -> F: (B,E,Q,3,3)
        F_e = torch.einsum('beni,eqnj->beqij', q_e, self.gradN_q.to(q_e.dtype)) # (B,E,Q,3,3)

        return F_e

    def sigma_R_from_F(self, F, eps=1e-12):
        
        C = F.transpose(-1,-2) @ F

        evals, V = torch.linalg.eigh(C)                

        sigma = evals.sqrt()              

        inv_sigma = torch.where(sigma > eps, 1.0 / sigma, torch.zeros_like(sigma))
        VinvS = V * inv_sigma.unsqueeze(-2)            
        Sinv = VinvS @ V.transpose(-1,-2)                            

        R = F @ Sinv

        detR = torch.linalg.det(R)
        need_flip = detR < 0
        
        if need_flip.any():
            Ur, _, Vtr = torch.linalg.svd(R)           
            M = Ur @ Vtr                               
            s = torch.sign(torch.det(M))[..., None, None]
            Ur_fixed = torch.cat([Ur[..., :, :2], Ur[..., :, 2:3] * s], dim=-1)
            R = Ur_fixed @ Vtr                         

            detR = torch.linalg.det(R)
            assert torch.all(detR > 0)

        return sigma, R

    def element_velocityGrad(self, q, v, eps_rel=1e-6, eps_abs=1e-12):
        v_e = self.to_elements(v)                                                           # (B,E,Ne,3)
        F = self.element_deformationGrad(q)                                                 # (B,E,Q,3,3)

        # v_e: (B,E,Ne,3), gradN_q: (E,Q,Ne,3) -> grad_v_e_T: (B,E,Q,3,3)
        grad_v_e_T = torch.einsum('beni,eqnj->beqji', v_e, self.gradN_q.to(v_e.dtype))      # (B,E,Q,3,3)

        # F_T: (B,E,Q,3,3), L_T: (B,E,Q,3,3) -> grad_v_e_T: (B,E,Q,3,3)
        try:
            L_T = torch.linalg.solve(F.transpose(-1,-2), grad_v_e_T)                        # (B,E,Q,3,3)
        except:
            n = F.shape[-1]
            I = torch.eye(n, dtype=F.dtype, device=F.device).expand(*F.shape[:-2], n, n)
            noise = (eps_abs) * I
            L_T = torch.linalg.solve(F.transpose(-1,-2) + noise, grad_v_e_T)                # (B,E,Q,3,3)
        L = L_T.transpose(-1,-2)                                                            # (B,E,Q,3,3)

        D = 0.5 * (L + L.transpose(-1,-2))                                                  # (B,E,Q,3,3)
        W = 0.5 * (L - L.transpose(-1,-2))                                                  # (B,E,Q,3,3)

        return D, W
    
    def element_geom(self, q, v, m, r_mult=2, eps = 1e-12):

        q_e = self.to_elements(q)                                                           # (B,E,Ne,3)
        v_e = self.to_elements(v)                                                           # (B,E,Ne,3)
        B,E,Ne,D = q_e.shape
        Q = self.N_q.shape[1]

        if self.changing_boundary_indices is not None:
            v_boundary = v.reshape(B,-1)[:,self.changing_boundary_indices].reshape(B,-1,3)              # (B,V,3)
            v_base = v_boundary.mean(dim=1, keepdim=True).unsqueeze(1)                                  # (B,1,1,3)

            q_boundary = q.reshape(B,-1)[:,self.changing_boundary_indices].reshape(B,-1,3)              # (B,V,3)
            q_base = self.q0[self.changing_boundary_indices].reshape(1,-1,3)                            # (1,V,3)
            q_offset = (q_boundary - q_base).mean(dim=1, keepdim=True).unsqueeze(1)                     # (B,1,1,3)
        else:
            v_base = torch.zeros((B,1,1,3), device=v_e.device, dtype=v_e.dtype)                         # (B,1,1,3)
            q_offset = torch.zeros((B,1,1,3), device=v_e.device, dtype=v_e.dtype)                       # (B,1,1,3)

        # print(v_base)
        # print(q_offset)

        # Interpolate element nodes to samples
        # *_e: (B,E,Ne,*), N_q: (E,Q,Ne) -> q_e: (B,E,Q,*)
        q_e = torch.einsum('beni,eqn->beqi', q_e, self.N_q.to(q_e.dtype))                   # (B,E,Q,3)
        v_e = torch.einsum('beni,eqn->beqi', v_e - v_base, self.N_q.to(v_e.dtype))          # (B,E,Q,3)


        # dir2face = self.dir2face.unsqueeze(0).expand(B,E,Q,3)                                    # (B,E,Q,3)


        # # Element Radii
        # V_e = self.element_sample_volume.unsqueeze(0)                                               # (1,E,Q,1)
        # h_e = (6.0 * V_e.sum(dim=2)).clamp_min(eps).pow(1.0 / 3.0)                                  # (1,E,1)
        # r_e = (r_mult * h_e).unsqueeze(-1)                                                          # (1,E,1,1)


        # # Mass of Intersection slice element samples
        # q0 = self.element_q0.reshape(E*Q,1,3) #(E*Q,1,3)
        # all_q0 = self.element_q0.reshape(E*Q,1,3) #(E*Q,1,3)
        # all_target_q0 = self.element_q0.reshape(1,E*Q,3) #(1,E*Q,3)
        # q0_2_q0 = (all_target_q0 - all_q0).reshape(1,E,Q,E*Q,3) #(E,Q,E*Q,3)
        # q_rel_cross_srf = torch.linalg.norm(torch.cross(dir2face.reshape(B,E,Q,1,3), q0_2_q0, dim=-1), dim=-1)   #(B,E,Q,E*Q)
        # m_slice = torch.where(q_rel_cross_srf < r_e, m.reshape(B,1,1,E*Q), 0).unsqueeze(-1)           #(B,E,Q,E*Q,1)
        # M = m_slice.sum(dim=3) #(B,E,Q,1)

        # q_sum = (m_slice * q_e.reshape(B,1,1,E*Q,3)).sum(dim=3)                  #(B,E,Q,3)
        # q_cm = q_sum / M                                                         #(B,E,Q,3)
        # v_sum = (m_slice * v_e.reshape(B,1,1,E*Q,3)).sum(dim=3)                  #(B,E,Q,3)
        # v_cm = v_sum / M                                                         #(B,E,Q,3)

        # Center of Mass using element densities
        M = m.sum(dim=[1,2]).reshape(B,1,1,1)                                               # (B,1,1,1)
        q_sum = (m * q_e).sum(dim=[1,2]).reshape(B,1,1,3)                                   # (B,1,1,3)
        v_sum = (m * v_e).sum(dim=[1,2]).reshape(B,1,1,3)                                   # (B,1,1,3)
        q_cm = q_sum / M - q_offset                                                         # (B,1,1,3)
        v_cm = v_sum / M                                                                    # (B,1,1,3)


        # Relative position and velocity wrt Slice
        q_rel = q_e - q_cm                                                                  # (B,E,Q,3)
        v_rel = v_e - v_cm                                                                  # (B,E,Q,3)

        r = q_rel.norm(dim=-1, keepdim=True)
        rhat = q_rel / torch.clamp_min(r, eps)
        v_rad = (v_rel * rhat).sum(dim=-1, keepdim=True) * rhat

        v_perp = v_rel - v_rad

        sL_rel = torch.cross(q_rel, v_rel, dim=-1)                                          # (B,E,Q,3)

        L_cm = (m * torch.cross(q_rel, v_rel, dim=-1)).sum(dim=[1,2])                   # (B,3)

        # Compute per-particle inertia contribution
        r2 = (q_rel * q_rel).sum(dim=-1, keepdim=True).reshape(B,E,Q,1,1)                # (B,E,Q,1,1)
        eye = torch.eye(3, device=q_rel.device, dtype=q_rel.dtype).view(1,1,1,3,3)
        I_per = m.reshape(B,E,Q,1,1) * (r2 * eye - q_rel.reshape(B,E,Q,3,1) * q_rel.reshape(B,E,Q,1,3)) # (B,E,Q,3,3)
        I_cm = I_per.sum(dim=[1,2])  # (B,3,3)

        Omega_cm = torch.linalg.solve(I_cm, L_cm.reshape(B,3,1)).reshape(B,1,1,3)


        return q_rel, v_e, v_rel, v_rad, v_perp, sL_rel, q_cm, v_cm.expand(B,E,Q,3), Omega_cm.expand(B,E,Q,3)


    def offset_to_faces(self, q, k=3):

        q_e = self.to_elements(q)                                                                   # (B,E,Ne,3)
        q_f = self.to_faces(q)                                                                      # (B,F,Nf,3)

        B,E,Ne,D = q_e.shape
        _, F, _, _ = q_f.shape
        Q = self.N_q.shape[1]

        face_positions = q_f.mean(2)                                                                # (B,F,3)
        element_positions = torch.einsum('beni,eqn->beqi', q_e, self.N_q)                           # (B,E,Q,3)


        if self._mesh_type == 'hex':
            # Collect Triangles
            tris = [(0,1,2), (0,2,3), (0,1,3), (3,1,2)]
            unnormalized_n = torch.zeros((B, F, 3), dtype=q.dtype, device=q.device)     # (B,F,3)
            A_sum = torch.zeros((B, F, 1), dtype=q.dtype, device=q.device)              # (B,F,1)
            for a, b, c in tris:
                p0, p1, p2 = q_f[:, :, a, :], q_f[:, :, b, :], q_f[:, :, c, :]             # (B,F,3)
                cp = torch.cross(p1 - p0, p2 - p1, dim=-1)                              # (B,F,3)
                unnormalized_n += cp                                                    # (B,F,3)
                A_sum += cp.norm(dim=-1, keepdim=True)                                  # (B,F,1)

            # Average area and norm over triangles
            A_f = A_sum / 4.0                                                           # (B,F,1)
            n_f = unnormalized_n / unnormalized_n.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # (B,F,3)

        dists = torch.cdist(element_positions.reshape(B,-1,3), face_positions, p=2)                 # (B,E*Q,F)

        alignment = torch.sum(n_f.reshape(B,1,F,3) * (face_positions.reshape(B,1,F,3) - element_positions.reshape(B,-1,1,3)), dim=-1) # (B,E*Q,F)

        aligned_dists = torch.where(alignment > 0, dists, float('inf')) # (B,E*Q,F)

        _, min_dist_face_idx = torch.topk(aligned_dists, k, largest=False, dim=2)                           # (B,E*Q,k)                          # (B,E*Q)

        idx = min_dist_face_idx.reshape(B,E*Q*k,1).expand(-1, -1, 3)                                   # (B,E*Q*k,3)


        # face2face_offset = face_positions.reshape(B,1,F,3) - face_positions.reshape(B,F,1,3) #(B,F,F,3)
        # face2face_dist = torch.sum(-n_f.reshape(B,F,1,3) * face2face_offset, dim=-1)  #(B,F,F)
        # face2face_offsetPerp = torch.linalg.norm(face2face_offset + face2face_dist.reshape(B,F,F,1) * n_f.reshape(B,F,1,3), dim=-1)  #(B,F,F)
        
        # forward = face2face_dist > 0

        # lateral_ok = face2face_offsetPerp <= torch.sqrt(A_f).max(dim=1, keepdim=True)[0]

        # facing = (-n_f.reshape(B,F,1,3) * n_f.reshape(B,1,F,3)).sum(dim=-1) > 0.5 #(F,F)

        # not_self = ~(torch.eye(F, dtype=torch.bool, device=q.device)).reshape(1,F,F) #(1,F,F)

        # mask =  forward & lateral_ok & facing & not_self #(B,F,F)

        # valid_face2face_dist = torch.where(mask, face2face_dist, torch.full_like(face2face_dist, float("inf"))) #(B,F,F)

        # max_depth_per_face = valid_face2face_dist.min(dim=-1)[0] #(B,F)

        # print(max_depth_per_face)

        max_depth_per_face = self.max_depth.reshape(1,F).expand(B,F)
        # print(orig_max_depth_per_face)

        target_max_depth = torch.gather(max_depth_per_face, dim=1, index=min_dist_face_idx.reshape(B,E*Q*k)).reshape(B,E*Q,k)
        target_face_normals = torch.gather(n_f, dim=1, index=idx).reshape(B,E*Q,k,3)                # (B,E*Q,k,3)
        target_face_positions = torch.gather(face_positions, dim=1, index=idx).reshape(B,E*Q,k,3)   # (B,E*Q,k,3)

        diff_to_face = target_face_positions - element_positions.reshape(B,-1,1,3)                # (B,E*Q,k,3)
        dist_to_face = torch.sum(diff_to_face * target_face_normals, dim =-1)       # (B,E*Q,k)
        depth_from_face = dist_to_face / target_max_depth #(B,E*Q,k)


        # filtered_depth_from_face = torch.where(torch.abs(depth_from_face) <= 0.5, depth_from_face, torch.abs(depth_from_face - 1))

        # offsets_to_faces = filtered_depth_from_face.reshape(B,E*Q,k,1) * target_face_normals.reshape(B,E*Q,k,3)
        # norm_offset_to_face = torch.gather(offsets_to_faces, dim=2, index=offsets_to_faces.abs().argmax(dim=2, keepdim=True)).squeeze(2)

        norm_offset_to_face = (depth_from_face.reshape(B,E*Q,k,1) * target_face_normals.reshape(B,E*Q,k,3)).max(dim=2)[0]
        mean_edge = (q_f[:, :, [0,1,2,3], :] - q_f[:, :, [1,2,3,0], :]).norm(dim=-1).mean(dim=[1,2]) # (B,)

        filtered_depth_from_face = torch.where(torch.abs(depth_from_face) <= 0.5, depth_from_face, torch.abs(1 - depth_from_face))
        filtered_target_face_normals = torch.where(torch.abs(depth_from_face.unsqueeze(-1)) <= 0.5, target_face_normals, -target_face_normals)


        mean_edge = (q_f[:, :, [0,1,2,3], :] - q_f[:, :, [1,2,3,0], :]).norm(dim=-1).mean(dim=[1,2]) # (B,)
        tau = 0.5 * mean_edge.clamp_min(1e-12)
        w = torch.softmax(-dist_to_face / tau.reshape(B,1,1), dim=2)              # (B,E*Q,k)

        # w = torch.softmax(-filtered_depth_from_face/0.1, dim=2)              # (B,E*Q,k)

        dir_to_face = (w.unsqueeze(-1) * filtered_target_face_normals).sum(dim=2)        # (B,E*Q,3)
        dir_to_face = dir_to_face / dir_to_face.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        dist_to_face = (w.unsqueeze(-1) * dist_to_face.unsqueeze(-1)).sum(dim=2)        # (B,E*Q,1)
        max_dist_to_face = (w.unsqueeze(-1) * target_max_depth.unsqueeze(-1)).sum(dim=2)
        # offset_to_face = dist_to_face * dir_to_face # (B,E*Q,3)
        # depth_from_face = (w.unsqueeze(-1) * filtered_depth_from_face.unsqueeze(-1)).sum(dim=2)        # (B,E*Q,1)
        # max_depth_from_face = depth_from_face.max(dim=1, keepdim=True)[0]
        # norm_dist_to_face = depth_from_face / max_depth_from_face
        max_dist_to_face, max_dist_idx = dist_to_face.max(dim=1,keepdim=True)
        norm_dist_to_face = dist_to_face / max_dist_to_face
        norm_offset_to_face = norm_dist_to_face * dir_to_face # (B,E*Q,3)

        # max_depth_per_face = self.max_depth.reshape(1,F,1).expand(B,F,1)
        # target_max_depth = torch.gather(max_depth_per_face, dim=1, index=min_dist_face_idx.reshape(B,E*Q*k,1)).reshape(B,E*Q,k,1)
        # max_depth = (w.unsqueeze(-1) * target_max_depth).sum(dim=2)

        # norm_dist_to_face = dist_to_face / max_depth
        # norm_offset_to_face = norm_dist_to_face * dir_to_face # (B,E*Q,3)

        # print(norm_dist_to_face.reshape(-1,E,Q))
        # print(norm_dist_to_face.max(dim=1)[0])

        # max_depth_per_face = self.max_depth.reshape(1,F).expand(B,F)
        # target_max_depth = torch.gather(max_depth_per_face, dim=1, index=min_dist_face_idx.reshape(B,E*Q*k)).reshape(B,E*Q,k)
        # target_face_normals = torch.gather(n_f, dim=1, index=idx).reshape(B,E*Q,k,3)                # (B,E*Q,k,3)
        # target_face_positions = torch.gather(face_positions, dim=1, index=idx).reshape(B,E*Q,k,3)   # (B,E*Q,k,3)

        # diff_to_face = target_face_positions - element_positions.reshape(B,-1,1,3)                # (B,E*Q,k,3)
        # dist_to_face = torch.sum(diff_to_face * target_face_normals, dim =-1)       # (B,E*Q,k)
        # depth_from_face = dist_to_face / target_max_depth #(B,E*Q,k)

        # mean_edge = (q_f[:, :, [0,1,2,3], :] - q_f[:, :, [1,2,3,0], :]).norm(dim=-1).mean(dim=[1,2]) # (B,)


        # w = torch.softmax(- 2 * depth_from_face, dim=2)              # (B,E*Q,k)

        # dir_to_face = (w.unsqueeze(-1) * target_face_normals).sum(dim=2)        # (B,E*Q,3)
        # dir_to_face = dir_to_face / dir_to_face.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        # # dist_to_face = (w.unsqueeze(-1) * dist_to_face.unsqueeze(-1)).sum(dim=2)        # (B,E*Q,1)
        # # offset_to_face = dist_to_face * dir_to_face # (B,E*Q,3)
        # norm_dist_to_face = (w.unsqueeze(-1) * depth_from_face.unsqueeze(-1)).sum(dim=2)        # (B,E*Q,1)

        # # max_dist_to_face, max_dist_idx = dist_to_face.max(dim=1,keepdim=True)
        # # norm_dist_to_face = dist_to_face / max_dist_to_face
        # # norm_offset_to_face = norm_dist_to_face * dir_to_face # (B,E*Q,3)

        # # max_depth_per_face = self.max_depth.reshape(1,F,1).expand(B,F,1)
        # # target_max_depth = torch.gather(max_depth_per_face, dim=1, index=min_dist_face_idx.reshape(B,E*Q*k,1)).reshape(B,E*Q,k,1)
        # # max_depth = (w.unsqueeze(-1) * target_max_depth).sum(dim=2)

        # # norm_dist_to_face = dist_to_face / max_depth
        # norm_offset_to_face = norm_dist_to_face * dir_to_face # (B,E*Q,3)

        # print(norm_dist_to_face.reshape(B,E,Q,1))
        # print(norm_offset_to_face.reshape(B,E,Q,3))
        # exit()

        return norm_offset_to_face.reshape(B,E,Q,3) #, dir_to_face.reshape(B,E,Q,3), norm_dist_to_face.reshape(B,E,Q,1)
        
        # B,_,_ = q.shape
        # E,Q,_ = self.N_q.shape

        # orig_dist_to_face = self.dist2face.reshape(1,E*Q,1).expand(B,E*Q,1)                           # (B,E*Q,1)
        # orig_dir_to_face = self.dir2face.reshape(1,E*Q,3).expand(B,E*Q,3)                                # (B,E*Q,3)
    
        # orig_offset_to_face = orig_dist_to_face * orig_dir_to_face      # (B,E*Q,3)

        # max_dist_to_face = (torch.sum(self.dir2face.reshape(E*Q,1,3) * self.dist2face.reshape(1,E*Q,1) * self.dir2face.reshape(1,E*Q,3), dim=-1)).max(dim=1)[0].reshape(1,E*Q,1)

        # norm_dist_to_face = orig_dist_to_face / max_dist_to_face 
        # norm_offset_to_face = orig_offset_to_face / max_dist_to_face                                   # (B,E*Q,3)

        # return norm_offset_to_face.reshape(B,E,Q,3), orig_dir_to_face.reshape(B,E,Q,3), norm_dist_to_face.reshape(B,E,Q,1)


    @staticmethod
    def _wendland_c2(q):
        # q >= 0; compact support for q < 1
        m = (1.0 - q).clamp(min=0.0, max=1.0)
        return (m ** 4) * (1.0 + 4.0 * q)


    def force_geom(self, q, f, m, q_cm, k = 1, r_mult = 2, eps = 1e-12):

        # Element Sample Positions
        q_e = self.to_elements(q)                                                                   # (B,E,Ne,3)
        q_e = torch.einsum('beni,eqn->beqi', q_e, self.N_q.to(q.dtype))                             # (B,E,Q,3)
        B, E, Q, _ = q_e.shape

        # Element Radii
        V_e = self.element_sample_volume.unsqueeze(0)                                               # (1,E,Q,1)
        h_e = (6.0 * V_e.sum(dim=2, keepdim=True)).clamp_min(eps).pow(1.0 / 3.0)                    # (1,E,1,1)
        r_e = (r_mult * h_e).expand(B, E, Q, 1)                                                     # (B,E,Q,1)
        #print(h_e.squeeze())

        # Element Sample Distance to closest Forced Nodes
        P  = q_e.reshape(B, E * Q, 3)                                                               # (B,E*Q,3) 
        dists = torch.cdist(P, q, p=2)                                                              # (B,E*Q,V)
        kmin_dists, kmin_idx = torch.topk(dists, k=k, dim=2, largest=False)                         # (B,E*Q,k)
        
        # Scale Sample Distance by Radii
        rb = r_e.reshape(B, E * Q, 1)                                                               # (B,E*Q,1)
        qn = kmin_dists / rb.clamp_min(eps)                                                              # (B,E*Q,V)

        # Weigh the force by scaled distance
        wi = torch.where(kmin_dists <= rb, self._wendland_c2(qn), torch.zeros_like(qn))                  # (B,E*Q,V)
        phi = wi / wi.sum(dim=2, keepdim=True).clamp_min(eps)                                       # (B,E*Q,V)
        _, V, _ = f.shape
        f_expanded = f.unsqueeze(1).expand(B,E*Q,V,3)                                               # (B,E*Q,V,3)
        f_neighbors = torch.gather(f_expanded, 2, kmin_idx.unsqueeze(-1).expand(B, E*Q, k, 3))      # (B,E*Q,k,3)
        f_q = torch.einsum('bmf,bmfi->bmi', phi, f_neighbors)                                       # (B,E*Q,3)
        #f_q = torch.einsum('bmf,bmfi->bmi', phi, f_expanded)                                        # (B,E*Q,3)

        # Add gravitational force
        f_e = f_q.view(B, E, Q, 3) + m * self.g.reshape(1,1,1,3)                                    # (B,E,Q,3)

        rho = self.rho.reshape(1,E,1,1)                                                             # (1,E,1,1)                                                                       
        M = (rho * V_e).sum(dim=[1,2])                                                              # (1,1)
        f_total = f.sum(dim=1) + M * self.g.reshape(1,3)                                            # (B,3)

        # torque per element due to external forces
        t_e = torch.cross(q_e - q_cm, f_e, dim=-1)                                                  # (B,E,Q,3)

        # Net torque around COM due to external forces
        # t_total = torch.cross(q - q_cm.reshape(B,1,3), f, dim=-1).sum(dim=1)                        # (B,3)

        return f_e, t_e, f_total.reshape(B,1,1,3).expand(B,E,Q,3)#, t_total.reshape(B,1,1,3).expand(B,E,Q,3)

    def elementForce_to_nodeForce(self, f_e):
        # f_e: (B,E,Q,3), N_q: (E,Q,Ne)  -> fn: (B,E,Ne,3)
        fn = torch.einsum('beqi,eqn->beni', f_e, self.N_q.to(f_e.dtype))    # (B,E,Ne,3)
        return fn    

    def elementStress_to_nodeForce(self, P_e):
        # P_e: (B,E,Q,3,3), gradN_q: (E,Q,Ne,3), V_q: (E,Q,1)  -> fn: (B,E,Ne,3)

        fn = -torch.einsum('beqij,eqnj,eqk->beni', P_e, self.gradN_q.to(P_e.dtype), self.element_sample_volume.to(P_e.dtype))
        return fn

    def elementNode_to_dof(self, fe_n):  
        B, E, Ne, D = fe_n.shape
        V = self.dof // D

        # element node to vertex idx mapping for all batches, elements, nodes
        # idx_flat[i] = batch_num * V + vertex_idx
        idx = self.elements.unsqueeze(0).expand(B, E, Ne)                       # (B,E,Ne)
        offsets = (torch.arange(B, device=fe_n.device) * V).view(B, 1, 1)       # (B,1,1)
        idx_flat = (idx + offsets).reshape(-1)                                  # (B*E*Ne,)

        # forces at each element node
        # fe_n_flat[i] = force at batch_num, element_idx, node_idx
        fe_n_flat = fe_n.reshape(B * E * Ne, D)                                 # (B*E*Ne,3)

        # Accumulate fe_n_flat[i] into f_flat[idx_flat[i]] for all i
        # i = batch_num * E * Ne + element_idx * Ne + node_idx
        f_flat = torch.zeros((B * V, D), dtype=fe_n.dtype, device=fe_n.device)  # (B*V,3)
        f_flat.index_add_(0, idx_flat, fe_n_flat)                               # (B*V,3)

        return f_flat.view(B, V, D).reshape(B, self.dof)    
    
    def force_unmodelled(self, q, v, f):
        I = torch.eye(3, device=q.device, dtype=q.dtype)                            # (1,1,1,3,3)


        F_e = self.element_deformationGrad(q)                                       # (B,E,Q,3,3)
        B,E,Q,_,_ = F_e.shape


        # U, sigma, Vh = torch.linalg.svd(F_e)                                      # (B,E,Q,3,3),(B,E,Q,3),(B,E,Q,3,3)
        # R = U @ Vh                                                                # (B,E,Q,3,3)
        sigma, R = self.sigma_R_from_F(F_e)
        Rt = R.transpose(-1, -2)                                                    # (B,E,Q,3,3)


        I1_F = sigma - 1.0                                                          # (B,E,Q,3)
        Cmat = F_e.transpose(-1,-2) @ F_e                                           # (B,E,Q,3,3)
        I2_F  = (Cmat -  I).reshape(B,E,Q,9)                                        # (B,E,Q,9)
        J = torch.linalg.det(F_e).unsqueeze(-1)                                     # (B,E,Q,1)
        I3_F = J - 1.0                                                              # (B,E,Q,1)
        
        D, W = self.element_velocityGrad(q,v)                                       # (B,E,Q,3,3)
        Dc = Rt @ D @ R                                                             # (B,E,Q,3,3)
        Wc = Rt @ W @ R                                                             # (B,E,Q,3,3)

        I1_D = Dc.diagonal(dim1=-2, dim2=-1)                                        # (B,E,Q,3)
        I2_D = I1_D.sum(dim=-1, keepdim=True)                                       # (B,E,Q,1)
        D_dev = Dc - (I2_D.view(B,E,Q,1,1)/3.0) * I.view(1,1,1,3,3)                 # (B,E,Q,3,3)
        I3_D = D_dev.reshape(B,E,Q,9)                                               # (B,E,Q,9)
        J2 = 0.5 * (D_dev*D_dev).sum(dim=(-2,-1)).unsqueeze(-1)                     # (B,E,Q,1)
        I4_D = torch.sqrt(J2 + 1e-20)                                               # (B,E,Q,1)
        
        I1_W = torch.stack([Wc[...,2,1], Wc[...,0,2], Wc[...,1,0]], dim=-1)         # (B,E,Q,3)
        I2_W = (2 * I1_W).norm(dim=-1, keepdim=True)                                # (B,E,Q,1)

        rho = self.rho.reshape(1,E,1,1).expand(B,E,Q,1)                             # (B,E,Q,1)
        volume = self.element_sample_volume.unsqueeze(0)                            # (B,E,Q,1)
        m = rho * volume                                                            # (B,E,Q,1)

        q_rel, v_e, v_rel, v_rad, v_perp, sL_rel, q_cm, v_cm, Omega_cm = self.element_geom(q, v, m)  # (B,E,Q,3)

        q_rel_c = torch.einsum('beqij,beqj->beqi', Rt, q_rel)                       # (B,E,Q,3)
        v_e_c = torch.einsum('beqij,beqj->beqi', Rt, v_e)                       # (B,E,Q,3)
        v_rel_c = torch.einsum('beqij,beqj->beqi', Rt, v_rel)                       # (B,E,Q,3)
        v_rad_c = torch.einsum('beqij,beqj->beqi', Rt, v_rad)                       # (B,E,Q,3)
        v_perp_c = torch.einsum('beqij,beqj->beqi', Rt, v_perp)                       # (B,E,Q,3)
        sL_rel_c = torch.einsum('beqij,beqj->beqi', Rt, sL_rel)                       # (B,E,Q,3)
        v_cm_c = torch.einsum('beqij,beqj->beqi', Rt, v_cm)                         # (B,E,Q,3)
        Omega_cm_c = torch.einsum('beqij,beqj->beqi', Rt, Omega_cm)                         # (B,E,Q,3)

        q_rel_mag = q_rel.norm(dim=-1, keepdim=True)                                # (B,E,Q,1)
        v_e_mag = v_e.norm(dim=-1, keepdim=True)                                # (B,E,Q,1)
        v_rel_mag = v_rel.norm(dim=-1, keepdim=True)                                # (B,E,Q,1)
        v_cm_mag  = v_cm.norm(dim=-1, keepdim=True)                                 # (B,E,Q,1)
        Omega_cm_mag = Omega_cm.norm(dim=-1, keepdim=True)
        # q_0 = self.q0.reshape(1,-1,3)
        # q_centered = q_0 - q_0.mean(dim=1, keepdim=True)
        # q_centered = self.to_elements(q_centered)                             # (1,E,Ne,3)
        # q_centered = torch.einsum('beni,eqn->beqi', q_centered, self.N_q.to(q_centered.dtype))           # (1,E,Q,3)
        # q_centered = q_centered.expand(B,E,Q,3)

        # dist2face = self.dist2face.unsqueeze(0).expand(B,E,Q,1)
        # dir2face = self.dir2face.unsqueeze(0).expand(B,E,Q,3) 
        # normdist2face = (self.dist2face - self.dist2face.min()) / (self.dist2face.max() - self.dist2face.min())
        # normdir2face = (normdist2face * self.dir2face).unsqueeze(0).expand(B,E,Q,3)

        #norm_offset2face, dir2face, normdist2face = self.offset_to_faces(q)
        norm_offset2face = self.offset_to_faces(q)
        norm_offset2face_c = torch.einsum('beqij,beqj->beqi', Rt, norm_offset2face)         # (B,E,Q,3)
        #dir2face_c = torch.einsum('beqij,beqj->beqi', Rt, dir2face)         # (B,E,Q,3)
        # max_dist_to_face = (torch.sum(dir2face_c.reshape(B,E*Q,1,3) * offset2face_c.reshape(B,1,E*Q,3), dim=-1)).max(dim=2)[0].reshape(B,E,Q,1)
        # norm_offset2face_c = offset2face_c / max_dist_to_face
        # print(norm_offset2face_c.reshape(B,-1,3).max(dim=1)[0])
        # print(norm_offset2face_c.reshape(B,-1,3).min(dim=1)[0])
        # print(norm_offset2face_c)
        # exit()
        # max_offset2face_c = torch.abs(offset2face_c.reshape(B,E*Q,3)).max(dim=1)[0].reshape(B,1,1,3)                 # (B,1,3)
        # norm_offset2face_c = offset2face_c / max_offset2face_c                                   # (B,E*Q,3)

        # orig_dist_to_face = self.dist2face.expand(B,E,Q,1)                           # (B,E,Q,1)
        # max_dist_to_face = orig_dist_to_face.reshape(B,E*Q,1).max(dim=1)[0].reshape(B,1,1,1)
        # orig_offset2face = orig_dist_to_face * self.dir2face.expand(B,E,Q,3)      # (B,E*Q,3)

        # max_offset2face = torch.abs(orig_offset2face.reshape(B,E*Q,3)).max(dim=1)[0].reshape(B,1,1,3)                 # (B,1,3)
        # norm_offset2face_c = offset2face_c / max_offset2face                                   # (B,E,Q,3)
        # norm_orig_offset2face = orig_offset2face.reshape(B,E,Q,3) / max_dist_to_face #max_offset2face
        #norm_orig_dist2face = orig_dist_to_face / torch.linalg.norm(max_offset2face, dim=-1, keepdim=True)
        
        orig_offset2face = self.offset2face.reshape(1,E,Q,3).expand(B,E,Q,3)

        # 13-dim features (co-rotated frame):
        feat_deformation = torch.cat([
                                        I1_F,                                       # principle stretches (3)
                                        I2_F,                                       # flattened(C - I) (9)
                                        I3_F,                                       # det(F) - 1 (1)
                                    ], dim=-1)                                      # (B,E,Q,13)
        

        # 14-dim features (co-rotated frame):
        feat_strainRate = torch.cat([
                                    I1_D,                                           # diag(Dc) (3)
                                    I2_D,                                           # trace(Dc) (1)
                                    I3_D,                                           # flattened(dev(Dc)) (9)
                                    I4_D,                                           # sqrt(0.5 * ||dev(Dc)||_F^2) (1)
                                ], dim=-1)                                          # (B,E,Q,14)
        
        # 4-dim features (co-rotated frame): 
        feat_spin = torch.cat([
                                I1_W,                                               # axial(W) (3)
                                I2_W,                                               # vorticity magnitude (1)
                            ], dim=-1)                                              # (B,E,Q,4)
        
        # 11-dim features (co-rotated frame): 
        feat_rigidMotion = torch.cat([
                                    #v_rad_c,
                                    #v_perp_c,
                                    norm_offset2face_c,
                                    #orig_offset2face,
                                    #dir2face_c,
                                    #normdist2face,
                                    #v_e_c,
                                    #q_rel_c,
                                    #v_rel_c,
                                    #sL_rel_c,
                                    # v_rel_srf_c,
                                    v_cm_c,
                                    #Omega_cm_c,
                                    #v_e_mag,
                                    #v_rel_mag,
                                    v_cm_mag,
                                    #Omega_cm_mag
                                ], dim=-1)                                          # (B,E,Q,11)

        if self.actuated:
            f_e, t_e, f_total = self.force_geom(q, f, m, q_cm)             # (B,E,Q,3)
            
            f_e_c = torch.einsum('beqij,beqj->beqi', Rt, f_e)                       # (B,E,Q,3)
            
            # 3-dim features (co-rotated frame): 
            feat_force = torch.cat([
                                f_e_c,                                              # Effective external force per element (3)
                                ], dim=-1)                                          # (B,E,Q,3)
        else:
            feat_force = None

        # # Acceleration density scaled by element sample mass
        f_e_c = m * self.unmodelled_nn(feat_deformation,
                                        feat_strainRate,
                                        feat_spin,
                                        feat_rigidMotion,
                                        feat_force)                                 # (B,E,Q,3)

        # Rotate back to world frame
        f_e = torch.einsum('beqij,beqj->beqi', R, f_e_c)                            # (B,E,Q,3)

        # Distribute to nodes
        fn = self.elementForce_to_nodeForce(f_e)                                    # (B,E,Ne,3)

        # P_e_c = self.unmodelled_nn(feat_deformation,
        #                                 feat_strainRate,
        #                                 feat_spin,
        #                                 feat_rigidMotion,
        #                                 feat_force)                                 # (B,E,Q,3,3)

        # rho = self.rho.reshape(1,E,1,1)
        # P_e =  torch.einsum('beqij,beqjk->beqik', R, P_e_c)


        # fn = rho * self.elementStress_to_nodeForce(P_e)

        output = self.elementNode_to_dof(fn)                                          # (B,3V) 


        return output
           

    def forward(self, X):
        if X.dim() == 1:
            X = X.unsqueeze(0)
        B = X.shape[0]

        if X.shape[1] == 2 * self.dof:
            q_flat, v_flat = torch.split(X, [self.dof, self.dof], dim=-1)
            f_ext_flat = torch.zeros_like(q_flat)
        else:
            q_flat, v_flat, f_ext_flat = torch.split(X, [self.dof, self.dof, self.dof], dim=-1)

        q = q_flat.view(B, -1, 3)                       # Position
        v = v_flat.view(B, -1, 3)                       # Velocity
        f_ext = f_ext_flat.view(B,-1, 3)                # Actuation Forces


        f_umdld = self.force_unmodelled(q, v, f_ext)

        return f_umdld 

    def count_parameters(self):
        
        n = sum(p.numel() for p in self.unmodelled_nn.parameters() if p.requires_grad)
        return n