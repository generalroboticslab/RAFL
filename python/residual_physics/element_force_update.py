from typing import Optional
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch import Tensor
from einops.layers.torch import Rearrange
from einops import rearrange
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

def _make_mlp(in_dim, hidden_dims, out_dim, nonlinearity='gelu', no_bias=False, norm=None, dropout=None, last_nonlinearity=False):
    act = get_nonlinearity(nonlinearity)

    layers = []
    d = in_dim
    for h in hidden_dims:
        layers.append(MLPBlock(d, h, no_bias, norm, nonlinearity))
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(MLPBlock(d, out_dim, no_bias, None, None if not last_nonlinearity else nonlinearity))
    if dropout is not None:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class Unmodelled_Acceleration(nn.Module):
    def __init__(self, actuated, hidden_dims, nonlinearity, no_bias=True, normalize_inputs=True, conditioned=True, stress=False):
        super().__init__()
        self.actuated = actuated
        self.input_dim = 37 if self.actuated else 34

        self.conditioned = conditioned

        if not self.conditioned:
            self.input_dim -= 3

        print(self.conditioned)
        print(stress)

        self.stress = stress
        self.normalize_inputs = normalize_inputs
        if self.normalize_inputs:
            self.normalize = torch.nn.BatchNorm1d(self.input_dim, dtype=torch.float64)

        self.net = _make_mlp(in_dim=self.input_dim, hidden_dims=hidden_dims, out_dim=3 if not stress else 6,
                             nonlinearity=nonlinearity, no_bias=no_bias)
    
    @staticmethod
    def _sym_from_6(x6: Tensor) -> Tensor:
        """Map (...,6)->(...,3,3) with order [xx, yy, zz, xy, yz, zx]."""
        xx, yy, zz, xy, yz, zx = torch.unbind(x6, dim=-1)
        X = torch.stack([
            torch.stack([xx, xy, zx], dim=-1),
            torch.stack([xy, yy, yz], dim=-1),
            torch.stack([zx, yz, zz], dim=-1)
        ], dim=-2)
        return X

    def input_features(self, feat_deformation, feat_strainRate, feat_spin, feat_rigidMotion, feat_force):

        if self.conditioned:
            feat = torch.cat([feat_deformation,
                                feat_strainRate,
                                feat_spin,
                                feat_rigidMotion[:,:,:,:3]
                                ], dim=-1)
        else:
            feat = torch.cat([feat_deformation,
                                feat_strainRate,
                                feat_spin
                                ], dim=-1)

        if self.actuated:
            feat = torch.cat([feat,
                                feat_force[:,:,:,3:6]
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

        if self.normalize_inputs:
            norm_feat = self.normalize(feat)
            output = self.net(norm_feat)
        else:
            output = self.net(feat)
        
        if self.stress:
            a = self._sym_from_6(output).view(*orig_shape,3,3)
        else:
            a = output.view(*orig_shape,3)

        return a


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "Expected (B,C)"
        dims = (0,)
        # print("training:", self.training)
        # print("track:", self.track_running_stats)
        if self.training or not self.track_running_stats:
            var = x.var(dims, keepdim=False, unbiased=False)  # (C,)
            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    m = self.momentum
                    self.running_var.lerp_(var, m)
        else:
            var = self.running_var
            # print("eval")

        inv_std = torch.rsqrt(var + self.eps)                 # (C,)
        shape = (1, -1) 
        y = x * inv_std.view(*shape)                          # no mean subtraction, no bias
        if self.affine:
            y = y * self.weight.view(*shape)
        return y


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, nonlinearity, no_bias=True, normalize_inputs=True, dropout = 0.1, last_nonlinearity=False):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_inputs = normalize_inputs

        if self.normalize_inputs:
            if no_bias:
                self.normalize = VarNorm1d(self.input_dim)
            else:
                self.normalize = torch.nn.BatchNorm1d(self.input_dim, affine=False, dtype=torch.float64)

        self.net = _make_mlp(in_dim=self.input_dim, hidden_dims=hidden_dims, out_dim=self.output_dim,
                             nonlinearity=nonlinearity, no_bias=no_bias, dropout=dropout, last_nonlinearity=last_nonlinearity)

    def forward(self, x):

        if self.normalize_inputs:
            x = self.normalize(x)

        return self.net(x)

class Unmodelled_Acceleration_Separated(nn.Module):
    def __init__(self, actuated, multi_shape, dim_head, heads, hidden_dims, nonlinearity, normalize_inputs=True, dropout = None):
    # def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()

        self.actuated = actuated
        self.multi_shape = multi_shape
        self.dim_head = dim_head
        self.heads = heads #2 * heads if self.multi_shape else heads
        self.normalize_inputs = normalize_inputs

        self.condition_dim = 3 #8 if self.multi_shape else 7 
        self.feat_dim = 34 if self.actuated else 31

        self.nonlinearity = get_nonlinearity(nonlinearity)
        head_hidden_dims = [8] #[16] if self.multi_shape else [8]
        
        self.to_q = FeedForward(self.condition_dim, head_hidden_dims, self.dim_head *  self.heads, nonlinearity, no_bias=False, normalize_inputs=normalize_inputs, dropout = dropout) 
        self.to_k = FeedForward(self.feat_dim, [32], self.dim_head *  self.heads, nonlinearity, no_bias=True, normalize_inputs=normalize_inputs, dropout = dropout) 
        
        self.to_out = FeedForward(self.heads *  self.heads, hidden_dims, 3, nonlinearity, no_bias=True, normalize_inputs=False, dropout = dropout)


    def forward(self, feat_deformation, feat_strainRate, feat_spin, feat_rigidMotion, feat_force):

        x = feat_rigidMotion[:,:,:,:3] #7]

        z = torch.cat([feat_deformation,
                            feat_strainRate,
                            feat_spin,
                            ], dim=-1)
        
        if self.actuated:
            z = torch.cat([z,
                                feat_force[:,:,:,3:6]
                                ], dim=-1)
        
        if self.multi_shape:
            x = torch.cat([x, feat_rigidMotion[:,:,:,-1:]], dim=-1)
        
        B, E, Q = x.shape[:-1] 

        x = x.reshape(B*E*Q, self.condition_dim)                  # (B, E*Q, condition_dim)
        z = z.reshape(B*E*Q, self.feat_dim)                       # (B, E*Q, feat_dim)

        q = self.to_q(x)
        q = q.reshape(B,E*Q,self.heads,self.dim_head) # (B,E*Q,heads,dim_head)

        k = self.to_k(z)
        k = k.reshape(B,E*Q,self.heads,self.dim_head) # (B,E*Q,heads,dim_head)

        dots = torch.matmul(q, k.transpose(-1, -2)) # (B,E*Q,heads,heads)

        dots = self.nonlinearity(dots)  # (B,E*Q,heads,heads)

        out_feat = dots.reshape(B, E*Q, self.heads * self.heads) # (B,E*Q,heads*heads)

        out = self.to_out(out_feat) # (B,E*Q,3)

        return out.view(B,E,Q,3)





class ElementResidual(nn.Module):
    def __init__(self, dof, 
                        elements, 
                        faces, 
                        mu, 
                        lam, 
                        rho, 
                        X_e, 
                        dx, 
                        hidden_size=128, 
                        num_hidden_layer=4, 
                        nonlinearity='elu', 
                        no_bias=True, 
                        actuated=False, 
                        force_nodes=None, 
                        gravity=True, 
                        scale=1, 
                        changing_boundary_indices=None, 
                        normalize_inputs=True, 
                        multi_shape=False,
                        separated=True,
                        conditioned=True,
                        stress=False):
        super().__init__()
        
        self.dof = dof                                      # num_vertices * 3
        self.register_buffer('elements', elements.long())   # element vertex index mapping  
        self.register_buffer('faces', faces.long())         # element vertex index mapping  
        self.register_buffer('mu', mu)                      # material Lame parameter mu
        self.register_buffer('lam', lam)                    # material Lame parameter lambda
        self.register_buffer('rho', rho)                    # material density
        self._precompute_quadrature(X_e, dx)                # shape functions for FEM    
        self._precompute_surface_offsets()        

        self.actuated = actuated
        self.multi_shape = multi_shape
        if self.actuated:
            if gravity:
                self.register_buffer('g', scale * torch.tensor([0, 0, -9.80709], dtype=torch.float64))
            else:
                self.register_buffer('g', torch.tensor([0, 0, 0], dtype=torch.float64))
        
        if force_nodes is not None:
            self.register_buffer('force_nodes', force_nodes.long())
        else:
            self.force_nodes = None

        self.stress = stress
        print(separated)
        print(normalize_inputs)
        # features -> acceleration density
        if separated:
            self.unmodelled_nn = Unmodelled_Acceleration_Separated(actuated=self.actuated, 
                                                            multi_shape=self.multi_shape,
                                                            dim_head = 8 if hidden_size >= 32 else 4,
                                                            heads = 8 if hidden_size >= 32 else 4,
                                                            hidden_dims=num_hidden_layer*[hidden_size],
                                                            nonlinearity=nonlinearity,
                                                            normalize_inputs=normalize_inputs,
                                                            dropout = None)   
        else:
            self.unmodelled_nn = Unmodelled_Acceleration(actuated=self.actuated, 
                                                        hidden_dims=num_hidden_layer*[hidden_size],
                                                        nonlinearity=nonlinearity,
                                                        normalize_inputs=normalize_inputs,
                                                        no_bias=no_bias,
                                                        conditioned=conditioned,
                                                        stress=stress)   

        self.changing_boundary_indices = changing_boundary_indices

    def reset_mesh(self, X_e, elements, faces, dx):

        self.elements = elements.long().to(self.rho.device)
        self.faces = faces.long().to(self.rho.device)
        self._precompute_quadrature(X_e, dx, False)
        self._precompute_surface_offsets(initial=False)


    def _precompute_quadrature(self, X_e, dx, initial=True):
        E, Ne = self.elements.shape

        # Store reference positions q0 (flattened 3*V)
        if initial:
            self.register_buffer('q0', torch.tensor(X_e, dtype=torch.float64))
        else:
            self.q0 = torch.tensor(X_e, dtype=torch.float64).to(self.rho.device)

        if Ne == 8:

            if initial:
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

            # print(samples)
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

            element_sample_reference_J_e = torch.stack(8*[torch.diag(dx_t)])

            # Replicate for all elements
            N_q = [N_q_e for _ in range(E)]
            gradN_q = [gradN_q_e for _ in range(E)]
            element_sample_volume = [element_sample_volume_e for _ in range(E)]
            element_sample_reference_J = [element_sample_reference_J_e for _ in range(E)]

            if initial:
                self.register_buffer('N_q', torch.stack(N_q, dim=0))                     # (E, Q=8, Ne=8)
                self.register_buffer('gradN_q', torch.stack(gradN_q, dim=0))             # (E, Q=8, Ne=8, 3)
                self.register_buffer('element_sample_volume',
                                    torch.stack(element_sample_volume, dim=0))          # (E, Q=8, 1)
                self.register_buffer('element_sample_reference_J', torch.stack(element_sample_reference_J, dim=0)) # (E, Q=8, 3, 3)
            else:
                self.N_q = torch.stack(N_q, dim=0).to(self.rho.device)
                self.gradN_q = torch.stack(gradN_q, dim=0).to(self.rho.device)
                self.element_sample_volume = torch.stack(element_sample_volume, dim=0).to(self.rho.device)
                self.element_sample_reference_J = torch.stack(element_sample_reference_J, dim=0).to(self.rho.device)

        elif Ne == 4:

            if initial:
                print("Tet")

            self._mesh_type = 'tet'

            N_q = []                    # Shape function N
            gradN_q = []                # Shape function gradient ∇N
            element_sample_volume = []  # Element sample volume
            element_sample_reference_J = []   # Element xyz scale
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

                element_sample_reference_J.append(J)

            if initial:
                self.register_buffer('N_q', torch.stack(N_q, dim=0).unsqueeze(1))                 # (E, Q=1, Ne)
                self.register_buffer('gradN_q', torch.stack(gradN_q, dim=0).unsqueeze(1))         # (E, Q=1, Ne,3)
                self.register_buffer('element_sample_volume',
                                    torch.stack(element_sample_volume, dim=0).unsqueeze(1))      # (E, Q=1, 1)
                self.register_buffer('element_sample_reference_J', torch.stack(element_sample_reference_J, dim=0).unsqueeze(1)) # (E, Q=1, 3, 3)
            else:
                self.N_q = torch.stack(N_q, dim=0).unsqueeze(1).to(self.rho.device)
                self.gradN_q = torch.stack(gradN_q, dim=0).unsqueeze(1).to(self.rho.device)
                self.element_sample_volume = torch.stack(element_sample_volume, dim=0).unsqueeze(1).to(self.rho.device)
                self.element_sample_reference_J = torch.stack(element_sample_reference_J, dim=0).unsqueeze(1).to(self.rho.device)

        else:
            # Current implementation only supports hex and tet mesh
            exit()

        self._precompute_COM(initial)

    def _precompute_COM(self,initial):

        E, Ne = self.elements.shape
        q0 = self.q0.reshape(-1,3)
        q0_nodes = q0[self.elements, :]                                         # (E,Ne,3)
        q0_e = torch.einsum('eni,eqn->eqi', q0_nodes, self.N_q)                 # (E,Q,3)

        rho = self.rho.reshape(E,1,1)                                           # (E,1,1)
        volume = self.element_sample_volume                                     # (E,Q,1)
        m = rho * volume                                                        # (E,Q,1)
        X_cm = (m * q0_e).sum(dim=[0,1]) / m.sum(dim=[0,1])                     # (3)

        offset2COM = q0_e.reshape(-1,3) - X_cm.reshape(1,3) #(E*Q,3)
        
        max_offset2COM = torch.abs(offset2COM).max(dim=0, keepdim=True)[0] # (1,3)
        norm_offset2COM = offset2COM / max_offset2COM

        if initial:
            self.register_buffer('norm_offset2COM', norm_offset2COM)
        else:
            self.norm_offset2COM = norm_offset2COM

        element2COM = torch.linalg.norm(X_cm.reshape(1,1,3) - q0_e, dim=-1).mean(dim=-1)       # (E,)

        e_cm = torch.argmin(element2COM)

        X_e = q0[self.elements[e_cm]]                                     # (Ne, 3)

        if self._mesh_type == 'tet':

            A = torch.stack([X_e[1] - X_e[0], X_e[2] - X_e[0], X_e[3] - X_e[0]], dim=1)     # (3,3)
            u, v, w = torch.linalg.solve(A, (X_cm - X_e[0]).reshape(3, 1)).reshape(-1)      # (3,)
            N_e = torch.stack([1.0 - (u + v + w), u, v, w], dim=0)                          # (Ne,)

        elif self._mesh_type == 'hex':
            
            A = torch.stack([X_e[4]-X_e[0], X_e[2]-X_e[0], X_e[1]-X_e[0]], dim=1)           # (3,3)
            nx, ny, nz = torch.linalg.solve(A, (X_cm - X_e[0]).reshape(3,1)).reshape(-1).clamp(0.,1.)    # (3,)
            cnx, cny, cnz = 1.0 - nx, 1.0 - ny, 1.0 - nz

            N_e = torch.stack([
                        cnx * cny * cnz,   # N000
                        cnx * cny * nz,    # N001
                        cnx * ny * cnz,    # N010
                        cnx * ny * nz,     # N011
                        nx * cny * cnz,    # N100
                        nx * cny * nz,     # N101
                        nx * ny * cnz,     # N110
                        nx * ny * nz,      # N111
                    ], dim=0)                                                                  # (Ne,)
            

        
        N_cm = torch.zeros((E,Ne), dtype=N_e.dtype, device=N_e.device)                         # (E, Ne)                   
        N_cm[e_cm] = N_e

        if initial:
            self.register_buffer('N_cm', N_cm)
        else:
            self.N_cm = N_cm.to(self.rho.device)

        




    def _precompute_surface_offsets(self, k=3, rel_tol=1e-6, initial=True):

        E, Ne = self.elements.shape
        F = self.faces.shape[0]
        
        q0 = self.q0.reshape(-1,3)
        Qf = q0[self.faces, :]
        face_positions = q0[self.faces, :].mean(1)
        element_positions = torch.einsum('eni,eqn->eqi', q0[self.elements, :], self.N_q) 

        if initial:
            self.register_buffer('element_q0', element_positions)
        else:
            self.element_q0 = element_positions.to(self.rho.device)

        if self._mesh_type == 'hex':

            Q = 8

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

            face2face_offset = face_positions.reshape(1,F,3) - face_positions.reshape(F,1,3) #(F,F,3)
            face2face_dist = torch.sum(-n_f.reshape(F,1,3) * face2face_offset, dim=-1)  #(F,F)
            face2face_offsetPerp = torch.linalg.norm(face2face_offset + face2face_dist.reshape(F,F,1) * n_f.reshape(F,1,3), dim=-1)  #(F,F)
            
            forward = face2face_dist >= 0 # (F,F)
            facing = (-n_f.reshape(F,1,3) * n_f.reshape(1,F,3)).sum(dim=-1) > 0 #.8 #(F,F)
            not_self = ~(torch.eye(F, dtype=torch.bool, device=q0.device)) #(F,F)
            lateral_ok = face2face_offsetPerp <= torch.sqrt(A_f) / 2 # (F,F)

            mask =  forward & lateral_ok & facing & not_self #(F,F)

            # assert(torch.all(mask.sum(dim=-1) == 1))

            valid_face2face_dist = torch.where(mask, face2face_dist, torch.full_like(face2face_dist, float("inf"))) #(F,F)

            max_depth_per_face = valid_face2face_dist.min(dim=-1)[0] #(F)
            if initial:
                self.register_buffer('max_depth', max_depth_per_face)
            else:
                self.max_depth = max_depth_per_face
            # assert(torch.all(max_depth_per_face.isfinite()))

            element_to_face_diff = face_positions.reshape(1,F,3) - element_positions.reshape(E*Q,1,3)   # (E*Q,F,3)
            alignment = torch.sum(n_f.reshape(1,F,3) * element_to_face_diff, dim=-1) # (E*Q,F)
            element_to_face_offsetPerp = torch.linalg.norm(element_to_face_diff - alignment.unsqueeze(-1) * n_f.reshape(1,F,3), dim=-1) # (E*Q,F)
            element_to_face_dists = torch.cdist(element_positions.reshape(-1,3), face_positions, p=2)                 # (E*Q,F)
            element_depth_per_face = alignment / max_depth_per_face.reshape(1,F) # (E*Q,F)

            alignment_ok = alignment > 0

            element_to_face_lateral_ok = element_to_face_offsetPerp <= torch.sqrt(A_f).reshape(1,F) / 2 

            element_mask = alignment_ok & element_to_face_lateral_ok

            # assert(torch.all(element_mask.sum(dim=-1) == 6))

            # exit()
            dists = torch.where(element_mask, element_depth_per_face, float('inf')) # (E*Q,F)
            _, min_dist_face_idx = torch.topk(dists, largest=False, k =k, dim=1)        # (E*Q,k)                        
            idx = min_dist_face_idx.reshape(E*Q*k,1)                     # (E*Q*k,3)

            if initial:
                self.register_buffer('target_faces_idx', idx)
            else:
                self.target_faces_idx = idx
            target_face_positions = torch.gather(face_positions, dim=0, index=idx.expand(-1, 3)).reshape(E*Q,k,3)  # (E*Q*k,3)
            target_face_normals = torch.gather(n_f, dim=0, index=idx.expand(-1, 3)).reshape(E*Q,k,3)               # (E*Q*k,3)
            target_max_depth = torch.gather(max_depth_per_face, dim=0, index=idx.squeeze(-1)).reshape(E*Q,k)       # (E*Q*k,)

            diff_to_faces = target_face_positions - element_positions.reshape(E*Q,1,3)                # (E*Q,k,3)
            dist_to_faces = torch.sum(diff_to_faces * target_face_normals, dim =-1)     # (E*Q,k)
            depth_to_faces = dist_to_faces / target_max_depth  # (E*Q,k)


        elif self._mesh_type == 'tet':

            Q = 1

            # Single Triangle
            p0, p1, p2 = Qf[:, 0, :], Qf[:, 1, :], Qf[:, 2, :]                 # (F,3)
            cp   = torch.cross(p1 - p0, p2 - p1, dim=-1)                       # (F,3)
            area_par = cp.norm(dim=-1, keepdim=True)                           # (F,1)
            
            # area and norm of triangles
            A_f  = 0.5 * area_par                                              # (F,1)
            n_f = cp / (area_par + 1e-12)                                      # (F,3)

            # edges for barycentric coordinates
            e1 = p1 - p0 # (F,3)
            e2 = p2 - p0 # (F,3)

            # Face normals projected to xyz directions (inward)
            # Filter directions face is nearly parallel to
            xyz = torch.eye(3, dtype = q0.dtype, device= q0.device) # (3,3)
            dots = torch.einsum('fi,ai->fa', -n_f, xyz)              # (F,3)
            dots_signs = torch.where(torch.abs(dots) > 1e-6, torch.sign(dots), 0)
            dirs = dots_signs.reshape(F,3,1) * xyz.reshape(1,3,3)  # (F,3,3)

            # Origin and bases for barycentric coordinates
            V0 = p0.reshape(1,1,F,3)       # (1,1,F,3)
            E1 = e1.reshape(1,1,F,3)       # (1,1,F,3)
            E2 = e2.reshape(1,1,F,3)       # (1,1,F,3)

            # Factors for solving barycentric coordinates
            # O + tD = V0 + uE1 + vE2
            O = face_positions.reshape(F,1,1,3)  # (F,1,1,3)
            # eps_face = 1e-6  # scale to your mesh edge length
            # O = (face_positions - eps_face * n_f).reshape(F,1,1,3)  # n_f outward -> subtract goes inward
            D = dirs.reshape(F,3,1,3)     # (F,3,1,3)
            V0 = p0.reshape(1,1,F,3)       # (1,1,F,3)
            E1 = e1.reshape(1,1,F,3)       # (1,1,F,3)
            E2 = e2.reshape(1,1,F,3)       # (1,1,F,3)
            
            # Rearrange system to solve using Cramer's Rule
            # T = O - V0 = uE1 + vE2 - tD
            # u = det[T,D,E2] / det[E1,D,E2]
            # v = det[D,T,E1] / det[E1,D,E2]
            # t = det[E2,T,E1] / det[E1,D,E2]
            T = O - V0                              # (F,1,F,3)
            P = torch.cross(D, E2, dim=-1)          # (F,3,F,3)
            S = torch.cross(T, E1, dim=-1)          # (F,1,F,3)
            det = torch.sum(E1 * P, dim=-1)         # (F,3,F)
            inv_det = torch.where(det.abs() > 1e-10, 1.0 / det, torch.zeros_like(det)) # (F,3,F)

            # barycentric weights along E1, E2
            u = torch.sum(T * P, dim=-1) * inv_det     # (F,3,F)
            v = torch.sum(D * S, dim=-1) * inv_det     # (F,3,F)
            t = torch.sum(E2 * S, dim=-1) * inv_det    # (F,3,F)

            # Mask target faces
            # target face "infront" respect to projection
            forward = (t > 1e-10)    #(F,3,F)
            # target face is NOT nearly coplanar with projection

            opposing = (torch.sum(dirs.reshape(F,3,1,3) * dirs.transpose(0,1).reshape(1,3,F,3), dim=-1) < 0) # (F,3,F)
            facing = det.abs() > 1e-10 # (F,3,F)
            # intersection is inside the target triangle
            lateral_ok = (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12) #(F,3,F)
            # target face is not itself
            not_self = ~(torch.eye(F, dtype=torch.bool, device=q0.device)).unsqueeze(1).expand(F,3,F) #(F,3,F)
            # combined mask
            mask =   forward & opposing & facing & lateral_ok & not_self

            # max depth of face (xyz directions) is minimum among filtered distances
            valid_face2face_dist = torch.where(mask, t, torch.full_like(t, float("inf")))  # (F,3,F)
            max_depth_per_face = valid_face2face_dist.min(dim=-1)[0] #(F,3)

            if initial:
                self.register_buffer('max_depth', max_depth_per_face)
            else:
                self.max_depth = max_depth_per_face

            # Element's depth from surfaces in +-xyz directions
            origins = element_positions.reshape(E*Q,1,3).expand(E*Q, 6, 3) # (E*Q,6,3)
            dirs6 = torch.tensor([
                [ 1, 0, 0],
                [-1, 0, 0],
                [ 0, 1, 0],
                [ 0,-1, 0],
                [ 0, 0, 1],
                [ 0, 0,-1],
            ], device=q0.device, dtype=q0.dtype)  # (6,3)
            
            # Factor for solving barycentric coordinates
            # O + tD = V0 + uE1 + vE2
            O  = origins.reshape(E*Q,6,1,3)        # (E*Q,6,1,3)
            D  = dirs6.reshape(1,6,1,3)             # (1,6,1,3)
            
            # Rearrange system to solve  using Cramer's Rule
            # T = O - V0 = uE1 + vE2 - tD
            # u = det[T,D,E2] / det[E1,D,E2]
            # v = det[E1,T,D] / det[E1,D,E2]
            # t = det[E1,E2,T] / det[E1,D,E2]
            # (E*Q,6,1,3) - (1,1,F,3)
            T = O - V0                                    # (E*Q,6,F,3)
            # (1,6,1,3) x (1,1,F,3) 
            P = torch.cross(D, E2, dim=-1)                # (1,6,F,3)
            # (E*Q,6,F,3) x  (1,1,F,3)
            S = torch.cross(T, E1, dim=-1)                # (E*Q,6,F,3)
            # (1,1,F,3) . (1,6,F,3)
            det = torch.sum(E1 * P, dim=-1)               # (1,6,F)
            inv_det = torch.where(det.abs() > 1e-10, 1.0 / det, torch.zeros_like(det)) # (1,6,F)

            # (E*Q,6,1,3) . (1,6,F,3)
            u = torch.sum(T * P, dim=-1) * inv_det        # (E*Q,6,F)
            # (1,6,1,3) . (E*Q,6,F,3)
            v = torch.sum(D * S, dim=-1) * inv_det        # (E*Q,6,F)
            # (1,1,F,3) . (E*Q,6,F,3)
            t = torch.sum(E2 * S, dim=-1) * inv_det       # (E*Q,6,F)

            # Mask target faces
            # target face "infront" respect to direction
            forward = (t > 1e-10)    # (E*Q,6,F)
            # target face is NOT nearly coplanar with direction
            opposing = (torch.sum(dirs6.reshape(6,1,3) * dirs.transpose(0,1).reshape(3,1,F,3).expand(3,2,F,3).reshape(6,F,3), dim=-1) < 0).unsqueeze(0) # (1,6,F)

            assert(torch.all(opposing.sum(dim=[0,1])) <= 3)
            # for i in range(opposing.shape[2]):

            #     if opposing[:,:,i].sum() < 3:
            #         print(opposing[:,:,i])
            #         print(n_f[i])
            facing = det.abs() > 1e-10 # (E*Q,6,F)
            # intersection is inside the target triangle
            lateral_ok = (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12) # (E*Q,6,F)
            # combined mask
            mask = forward & opposing & facing & lateral_ok # (E*Q,6,F)

            # element distance to surface in each direction is minimum among filtered distances to faces
            valid_element2face_dist = torch.where(mask, t, torch.full_like(t, float("inf"))) # (E*Q,6,F)
            dist6, idx = torch.min(valid_element2face_dist, dim=2) # (E*Q,6)

            # map +-xyz direction indices to xyz indices in {0,1,2}
            axis6 = torch.arange(6, device=q0.device).view(1, 6).expand(E*Q, 6) // 2 # (E*Q,6) 

            # get max depths of target surfaces
            # ignore non-existant directions
            idx = torch.where(torch.isfinite(dist6), idx, torch.full_like(idx, -1)) # (E*Q,6)
            hit6 = idx.clamp_min(0)
            valid_hit = (idx >= 0)  # (E*Q,6)

            rows = max_depth_per_face[hit6]  # (E*Q,6,3)
            maxd_face_axis = torch.gather(rows, dim=2, index=axis6.unsqueeze(-1)).squeeze(-1)

            maxd_face_axis = torch.where(valid_hit, maxd_face_axis, torch.full_like(maxd_face_axis, float("inf")))

            # invalid if: no hit, dist inf, maxd inf
            invalid = ~valid_hit | (~torch.isfinite(maxd_face_axis))

            # normalized depth = dist / max_depth(face,axis)
            depth6 = dist6 / maxd_face_axis # (E*Q,6)
            depth6 = torch.where(invalid, torch.full_like(depth6, float("inf")), depth6)  # (E*Q,6)

            depth_to_faces, top_dir = torch.topk(depth6, largest=False, k =k, dim=1)        # (E*Q,k)    

            dist_to_faces = torch.gather(dist6, dim=1, index=top_dir) # (E*Q,k)
            
            top_hit  = torch.gather(hit6,  dim=1, index=top_dir) # (E*Q,k)
            
            target_face_normals = dirs6[top_dir]  # (E*Q,k,3)
            # print("target face normals: ", target_face_normals.shape)

            # for i in range(target_face_normals.shape[0]):
            #     idx_max = torch.max(target_face_normals[i], dim=0)[0]
            #     idx_min = torch.min(target_face_normals[i], dim=0)[0]

            #     if torch.any(torch.abs(idx_max) + torch.abs(idx_min) > 1):
            #         print(idx[i])
            #         print(dist6[i])
            #         print(depth6[i])
            #         print(maxd_face_axis[i])
            #         print(dist_to_faces[i])
            #         print(depth_to_faces[i])
            #         print(target_face_normals[i])


        else:
            exit()
        
        # directed_depth_to_faces = depth_to_faces.unsqueeze(-1) * target_face_normals.reshape(E*Q,k,3) # (E*Q,k,3)

        # scale = torch.diagonal(self.element_sample_reference_J, dim1=2,dim2=3).reshape(E*Q,1,3) / torch.diagonal(self.element_sample_reference_J, dim1=2,dim2=3).reshape(E*Q,1,3).mean(dim=-1, keepdim=True)
        # print(scale.mean(dim=0))

        # scaled_directed_depth_to_faces = torch.linalg.norm(directed_depth_to_faces * scale, dim=-1) #(E*Q,k)

        # w = torch.softmax(-scaled_directed_depth_to_faces/ 0.5, dim=1)


        scale = torch.abs((target_face_normals.reshape(E*Q,k,3) * torch.diagonal(self.element_sample_reference_J, dim1=2,dim2=3).reshape(E*Q,1,3)).sum(dim=-1)) / .01 #/ torch.diagonal(self.element_sample_reference_J, dim1=2,dim2=3).reshape(E*Q,3).mean(dim=-1, keepdim=True)

        # print(scale)
        w = torch.softmax( scale * depth_to_faces / 0.5, dim=1)

        dir_to_face = (w.unsqueeze(-1) * target_face_normals.reshape(E*Q,k,3)).sum(dim=1)        # (E*Q,3)

        # print(dir_to_face.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        dir_to_face = dir_to_face / dir_to_face.norm(dim=-1, keepdim=True).clamp_min(1e-12)      # (E*Q,3)

        depth_to_face = (w.unsqueeze(-1) * depth_to_faces.unsqueeze(-1) * scale.unsqueeze(-1)).sum(dim=1)
        dist_to_face = (w.unsqueeze(-1) * dist_to_faces.unsqueeze(-1)).sum(dim=1)        # (E*Q,3,1)
        # offset_to_face = (w.unsqueeze(-1) * ((0.5 - depth_to_faces).unsqueeze(-1) / depth_to_faces.unsqueeze(-1)) * dist_to_faces.unsqueeze(-1) * target_face_normals.reshape(E*Q,k,3)).sum(dim=1)
        # offset_to_face = (w.unsqueeze(-1) * dist_to_faces.unsqueeze(-1) * target_face_normals.reshape(E*Q,k,3)).sum(dim=1)
        # offset_to_face = (w.unsqueeze(-1) * (0.5 - depth_to_faces).unsqueeze(-1) * target_face_normals.reshape(E*Q,k,3)).sum(dim=1)
        

        offset_to_face = dist_to_face * dir_to_face # (E*Q,3)

        # offset_to_face = ((0.5 - depth_to_faces.unsqueeze(-1)) / depth_to_faces.unsqueeze(-1) * dist_to_faces.unsqueeze(-1) * target_face_normals.reshape(E*Q,k,3)).sum(dim=1) # (E*Q,3)
        offset_to_face = ((1 - 2 * depth_to_faces.unsqueeze(-1)) * target_face_normals.reshape(E*Q,k,3)).sum(dim=1) # (E*Q,3)
        # offset_to_face = 1e3 * offset_to_face
        # offset_to_face = (dist_to_faces.unsqueeze(-1) * target_face_normals.reshape(E*Q,k,3)).sum(dim=1) # (E*Q,3)
        # directed_depth_to_faces = (depth_to_faces.unsqueeze(-1) * torch.abs(target_face_normals).reshape(E*Q,k,3)).sum(dim=1) # (E*Q,3)

        # w = torch.softmax(-directed_depth_to_faces[:,1:] / 0.5, dim=1) #(E*Q,2)
        # weights = torch.cat([torch.ones((E*Q,1)), w], dim=1) #(E*Q,3)

        # offset_to_face = directed_offset_to_faces * dir_to_face
        # offset_to_face = (dist_to_face * target_face_normals.reshape(E*Q,k,3)).sum(dim=1)
        # if self._mesh_type == 'hex':
        #     max_offset_to_face = torch.abs(offset_to_face).max(dim=0, keepdim=True)[0] # (1,3)
        # elif self._mesh_type == 'tet':
        # max_offset_to_face = torch.max(torch.linalg.norm(offset_to_face, dim=-1)) # (1,)
        
        max_offset_to_face = torch.abs(offset_to_face).max(dim=0, keepdim=True)[0] # (1,3)
        min_offset_to_face = torch.abs(offset_to_face).min(dim=0, keepdim=True)[0] # (1,3)
        # max_offset_to_face[0,0] = 1.0
        norm_offset_to_face = offset_to_face #torch.sign(offset_to_face) * (torch.abs(offset_to_face) -  min_offset_to_face)/ (max_offset_to_face - min_offset_to_face)
        
        if initial:
            self.register_buffer('offset2face', offset_to_face.reshape(E,Q,3))
            self.register_buffer('norm_offset2face', norm_offset_to_face.reshape(E,Q,3))
        else:
            self.offset2face = offset_to_face.reshape(E,Q,3)
            self.norm_offset2face = norm_offset_to_face.reshape(E,Q,3)
        

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
        

        # U, sigma, Vh = torch.linalg.svd(F)                             # (N,3,3),(N,3),(N,3,3)
        # R = U @ Vh 

        # return sigma, R

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

        
        q_n = self.to_elements(q)                                                           # (B,E,Ne,3)
        v_n = self.to_elements(v)                                                           # (B,E,Ne,3)
        B,E,Ne,D = q_n.shape
        Q = self.N_q.shape[1]

        if self.changing_boundary_indices is not None:
            v_boundary = v.reshape(B,-1)[:,self.changing_boundary_indices].reshape(B,-1,3)              # (B,V,3)
            v_base = v_boundary.mean(dim=1, keepdim=True).unsqueeze(1)                                  # (B,1,1,3)

            q_boundary = q.reshape(B,-1)[:,self.changing_boundary_indices].reshape(B,-1,3)              # (B,V,3)
            q_base = self.q0[self.changing_boundary_indices].reshape(1,-1,3)                            # (1,V,3)
            q_offset = (q_boundary - q_base).mean(dim=1, keepdim=True).unsqueeze(1)                     # (B,1,1,3)
        else:
            v_base = torch.zeros((B,1,1,3), device=v_n.device, dtype=v_n.dtype)                         # (B,1,1,3)
            q_offset = torch.zeros((B,1,1,3), device=q_n.device, dtype=q_n.dtype)                       # (B,1,1,3)
            q_boundary = torch.zeros((B,1,1,3), device=v_n.device, dtype=v_n.dtype )

        # Interpolate element nodes to samples
        # *_e: (B,E,Ne,*), N_q: (E,Q,Ne) -> q_e: (B,E,Q,*)
        q_e = torch.einsum('beni,eqn->beqi', q_n, self.N_q.to(q_n.dtype))                   # (B,E,Q,3)
        v_e = torch.einsum('beni,eqn->beqi', v_n - v_base, self.N_q.to(v_n.dtype))          # (B,E,Q,3)


        # # Center of Mass using element densities
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

        L_cm = (m * torch.cross(q_rel, v_rel, dim=-1)).sum(dim=[1,2])                       # (B,3)

        # Compute per-particle inertia contribution
        r2 = (q_rel * q_rel).sum(dim=-1, keepdim=True).reshape(B,E,Q,1,1)                   # (B,E,Q,1,1)
        eye = torch.eye(3, device=q_rel.device, dtype=q_rel.dtype).view(1,1,1,3,3)
        I_per = m.reshape(B,E,Q,1,1) * (r2 * eye - q_rel.reshape(B,E,Q,3,1) * q_rel.reshape(B,E,Q,1,3)) # (B,E,Q,3,3)
        I_cm = I_per.sum(dim=[1,2])  # (B,3,3) 

        Omega_cm = torch.linalg.solve(I_cm, L_cm.reshape(B,3,1)).reshape(B,1,1,3)

        return q_rel, v_e, v_rel, v_rad, v_perp, sL_rel, q_cm, v_cm.expand(B,E,Q,3), Omega_cm.expand(B,E,Q,3)


    @staticmethod
    def _wendland_c2(q):
        # q >= 0; compact support for q < 1
        m = (1.0 - q).clamp(min=0.0, max=1.0)
        return (m ** 4) * (1.0 + 4.0 * q)


    def force_geom(self, q, f, m, q_cm, k = 3, r_mult = 2, eps = 1e-12):

        # Element Sample Positions
        q_e = self.to_elements(q)                                                                   # (B,E,Ne,3)
        q_e = torch.einsum('beni,eqn->beqi', q_e, self.N_q.to(q.dtype))                             # (B,E,Q,3)
        B, E, Q, _ = q_e.shape

        # Element Radii
        V_e = self.element_sample_volume.unsqueeze(0)                                               # (1,E,Q,1)
        h_e = (6.0 * V_e.sum(dim=2, keepdim=True)).clamp_min(eps).pow(1.0 / 3.0)                    # (1,E,1,1)
        r_e = (r_mult * h_e).expand(B, E, Q, 1)                                                     # (B,E,Q,1)
        #print(h_e.squeeze())

        if self.force_nodes is not None:
            q = q[:,self.force_nodes,:]
            f = f[:,self.force_nodes,:]

        # Element Sample Distance to closest Forced Nodes
        P  = q_e.reshape(B, E * Q, 3)                                                               # (B,E*Q,3) 
        dists = torch.zeros(B, E*Q,q.shape[1], dtype = P.dtype, device = P.device)

        batch_size = 256
        for batch in range(B//batch_size):
            dists[:,batch_size * batch: min(batch_size * batch + 1, E*Q),:] = torch.cdist(P[:,batch_size * batch: min(batch_size * batch + 1, E*Q),:], q, p=2)                                                              # (B,E*Q,V)
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


        orig_offset2face = self.offset2face.reshape(1,E,Q,3).expand(B,E,Q,3)
        orig_norm_offset2face = self.norm_offset2face.reshape(1,E,Q,3).expand(B,E,Q,3)

        deformed_offset2face = torch.einsum('beqij,beqj->beqi', F_e, orig_offset2face) 
        offset2face_c = torch.einsum('beqij,beqj->beqi', Rt, deformed_offset2face)
        # orig_norm_offset2COM = self.norm_offset2COM.reshape(1,E,Q,3).expand(B,E,Q,3)

        rho = self.rho.reshape(1,E,1,1).expand(B,E,Q,1)                             # (B,E,Q,1)
        volume = self.element_sample_volume.unsqueeze(0)                            # (B,E,Q,1)
        m = rho * volume                                                            # (B,E,Q,1)

        M = m.sum(dim=[1,2], keepdim=True)

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
        v_rel_mag = v_rel.norm(dim=-1, keepdim=True)                              # (B,E,Q,1)
        v_cm_mag  = v_cm.norm(dim=-1, keepdim=True)                                 # (B,E,Q,1)
        Omega_cm_mag = Omega_cm.norm(dim=-1, keepdim=True)
        
        
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
        

        # print(feat_spin)
        # 11-dim features (co-rotated frame): 
        feat_rigidMotion = torch.cat([
                                    # offset2face_c,
                                    # orig_offset2face,
                                    orig_norm_offset2face,
                                    v_cm_c,
                                    v_cm_mag,
                                    v_rel_c,
                                    v_rel_mag,
                                    # v_e_c,
                                    # v_e_mag,
                                    # Omega_cm_c,
                                    # Omega_cm_mag
                                ], dim=-1)                                          # (B,E,Q,11)
        
        if self.multi_shape:

            max_q = self.q0.reshape(-1,3).max(dim=0)[0]
            min_q = self.q0.reshape(-1,3).min(dim=0)[0]

            length = max_q[0] - min_q[0]
            length_sqrd = length ** 2
            width = max_q[1] - min_q[1]
            thickness = max_q[2] - min_q[2]
            thickness_sqrd = thickness ** 2

            area = width * thickness

            # ratio = (length_sqrd **2 / thickness_sqrd)
            # ratio = (ratio - 0.11111111111) / 0.11111111111

            ratio = (area / length)
            ratio = (ratio - 0.009) / 0.009

            ratio = ratio.reshape(1,1,1,1).expand(B,E,Q,1)

            feat_rigidMotion = torch.cat([feat_rigidMotion, 
                                            ratio], dim=-1)


        if self.actuated:
            f_e, t_e, f_total = self.force_geom(q, f, m, q_cm)             # (B,E,Q,3)
            
            f_e_c = torch.einsum('beqij,beqj->beqi', Rt, f_e)                       # (B,E,Q,3)

            a_e_c = f_e_c / m

            # 3-dim features (co-rotated frame): 
            feat_force = torch.cat([
                                f_e_c,                                              # Effective acceleration due to external force per element (3)
                                a_e_c,
                                m
                                ], dim=-1)                                          # (B,E,Q,3)
        else:
            feat_force = None

        # # Acceleration density scaled by element sample mass

        if not self.stress:
            f_e_c = m * self.unmodelled_nn(feat_deformation,
                                            feat_strainRate,
                                            feat_spin,
                                            feat_rigidMotion,
                                            feat_force)                                 # (B,E,Q,3)

            # Rotate back to world frame
            f_e = torch.einsum('beqij,beqj->beqi', R, f_e_c)                            # (B,E,Q,3)

            # Distribute to nodes
            fn = self.elementForce_to_nodeForce(f_e)                                    # (B,E,Ne,3)
        else:
            P_e_c = self.unmodelled_nn(feat_deformation,
                                            feat_strainRate,
                                            feat_spin,
                                            feat_rigidMotion,
                                            feat_force)                                 # (B,E,Q,3)

            # Rotate back to world frame
            P_e = torch.einsum('beqij,beqjk->beqik', R, P_e_c)                            # (B,E,Q,3)

            fn = self.elementStress_to_nodeForce(P_e)

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