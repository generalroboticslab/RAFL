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



class Unmodelled_Acceleration(nn.Module):
    def __init__(self, hidden_dims, nonlinearity, no_bias=True, normalize_inputs=False):
        super().__init__()
        self.input_dim = 34 

        self.normalize_inputs = normalize_inputs
        if self.normalize_inputs:
            self.normalize = torch.nn.BatchNorm1d(self.input_dim, dtype=torch.float64)

        self.net = _make_mlp(in_dim=self.input_dim, hidden_dims=hidden_dims, out_dim=3,
                             nonlinearity=nonlinearity, no_bias=no_bias)
    

    def input_features(self, feat_deformation, feat_strainRate, feat_spin, feat_force):

        feat = torch.cat([feat_deformation,
                                feat_strainRate,
                                feat_spin,
                                feat_force
                                ], dim=-1)
        

        return feat

    def forward(self, feat_deformation, feat_strainRate, feat_spin, feat_force):  

        
        feat = self.input_features(feat_deformation, 
                                    feat_strainRate, 
                                    feat_spin, 
                                    feat_force)
        orig_shape = feat.shape[:-1]                            # (...,input_dim)
        feat = feat.reshape(-1,self.input_dim)                  # (N,input_dim)

        if self.normalize_inputs:
            norm_feat = self.normalize(feat)
            output = self.net(norm_feat)
        else:
            output = self.net(feat)
        
        a = output.view(*orig_shape,3)

        return a


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
                        gravity=True, 
                        normalize_inputs=False, 
                        ):
        super().__init__()
        
        self.dof = dof                                      # num_vertices * 3
        self.register_buffer('elements', elements.long())   # element vertex index mapping  
        self.register_buffer('faces', faces.long())         # element vertex index mapping  
        self.register_buffer('mu', mu)                      # material Lame parameter mu
        self.register_buffer('lam', lam)                    # material Lame parameter lambda
        self.register_buffer('rho', rho)                    # material density
        self._precompute_quadrature(X_e, dx)                # shape functions for FEM    

        if gravity:
            self.register_buffer('g', torch.tensor([0, 0, -9.80709], dtype=torch.float64))
        

        
        self.unmodelled_nn = Unmodelled_Acceleration(hidden_dims=num_hidden_layer*[hidden_size],
                                                    nonlinearity=nonlinearity,
                                                    normalize_inputs=normalize_inputs,
                                                    no_bias=no_bias)   

    def reset_mesh(self, X_e, elements, faces, dx):

        self.elements = elements.long().to(self.rho.device)
        self.faces = faces.long().to(self.rho.device)
        self._precompute_quadrature(X_e, dx, False)


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
            else:
                self.N_q = torch.stack(N_q, dim=0).unsqueeze(1).to(self.rho.device)
                self.gradN_q = torch.stack(gradN_q, dim=0).unsqueeze(1).to(self.rho.device)
                self.element_sample_volume = torch.stack(element_sample_volume, dim=0).unsqueeze(1).to(self.rho.device)
                self.element_sample_reference_J = torch.stack(element_sample_reference_J, dim=0).unsqueeze(1).to(self.rho.device)

        else:
            # Current implementation only supports hex and tet mesh
            exit()
        

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


    @staticmethod
    def _wendland_c2(q):
        # q >= 0; compact support for q < 1
        m = (1.0 - q).clamp(min=0.0, max=1.0)
        return (m ** 4) * (1.0 + 4.0 * q)


    def force_geom(self, q, f, m):

        # Element Sample Positions
        q_e = self.to_elements(q)                                                                   # (B,E,Ne,3)
        q_e = torch.einsum('beni,eqn->beqi', q_e, self.N_q.to(q.dtype))                             # (B,E,Q,3)
        B, E, Q, _ = q_e.shape

        f_e = self.to_elements(f)                                                                   # (B,E,Ne,3)
        f_q = torch.einsum('beni,eqn->beqi', f_e, self.N_q.to(q.dtype))                             # (B,E,Q,3)
        f_e = f_q.view(B, E, Q, 3) + m * self.g.reshape(1,1,1,3)                                    # (B,E,Q,3)

        return f_e

    def elementForce_to_nodeForce(self, f_e):
        # f_e: (B,E,Q,3), N_q: (E,Q,Ne)  -> fn: (B,E,Ne,3)
        fn = torch.einsum('beqi,eqn->beni', f_e, self.N_q.to(f_e.dtype))    # (B,E,Ne,3)
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

        rho = self.rho.reshape(1,E,1,1).expand(B,E,Q,1)                             # (B,E,Q,1)
        volume = self.element_sample_volume.unsqueeze(0)                            # (B,E,Q,1)
        m = rho * volume                                                            # (B,E,Q,1)
        
        
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

        f_e = self.force_geom(q, f, m)             # (B,E,Q,3)
        
        f_e_c = torch.einsum('beqij,beqj->beqi', Rt, f_e)                       # (B,E,Q,3)

        a_e_c = f_e_c / m

        # 3-dim features (co-rotated frame): 
        feat_force = torch.cat([
                            a_e_c,                                              # Effective acceleration due to external force per element (3)
                            ], dim=-1)                                          # (B,E,Q,3)

        # # Acceleration density scaled by element sample mass

        f_e_c = m * self.unmodelled_nn(feat_deformation,
                                            feat_strainRate,
                                            feat_spin,
                                            feat_force)                                 # (B,E,Q,3)

        # Rotate back to world frame
        f_e = torch.einsum('beqij,beqj->beqi', R, f_e_c)                            # (B,E,Q,3)

        # Distribute to nodes
        fn = self.elementForce_to_nodeForce(f_e)                                    # (B,E,Ne,3)

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