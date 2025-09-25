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

class Unmodelled_Acceleration(nn.Module):
    def __init__(self, hidden_dims=[128,128,128,128], nonlinearity='elu', no_bias=False):
        super().__init__()
        self.normalize = torch.nn.BatchNorm1d(41, dtype=torch.float64)
        self.net = _make_mlp(in_dim=41, hidden_dims=hidden_dims, out_dim=3,
                             nonlinearity=nonlinearity, no_bias=no_bias)
    def forward(self, feat, v_e, a_force):  
        orig_shape = feat.shape[:-1]                                            # (...,41)
        feat = feat.reshape(-1,41)                                              # (N,41)
        norm_feat = self.normalize(feat)                                        # (N,41)
        a = self.net(feat).view(*orig_shape,3)                                  # (B,E,Q,3)

        # # Make sure no extra energy is introduced in non forced directions
        # f_hat = a_force / a_force.norm(dim=-1, keepdim=True).clamp_min(1e-6)    # (B,E,Q,3) 
        # v_f = (v_e * f_hat).sum(dim=-1, keepdim=True) * f_hat                   # (B,E,Q,3)
        # v_nf = v_e - v_f                                                        # (B,E,Q,3)
        # v_nf_hat = v_nf / (v_nf.norm(dim=-1, keepdim=True) + 1e-12)             # (B,E,Q,3)
        # dot = (a * v_nf).sum(dim=-1, keepdim=True)                              # (B,E,Q,1) 
        # a_dis = a - torch.relu(dot) * v_nf_hat                                  # (B,E,Q,3)
        return a


class ElementResidual(nn.Module):
    def __init__(self, dof, elements, mu, lam, rho, X_e, dx, g, hidden_size=128, num_hidden_layer=4):
        super().__init__()
        
        self.dof = dof                                      # num_vertices * 3
        self.register_buffer('elements', elements.long())   # element vertex index mapping             
        self.register_buffer('mu', mu)                      # material Lame parameter mu
        self.register_buffer('lam', lam)                    # material Lame parameter lambda
        self.register_buffer('rho', rho)                    # material density
        self.register_buffer('g', g)                        # gravity
        self._precompute_quadrature(X_e, dx)                # shape functions for FEM            

        # 41-dim features -> acceleration density
        self.unmodelled_nn = Unmodelled_Acceleration(hidden_dims=num_hidden_layer*[hidden_size])     

    @staticmethod
    def independent(X: Tensor) -> Tensor:
        
        return torch.stack([X[...,0,0], X[...,0,1], X[...,0,2], 
                                        X[...,1,1], X[...,1,2], 
                                                    X[...,2,2]], dim=-1)

    @staticmethod
    def axial(X: Tensor) -> Tensor:

        return torch.stack([X[...,2,1], X[...,0,2], X[...,1,0]], dim=-1)

    def _precompute_quadrature(self, X_e, dx):
        
        E,Ne = self.elements.shape

        if Ne == 8:

            self._mesh_type = 'hex'

            # Define element samples
            samples = torch.tensor([[0.,0.,0.,0.,1.,1.,1.,1.],[0.,0.,1.,1.,0.,0.,1.,1.],[0.,1.,0.,1.,0.,1.,0.,1.]]).T
            samples -= 0.5 * torch.ones_like(samples)
            samples /= np.sqrt(3)
            samples += 0.5 * torch.ones_like(samples)
            samples *= dx 

            N_q = []                    # Shape function N
            gradN_q = []                # Shape function gradient ∇N
            element_sample_volume = []  # Element sample volume
            for e in range(E):

                N_q_e = []
                gradN_q_e = []

                # Iterate through element samples
                # Element Node order: [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]
                for s in range(8):
                    inv_dx = 1 / dx
                    x, y, z = samples[s,0], samples[s,1], samples[s,2]
                    nx, ny, nz = x * inv_dx, y * inv_dx, z * inv_dx
                    cnx, cny, cnz = 1 - nx, 1 - ny, 1 - nz

                    # Shape function N for sample s
                    N_q_e.append(
                                [
                                        cnx*cny*cnz,            
                                        cnx*cny*nz,             
                                        cnx*ny*cnz,             
                                        cnx*ny*nz,              
                                        nx*cny*cnz,             
                                        nx*cny*nz,              
                                        nx*ny*cnz,              
                                        nx*ny*nz,               
                                    ]
                                )
                    # Shape function gradient ∇N for sample s
                    gradN_q_e.append(
                                [[-inv_dx * cny * cnz, cnx * -inv_dx * cnz, cnx * cny * -inv_dx],   
                                [-inv_dx * cny * nz, cnx * -inv_dx * nz, cnx * cny * inv_dx],       
                                [-inv_dx * ny * cnz, cnx * inv_dx * cnz, cnx * ny * -inv_dx],       
                                [-inv_dx * ny * nz, cnx * inv_dx * nz, cnx * ny * inv_dx],          
                                [inv_dx * cny * cnz, nx * -inv_dx * cnz, nx * cny * -inv_dx],       
                                [inv_dx * cny * nz, nx * -inv_dx * nz, nx * cny * inv_dx],          
                                [inv_dx * ny * cnz, nx * inv_dx * cnz, nx * ny * -inv_dx],          
                                [inv_dx * ny * nz, nx * inv_dx * nz, nx * ny * inv_dx]]             
                                )

                N_q.append(torch.tensor(N_q_e, dtype=torch.float64))
                gradN_q.append(torch.tensor(gradN_q_e, dtype=torch.float64))
                element_sample_volume.append(torch.tensor(8*[dx**3 / 8], dtype=torch.float64).reshape(8,1))

            self.register_buffer('N_q', torch.stack(N_q, dim=0))                                        #(E,Q,Ne)
            self.register_buffer('gradN_q', torch.stack(gradN_q, dim=0))                                #(E,Q,Ne,3)
            self.register_buffer('element_sample_volume', torch.stack(element_sample_volume, dim=0))    #(E,Q,1)

        elif Ne == 4:
            
            self._mesh_type = 'tet'

            N_q = []                    # Shape function N
            gradN_q = []                # Shape function gradient ∇N
            element_sample_volume = []  # Element sample volume
            for e in range(E):

                N_q.append(torch.tensor(4 * [0.25], dtype=torch.float64))                           #(Ne)

                X0 = X_e[3*self.elements[e][0]:3*self.elements[e][0]+3]
                X1 = X_e[3*self.elements[e][1]:3*self.elements[e][1]+3]
                X2 = X_e[3*self.elements[e][2]:3*self.elements[e][2]+3]
                X3 = X_e[3*self.elements[e][3]:3*self.elements[e][3]+3]

                J = torch.stack([X1 - X0, X2 - X0, X3 - X0], dim=1)                                 # (3,3)
                B = torch.linalg.inv(J)                                                         # (3,3)
                gradN_q.append(torch.stack([-B[:,0] - B[:,1] - B[:,2], B[:,0], B[:,1], B[:,2]]))    # (Ne,3)

                vol_e = torch.abs(torch.det(J)) / 6.0                                               # scalar
                element_sample_volume.append(vol_e.reshape(1))                                      # (1)

            self.register_buffer('N_q', torch.stack(N_q, dim=0).unsqueeze(1))                                       #(E,Q,Ne)
            self.register_buffer('gradN_q', torch.stack(gradN_q, dim=0).unsqueeze(1))                               #(E,Q,Ne,3)
            self.register_buffer('element_sample_volume', torch.stack(element_sample_volume, dim=0).unsqueeze(1))   #(E,Q,1)
        else:
            # Current implementation only supports hex and tet mesh
            exit()

    def to_elements(self, q):
        # q: (B,V,3) -> (B,E,Ne,3)
        return q[:, self.elements, :]
    

    def element_deformationGrad(self, q):
        q_e = self.to_elements(q)                                               # (B,E,Ne,3)
        
        # q_e: (B,E,Ne,3), gradN_q: (E,Q,Ne,3) -> F: (B,E,Q,3,3)
        F_e = torch.einsum('beni,eqnj->beqij', q_e, self.gradN_q.to(q_e.dtype)) # (B,E,Q,3,3)

        return F_e

    def element_velocityGrad_corotated(self, v, U, sigma, Vt, eps=1e-6):
        v_e = self.to_elements(v)                                                       # (B,E,Ne,3)

        # v_e: (B,E,Ne,3), gradN_q: (E,Q,Ne,3) -> grad_v_e: (B,E,Q,3,3)
        grad_v_e = torch.einsum('beni,eqnj->beqij', v_e, self.gradN_q.to(v_e.dtype))    # (B,E,Q,3,3)

        Ut = U.transpose(-1,-2)                                                         # (B,E,Q,3,3)
        sigma_inv = torch.diag_embed(1.0 / sigma.clamp_min(eps))                        # (B,E,Q,3,3)
        V = Vt.transpose(-1,-2)                                                         # (B,E,Q,3,3)

        G = Ut @ grad_v_e @ V                                                           # (B,E,Q,3,3)
        L_c = V @ (G @ sigma_inv) @ V.transpose(-1, -2)                                 # (B,E,Q,3,3)

        return L_c

    def get_deformation_svd(self, F_e):

        U, sigma, Vt = torch.linalg.svd(F_e)                                    # (B,E,Q,3,3),(B,E,Q,3),(B,E,Q,3,3)
        detR = torch.det(U @ Vt)                                                # (B,E,Q)
        fix = (detR < 0).unsqueeze(-1).unsqueeze(-1)                            # (B,E,Q,1,1)
        I = torch.eye(3, device=F_e.device, dtype=F_e.dtype)                    # (3,3)
        I[2, 2] = -1.0                                                          # (3,3)
        I = I.expand_as(U)                                                      # (B,E,Q,3,3)
        U_corr  = torch.where(fix, U @ I, U)                                    # (B,E,Q,3,3)
        Vt_corr = torch.where(fix, I @ Vt, Vt)                                  # (B,E,Q,3,3)

        return U_corr, sigma, Vt_corr

    def element_geom(self, q, v, f):
        q_e = self.to_elements(q)                                               # (B,E,Ne,3)
        v_e = self.to_elements(v)                                               # (B,E,Ne,3)
        f_e = self.to_elements(f)                                               # (B,E,Ne,1)
        B,E,Ne,D = q_e.shape
        Q = self.N_q.shape[1]

        # Interpolate element nodes to samples
        # *_e: (B,E,Ne,*), N_q: (E,Q,Ne) -> q_e: (B,E,Q,*)
        q_e = torch.einsum('beni,eqn->beqi', q_e, self.N_q.to(q_e.dtype))       # (B,E,Q,3)
        v_e = torch.einsum('beni,eqn->beqi', v_e, self.N_q.to(v_e.dtype))       # (B,E,Q,3)
        f_e = torch.einsum('beni,eqn->beqi', f_e, self.N_q.to(f_e.dtype))       # (B,E,Q,1)

        # Center of Mass using element densities
        rho = self.rho.reshape(1,E,1,1).expand(1,E,Q,1)                         # (1,E,Q,1)
        volume = self.element_sample_volume.unsqueeze(0)                        # (1,E,Q,1)
        M = (volume * rho).sum(dim=[1,2]).reshape(1,1,1,1).expand(B,1,1,1)      # (B,1,1,1)
        q_sum = (volume * rho * q_e).sum(dim=[1,2]).reshape(B,1,1,3)            # (B,1,1,3)
        v_sum = (volume * rho * v_e).sum(dim=[1,2]).reshape(B,1,1,3)            # (B,1,1,3)
        q_cm = q_sum / M                                                        # (B,1,1,3)
        v_cm = v_sum / M                                                        # (B,1,1,3)

        # Relative position and velocity wrt Center of Mass
        q_rel = q_e - q_cm                                                      # (B,E,Q,3)
        v_rel = v_e - v_cm                                                      # (B,E,Q,3)

        return q_rel, v_rel, v_cm.expand(B,E,Q,3), f_e

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
        q_rel, v_rel, v_cm, f_e = self.element_geom(q, v, f)                                    # (B,E,Q,3)
        B,E,Q,_ = q_rel.shape

        I = torch.eye(3, device=q.device, dtype=q.dtype)                                        # (1,1,1,3,3)

        F_e = self.element_deformationGrad(q)                                                   # (B,E,Q,3,3)
        U, sigma, Vt = self.get_deformation_svd(F_e)                                            # (B,E,Q,3,3),(B,E,Q,3),(B,E,Q,3,3)
        R = U @ Vt                                                                              # (B,E,Q,3,3)
        Rt = R.transpose(-1, -2)                                                                # (B,E,Q,3,3)

        rho = self.rho.reshape(1,E,1,1).expand(B,E,Q,1)                                         # (B,E,Q,1)
        volume = self.element_sample_volume.unsqueeze(0)                                        # (1,E,Q,1)
        m = rho * volume                                                                        # (B,E,Q,1)
        a_e = f_e / m  +  self.g.reshape(1,1,1,3)                                               # (B,E,Q,3)

        q_rel_c = torch.einsum('beqij,beqj->beqi', Rt, q_rel)                                   # (B,E,Q,3)
        v_rel_c = torch.einsum('beqij,beqj->beqi', Rt, v_rel)                                   # (B,E,Q,3)
        v_cm_c = torch.einsum('beqij,beqj->beqi', Rt, v_cm)                                     # (B,E,Q,3)
        a_e_c = torch.einsum('beqij,beqj->beqi', Rt, a_e)                                       # (B,E,Q,3)
        
        I1_F = sigma - 1.0                                                                      # (B,E,Q,3)
        Cmat = F_e.transpose(-1,-2) @ F_e                                                       # (B,E,Q,3,3)
        I2_F  = self.independent(Cmat -  I)                                                     # (B,E,Q,6)
        J = torch.linalg.det(F_e).unsqueeze(-1)                                                 # (B,E,Q,1)
        I3_F = J - 1.0                                                                          # (B,E,Q,1)
        
        L_e_c = self.element_velocityGrad_corotated(v, U, sigma, Vt)                            # (B,E,Q,3,3)

        Dc = 0.5 * (L_e_c + L_e_c.transpose(-1,-2))                                             # (B,E,Q,3,3)
        
        I1_D = Dc.diagonal(dim1=-2, dim2=-1)                                                    # (B,E,Q,3)
        I2_D = I1_D.sum(dim=-1, keepdim=True)                                                   # (B,E,Q,1)
        D_dev = Dc - (I2_D.view(B,E,Q,1,1)/3.0) * I.view(1,1,1,3,3)                             # (B,E,Q,3,3)
        I3_D = self.independent(D_dev)                                                          # (B,E,Q,6)
        I4_D = torch.sqrt(0.5 * (D_dev*D_dev).sum(dim=(-2,-1))).unsqueeze(-1)                   # (B,E,Q,1)

        Wc = 0.5 * (L_e_c - L_e_c.transpose(-1,-2))                                             # (B,E,Q,3,3)
        
        I1_W = self.axial(Wc)                                                                   # (B,E,Q,3)
        I2_W = (2 * I1_W).norm(dim=-1, keepdim=True)                                            # (B,E,Q,1)
        
        mu = self.mu.reshape(1,E,1,1).expand(B,E,Q,1)                                           # (B,E,Q,1)
        muN = torch.log(mu)                                                                     # (B,E,Q,1)

        lam = self.lam.reshape(1,E,1,1).expand(B,E,Q,1)                                         # (B,E,Q,1)
        lamN = torch.log(lam)                                                                   # (B,E,Q,1)

        v_rel_mag = v_rel.norm(dim=-1, keepdim=True)                                            # (B,E,Q,1)
        v_cm_mag  = v_cm.norm(dim=-1, keepdim=True)                                             # (B,E,Q,1)

        # 41-dim features (co-rotated frame): 
        feat = torch.cat([q_rel_c,                                                              # relative position (3)
                            v_rel_c,                                                            # relative velocity (3)
                            v_cm_c,                                                             # COM velocity (3)
                            I1_F,                                                               # sigma - 1 (3)
                            I2_F,                                                               # independent(C - I) (6)
                            I3_F,                                                               # det(F) - 1 (1)
                            I1_D,                                                               # diag(Dc) (3)
                            I2_D,                                                               # trace(Dc) (1)
                            I3_D,                                                               # independent(dev(Dc)) (6)
                            I4_D,                                                               # sqrt(0.5 * ||dev(Dc)||_F^2) (1)
                            I1_W,                                                               # axial(W) (3)
                            I2_W,                                                               # vorticity magnitude (1)
                            a_e_c,                                                              # external force acceleration (3)
                            muN,                                                                # log mu (1)
                            lamN,                                                               # log lambda (1)
                            v_rel_mag,                                                          # relative speed (1)
                            v_cm_mag,                                                           # COM speed (1)
                            ], dim=-1)                                                          # (B,E,Q,41)

        # Acceleration density scaled by element sample mass
        f_e_c = m * self.unmodelled_nn(feat, v_rel_c + v_cm_c, a_e_c)                           # (B,E,Q,3)
        
        # Rotate back to world frame
        f_e = torch.einsum('beqij,beqj->beqi', R, f_e_c)                                        # (B,E,Q,3)

        # Distribute to nodes
        fn = self.elementForce_to_nodeForce(f_e)                                                # (B,E,Ne,3)

        return self.elementNode_to_dof(fn)                                                      # (B,3V)    
           

    def forward(self, qvf):
        if qvf.dim() == 1:
            qvf = qvf.unsqueeze(0)
        B = qvf.shape[0]

        if qvf.shape[1] == 2 * self.dof:
            # No force provided
            q_flat, v_flat = torch.split(qvf, [self.dof, self.dof], dim=-1)
            f_flat = torch.zeros((B, self.dof), dtype=q_flat.dtype).to(q_flat.device)
        else:
            q_flat, v_flat, f_flat = torch.split(qvf, [self.dof, self.dof, self.dof], dim=-1)
        
        q = q_flat.view(B, -1, 3)                   # Position
        v = v_flat.view(B, -1, 3)                   # Velocity
        f = f_flat.view(B, -1, 3)                   # Applied External Force

        f_umdld = self.force_unmodelled(q, v, f)    # Unmodelled forces  
        
        return f_umdld 

    def count_parameters(self):
        
        n = sum(p.numel() for p in self.unmodelled_nn.parameters() if p.requires_grad)
        return n