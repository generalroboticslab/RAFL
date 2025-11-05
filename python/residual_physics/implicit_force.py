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

class Feat_Encoder(nn.Module):
    def __init__(self, dof, hidden_dims=[512,512,512,512], latent_dim=16, nonlinearity='elu', no_bias=True):
        super().__init__()

        self.input_dim = dof
        self.latent_dim = latent_dim
        self.net = _make_mlp(in_dim=self.input_dim, hidden_dims=hidden_dims, out_dim=latent_dim,
                             nonlinearity=nonlinearity, no_bias=no_bias)
    

    def forward(self, x):  

        x = x.reshape(-1,self.input_dim)
        feat = self.net(x)
        return feat


class SirenLayer(torch.nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False, initialize=True):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = torch.nn.Linear(in_f, out_f, dtype=torch.float64)
        self.is_first = is_first
        self.is_last = is_last
        if initialize:
            self.init_weights()
    
    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


class Force_Decoder(nn.Module):

    def __init__(self, actuated, latent_dim=16):
        super().__init__()
        self.actuated = actuated
        self.input_dim = 3 * latent_dim + 3 if self.actuated else 2 * latent_dim + 3

        self.normalize = torch.nn.BatchNorm1d(self.input_dim - 3, dtype=torch.float64)
        self.layer1 = SirenLayer(self.input_dim, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 3,is_last=True)

    def decoder(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, feat, q_o):  

        orig_shape = q_o.shape[:-1]                                                                 # (...,input_dim)
        feat_dim = feat.shape[-1]

        normalized_feat = self.normalize(feat)

        normalized_feat = normalized_feat.unsqueeze(1).unsqueeze(1).expand(*orig_shape,feat_dim)    # (B,E,Q,feat_dim)

        combined_input = torch.cat([normalized_feat, q_o], dim=-1).reshape(-1,self.input_dim)       # (B,E,Q,feat_dim + 3)

        a = self.decoder(combined_input).view(*orig_shape,3)

        return a

class ImplicitResidual(nn.Module):
    def __init__(self, dof, elements, rho, X_e, dx, hidden_size=128, num_hidden_layer=4, nonlinearity='elu', no_bias=True, actuated=False, gravity=True, scale=1):
        super().__init__()
        
        self.dof = dof                                      # num_vertices * 3
        self.register_buffer('elements', elements.long())   # element vertex index mapping  
        self.register_buffer('rho', rho)                    # material density
        self._precompute_quadrature(X_e, dx)                # shape functions for FEM            

        self.actuated = actuated
        if self.actuated:
            if gravity:
                self.register_buffer('g', scale * torch.tensor([0, 0, -9.80709], dtype=torch.float64))
            else:
                self.register_buffer('g', torch.tensor([0, 0, 0], dtype=torch.float64))

        self.pos_encoder = Feat_Encoder(self.dof)
        self.vel_encoder = Feat_Encoder(self.dof)

        if actuated:
            self.force_encoder = Feat_Encoder(self.dof)

        # features -> acceleration density
        self.force_decoder = Force_Decoder(actuated=actuated)   
    
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
                B = torch.linalg.inv(J)                                                             # (3,3)
                gradN_q.append(torch.stack([-B[:,0] - B[:,1] - B[:,2], B[:,0], B[:,1], B[:,2]]))    # (Ne,3)

                vol_e = torch.abs(torch.det(J)) / 6.0                                               # scalar
                element_sample_volume.append(vol_e.reshape(1))                                      # (1)

            self.register_buffer('N_q', torch.stack(N_q, dim=0).unsqueeze(1))                                       #(E,Q,Ne)
            self.register_buffer('gradN_q', torch.stack(gradN_q, dim=0).unsqueeze(1))                               #(E,Q,Ne,3)
            self.register_buffer('element_sample_volume', torch.stack(element_sample_volume, dim=0).unsqueeze(1))   #(E,Q,1)

        else:
            # Current implementation only supports hex and tet mesh
            exit()
        
        X_e = X_e.reshape(-1,3)
        q_mean = X_e.mean(dim=0,keepdim=True)
        q_std = X_e.std(dim=0, keepdim=True)

        X_e_normalized = (X_e - q_mean) / q_std

        q_e = self.to_elements(X_e_normalized.unsqueeze(0)).squeeze(0) #(E,Ne,3)

        q_o = torch.einsum('eni,eqn->eqi', q_e, self.N_q.to(q_e.dtype))
        self.register_buffer('q_o', q_o)    #(E,Q,3)
    
    def to_elements(self, q):
        # q: (B,V,3) -> (B,E,Ne,3)
        return q[:, self.elements, :]
    
    def element_geom(self, q, v, m, eps = 1e-12):
        q_e = self.to_elements(q)                                                           # (B,E,Ne,3)
        v_e = self.to_elements(v)                                                           # (B,E,Ne,3)
        B,E,Ne,D = q_e.shape
        Q = self.N_q.shape[1]

        # Interpolate element nodes to samples
        # *_e: (B,E,Ne,*), N_q: (E,Q,Ne) -> q_e: (B,E,Q,*)
        q_e = torch.einsum('beni,eqn->beqi', q_e, self.N_q.to(q_e.dtype))                   # (B,E,Q,3)
        v_e = torch.einsum('beni,eqn->beqi', v_e, self.N_q.to(v_e.dtype))                   # (B,E,Q,3)
        
        return q_e, v_e
    
    def element_deformationGrad(self, q):
        q_e = self.to_elements(q)                                               # (B,E,Ne,3)
        
        # q_e: (B,E,Ne,3), gradN_q: (E,Q,Ne,3) -> F: (B,E,Q,3,3)
        F_e = torch.einsum('beni,eqnj->beqij', q_e, self.gradN_q.to(q_e.dtype)) # (B,E,Q,3,3)

        return F_e

    def R_from_F(self, F, eps=1e-12):
        
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

        return R

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

    def implicit_force(self, q, v, f):

        F_e = self.element_deformationGrad(q)               # (B,E,Q,3,3)
        B,E,Q,_,_ = F_e.shape

        R = self.R_from_F(F_e)                              # (B,E,Q,3,3)

        q_feat = self.pos_encoder(q)                        # (B,L)
        v_feat = self.vel_encoder(v)                        # (B,L)
        if self.actuated:
            f_feat = self.force_encoder(f)                      # (B,L)
            feat = torch.cat([q_feat, v_feat, f_feat], dim=-1)  # (B,3*L)
        else:
            feat = torch.cat([q_feat, v_feat], dim=-1)      # (B,3*L)

        q_o = self.q_o.unsqueeze(0).expand(B,E,Q,3)         # (B,E,Q,3)

        rho = self.rho.reshape(1,E,1,1).expand(B,E,Q,1)     # (B,E,Q,1)
        volume = self.element_sample_volume.unsqueeze(0)    # (B,E,Q,1)
        m = rho * volume                                    # (B,E,Q,1)
        
        f_e = self.force_decoder(feat, q_o)                 # (B,E,Q,3)

        # Distribute to nodes
        fn = self.elementForce_to_nodeForce(f_e)            # (B,E,Ne,3)

        return self.elementNode_to_dof(fn)                  # (B,3V) 


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

        f_res = self.implicit_force(q, v, f_ext)
        
        return f_res 

    def count_parameters(self):
        
        n = sum(p.numel() for p in self.unmodelled_nn.parameters() if p.requires_grad)
        return n
    