import torch
import torch.nn as nn
import torch.autograd as autograd

from py_diff_pd.core.py_diff_pd_core import StdRealVector, StdIntVector
from py_diff_pd.common.common import ndarray

import numpy as np

class SimFunction(autograd.Function):

    @staticmethod
    def forward(ctx, deformable, dofs, act_dofs, method, q, v, a, f_ext, dt, option, varying_boundary_indices=None, material_mode=None, beta=0.5):

        ctx.deformable = deformable
        ctx.dofs = dofs
        ctx.act_dofs = act_dofs
        ctx.method = method
        ctx.dt = dt
        ctx.option = option
        ctx.varying_boundary_indices = varying_boundary_indices

        q_array = StdRealVector(q.detach().numpy())
        v_array = StdRealVector(v.detach().numpy())
        a_array = StdRealVector(a.detach().numpy())
        f_ext_array = StdRealVector(f_ext.detach().numpy())
        q_next_array = StdRealVector(dofs)
        v_next_array = StdRealVector(dofs)

        if varying_boundary_indices is not None:
            for node_idx in varying_boundary_indices:
                #q_boundary = q[node_idx:node_idx + 3].clone().detach().numpy().copy()
                deformable.SetDirichletBoundaryCondition(node_idx, q_array[node_idx])
                deformable.SetDirichletBoundaryCondition(node_idx + 1, q_array[node_idx + 1])
                deformable.SetDirichletBoundaryCondition(node_idx + 2, q_array[node_idx + 2])

        if material_mode is not None:
            p_array = StdRealVector(material_mode.detach().numpy())
            deformable.PyForward(
                method, q_array, v_array, a_array, f_ext_array, p_array, dt, option, q_next_array, v_next_array, StdIntVector(0))
        else:
            deformable.PyForward(
                method, q_array, v_array, a_array, f_ext_array, dt, option, q_next_array, v_next_array, StdIntVector(0))

        # Store for backward
        q_next = torch.as_tensor(ndarray(q_next_array))
        v_next = torch.as_tensor(ndarray(v_next_array))
        ctx.mat_w_dofs = 2 * deformable.NumOfPdElementEnergies() if material_mode is not None else deformable.NumOfPdElementEnergies()
        ctx.act_w_dofs = deformable.NumOfPdMuscleEnergies()
        ctx.state_p_dofs = deformable.NumOfStateForceParameters()
        ctx.beta = beta

        ctx.save_for_backward(q, v, a, f_ext, q_next, v_next, material_mode)

        return q_next, v_next

    @staticmethod
    def backward(ctx, dl_dq_next, dl_dv_next):

        q, v, a, f_ext, q_next, v_next, material_mode = ctx.saved_tensors
        dofs, act_dofs, mat_w_dofs, act_w_dofs, state_p_dofs = ctx.dofs, ctx.act_dofs, ctx.mat_w_dofs, ctx.act_w_dofs, ctx.state_p_dofs

        q_array = StdRealVector(q.detach().numpy())
        v_array = StdRealVector(v.detach().numpy())
        a_array = StdRealVector(a.detach().numpy())
        f_ext_array = StdRealVector(f_ext.detach().numpy())
        q_next_array = StdRealVector(q_next.detach().numpy())
        v_next_array = StdRealVector(v_next.detach().numpy())

        dl_dq_next_array = StdRealVector(dl_dq_next.detach().numpy())
        dl_dv_next_array = StdRealVector(dl_dv_next.detach().numpy())

        dl_dq_array = StdRealVector(dofs)
        dl_dv_array = StdRealVector(dofs)
        dl_da_array = StdRealVector(act_dofs)
        dl_df_ext_array = StdRealVector(dofs)
        dl_dmat_wi = StdRealVector(mat_w_dofs)
        dl_dact_wi = StdRealVector(act_w_dofs)
        dl_dstate_pi = StdRealVector(state_p_dofs)

        if ctx.varying_boundary_indices is not None:
            for node_idx in ctx.varying_boundary_indices:

                # q_boundary = q[node_idx:node_idx + 3].clone().detach().numpy().copy()
                ctx.deformable.SetDirichletBoundaryCondition(node_idx, q_array[node_idx])
                ctx.deformable.SetDirichletBoundaryCondition(node_idx + 1, q_array[node_idx + 1])
                ctx.deformable.SetDirichletBoundaryCondition(node_idx + 2, q_array[node_idx + 2])

        if material_mode is not None:

            p_array = StdRealVector(material_mode.detach().numpy())
            dl_dp_array = StdRealVector(material_mode.shape[0])
            ctx.deformable.PyBackward(
                ctx.method, q_array, v_array, a_array, f_ext_array, p_array, ctx.dt,
                q_next_array, v_next_array, StdIntVector(0), dl_dq_next_array, dl_dv_next_array, ctx.option,
                dl_dq_array, dl_dv_array, dl_da_array, dl_df_ext_array, dl_dp_array, dl_dmat_wi, dl_dact_wi, dl_dstate_pi)
            
            dl_dq = torch.as_tensor(ndarray(dl_dq_array))
            dl_dv = torch.as_tensor(ndarray(dl_dv_array))
            dl_da = torch.as_tensor(ndarray(dl_da_array))
            dl_df_ext = torch.as_tensor(ndarray(dl_df_ext_array))

            sur_grad = 1 / (ctx.beta * (1 + np.abs(ctx.beta * (ndarray(p_array) - 0.5)))**2)
            
            return (None, None, None, None,
                torch.as_tensor(ndarray(dl_dq)),
                torch.as_tensor(ndarray(dl_dv)),
                torch.as_tensor(ndarray(dl_da)),
                torch.as_tensor(ndarray(dl_df_ext)),
                None, None, None,
                torch.as_tensor(sur_grad * ndarray(dl_dp_array)), None)

        else:

            ctx.deformable.PyBackward(
                ctx.method, q_array, v_array, a_array, f_ext_array, ctx.dt,
                q_next_array, v_next_array, StdIntVector(0), dl_dq_next_array, dl_dv_next_array, ctx.option,
                dl_dq_array, dl_dv_array, dl_da_array, dl_df_ext_array, dl_dmat_wi, dl_dact_wi, dl_dstate_pi)

            dl_dq = torch.as_tensor(ndarray(dl_dq_array))
            dl_dv = torch.as_tensor(ndarray(dl_dv_array))
            dl_da = torch.as_tensor(ndarray(dl_da_array))
            dl_df_ext = torch.as_tensor(ndarray(dl_df_ext_array))

            return (None, None, None, None,
                torch.as_tensor(ndarray(dl_dq)),
                torch.as_tensor(ndarray(dl_dv)),
                torch.as_tensor(ndarray(dl_da)),
                torch.as_tensor(ndarray(dl_df_ext)),
                None, None, None,
                None, None)



class Sim(nn.Module):
    def __init__(self, deformable):
        super(Sim, self).__init__()
        self.deformable = deformable

    def forward(self, dofs, act_dofs, method, q, v, a, f_ext, dt, option, varying_boundary_indices=None, material_mode=None, beta=0.5):
        return SimFunction.apply(
            self.deformable, dofs, act_dofs, method, q, v, a, f_ext, dt, option, varying_boundary_indices, material_mode, beta)