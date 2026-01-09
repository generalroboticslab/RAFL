import sys

sys.path.append("../..")

import argparse
from pathlib import Path
import numpy as np
import torch
from env_hanging_cantilever import HangingCantileverEnv3d
from video_generation import *
from tqdm import tqdm
from py_diff_pd.core.py_diff_pd_core import StdRealVector, StdIntVector
from py_diff_pd.common.common import ndarray, create_folder, copy_std_int_vector
from losses import trajectory_loss_and_grad, stepwise_loss_and_grad
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

def plot_materialMode_views(fname, p, element_pos):
    """
    Show four orthographic views of element material modes in a tight 2x2 layout,
    plus a tall subplot on the right showing only the middle (interior) elements.

    Assumes:
        - element_pos: (N, 3) array of element centers (global variable)
        - p: array-like of length N with continuous values in [0,1] (0 soft, 1 stiff threshold)
    """

    # binarize modes
    p = np.array([0 if p[i] < 0.5 else 1 for i in range(p.shape[0])]).astype(int)
    cmap = ListedColormap(["#d62728", "#1f77b4"])  # stiff=red, soft=blue

    # helper: estimate grid spacing for tolerance
    def _grid_step(vals):
        u = np.unique(np.round(vals, 8))
        if len(u) <= 1:
            return 1.0
        diffs = np.diff(u)
        diffs = diffs[diffs > 1e-10]
        return diffs.min() if len(diffs) > 0 else 1.0

    # ------ bounding box ------
    x = element_pos[:, 0]
    y = element_pos[:, 1]
    z = element_pos[:, 2]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    # for equal-ish aspect of the prism
    max_range = (np.array([
        x_max - x_min,
        y_max - y_min,
        z_max - z_min
    ]).max()) / 2.0

    mid_x = 0.5 * (x_min + x_max)
    mid_y = 0.5 * (y_min + y_max)
    mid_z = 0.5 * (z_min + z_max)

    # ------ middle (interior) elements: not on any outer face ------
    dx = _grid_step(x)
    dy = _grid_step(y)
    dz = _grid_step(z)

    ex = 0.5 * dx
    ey = 0.5 * dy
    ez = 0.5 * dz

    on_min_x = x <= x_min + ex
    on_max_x = x >= x_max - ex
    on_min_y = y <= y_min + ey
    on_max_y = y >= y_max - ey

    surface_mask = on_min_x | on_max_x | on_min_y | on_max_y 
    middle_mask = ~surface_mask

    middle_pos   = element_pos[middle_mask]
    middle_modes = p[middle_mask]

    # ------ figure and manual axes positions ------
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("Material Modes", fontsize=12, y=0.96)

    # Left 2×2 grid (slightly narrower to make room for tall right panel)
    # [left, bottom, width, height] in figure coords
    pos_top_left     = [0.04, 0.46, 0.40, 0.48]
    pos_top_right    = [0.38, 0.46, 0.40, 0.48]
    pos_bottom_left  = [0.04, 0.00, 0.40, 0.48]
    pos_bottom_right = [0.38, 0.00, 0.40, 0.48]

    # Tall right subplot (double height)
    pos_middle_panel = [0.76, 0.00, 0.20, 0.96]

    ax1 = fig.add_axes(pos_top_left,     projection='3d')
    ax2 = fig.add_axes(pos_top_right,    projection='3d')
    ax3 = fig.add_axes(pos_bottom_left,  projection='3d')
    ax4 = fig.add_axes(pos_bottom_right, projection='3d')
    axM = fig.add_axes(pos_middle_panel, projection='3d')

    # Views: (ax, title, elev, azim, hide_axis)
    views = [
        (ax1, "XZ (from -Y→+Y)", 0,  -90, "y", on_min_y, False),
        (ax2, "YZ (from -X→+X)", 0,    0, "x", on_min_x, True),
        (ax3, "XZ (from +Y→-Y)", 0,   90, "y", on_max_y, False),
        (ax4, "YZ (from +X→-X)", 0,  180, "x", on_max_x, True),
    ]

    for ax, title, elev, azim, hide_axis, mask, flip_h in views:

        pos_view = element_pos[mask]
        p_view   = p[mask]

        # turn off panes & grid for a cleaner look
        ax.grid(False)
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)

        # scatter only the visible front elements
        ax.scatter(
            pos_view[:, 0],
            pos_view[:, 1],
            pos_view[:, 2],
            c=p_view,
            s=10,
            alpha=0.9,
            cmap=cmap,
            vmin=0,
            vmax=1,
        )


        # equal-ish aspect
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if flip_h:
            ax.set_ylim(mid_y + max_range, mid_y - max_range)

        # camera
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, pad=-4, fontsize=8)

        # hide the axis going into the screen, label the others
        if hide_axis == "y":    # XZ views
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.set_xlabel("X", labelpad=0, fontsize=8)
            ax.set_zlabel("Z", labelpad=0, fontsize=8)
        else:                   # YZ views
            ax.set_xticks([])
            ax.set_xlabel("")
            ax.set_ylabel("Y", labelpad=0, fontsize=8)
            ax.set_zlabel("Z", labelpad=0, fontsize=8)


        ax.tick_params(axis='both', which='major', labelsize=7, pad=0)
        ax.tick_params(axis='z', which='major', labelsize=7, pad=0)


    # ------ Tall right subplot: middle (interior) elements only ------
    axM.grid(False)
    axM.xaxis.pane.set_visible(False)
    axM.yaxis.pane.set_visible(False)
    axM.zaxis.pane.set_visible(False)

    # scatter only interior elements
    axM.scatter(
        middle_pos[:, 0],
        middle_pos[:, 1],
        middle_pos[:, 2],
        c=middle_modes,
        s=10,
        alpha=0.9,
        cmap=cmap,
        vmin=0,
        vmax=1,
    )


    # same aspect as others
    axM.set_xlim(mid_x - max_range, mid_x + max_range)
    axM.set_ylim(mid_y - max_range, mid_y + max_range)
    axM.set_zlim(mid_z - max_range, mid_z + max_range)

    axM.view_init(elev=0, azim=-90)  
    axM.set_title("Middle (interior) elements", pad=-4, fontsize=8)

    axM.set_zlabel("Z", fontsize=8, labelpad=0)

    axM.set_yticks([])
    axM.set_xticks([])
    axM.tick_params(axis='z', which='major', labelsize=7, pad=0)

    plt.savefig(fname)
    plt.close()

def plot_lossCurve(objective, losses):

    min_epoch = np.argmin(losses)
    fig = plt.figure()
    plt.plot(np.arange(len(losses)), losses, label="Loss")
    plt.scatter([min_epoch], [losses[min_epoch]], color='red', s=60, label=f"Min loss (epoch {min_epoch})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(np.arange(0, len(losses), len(losses)//10 if len(losses) > 10 else 1))

    plt.savefig(f"{objective}_optimize/losses.png")
    plt.close()
    

def sim_forward(objective, p, sim_q0, sim_v0, frame_num, deformable, boundary_indices, forward_method, forward_opt, dt=0.01, amplitude=0.003, frequency=15, prepare = 200):
    
    oscillation_offset = lambda t : amplitude * np.sin(2 * np.pi * frequency * t)
    oscillation_vel = lambda t: 2 * np.pi * frequency * amplitude * np.cos(2 * np.pi * frequency * t)

    sim_act = [ndarray(np.zeros(deformable.act_dofs())) for _ in range(prepare + frame_num)]
    sim_f_ext = [ndarray(np.zeros(deformable.dofs())) for _ in range(prepare + frame_num)]

    q = [sim_q0,]
    v = [sim_v0,]

    p_array = StdRealVector(p.clone().detach().numpy())
    dofs = deformable.dofs()
    loss = 0
    active_contact_indices = [StdIntVector(0),]
    for i in range(prepare + frame_num):
        
        t = (i + 1) * dt

        vx_offset =  oscillation_offset(t)
        vx_vel = oscillation_vel(t)

        q_cur_array = q[-1].copy()
        v_cur_array = v[-1].copy()

        for node_idx in boundary_indices:
            vx, vy, vz = sim_q0[node_idx : node_idx + 3] 
            deformable.SetDirichletBoundaryCondition(node_idx, vx + vx_offset)
            deformable.SetDirichletBoundaryCondition(node_idx + 1, vy)
            deformable.SetDirichletBoundaryCondition(node_idx + 2, vz)

            q_cur_array[node_idx] = vx + vx_offset
            q_cur_array[node_idx + 1] = vy
            q_cur_array[node_idx + 2] = vz
            v_cur_array[node_idx] = vx_vel
            v_cur_array[node_idx + 1] = 0
            v_cur_array[node_idx + 2] = 0
        
        q_array = StdRealVector(q_cur_array)
        v_array = StdRealVector(v_cur_array)

        q_next_array = StdRealVector(dofs)
        v_next_array = StdRealVector(dofs)
        active_contact_idx = copy_std_int_vector(active_contact_indices[-1])

        deformable.PyForward(forward_method, q_array, v_array, sim_act[i], sim_f_ext[i], p_array, dt, forward_opt,
                q_next_array, v_next_array, active_contact_idx)
        q_next = ndarray(q_next_array)
        v_next = ndarray(v_next_array)
        active_contact_indices.append(active_contact_idx)

        ret = stepwise_loss_and_grad(objective, q_next, v_next, i + 1, frame_num)
        l, _, _ = ret[:3]

        if i >= prepare:
            loss += l

        q.append(q_next)
        v.append(v_next)
    
    full_ret = trajectory_loss_and_grad(objective,q,v)
    full_l, _, _ = full_ret[:3]
    loss += full_l

    return loss, q, v, sim_act, sim_f_ext, active_contact_indices

    
def sim_backward(objective, p, q, v, sim_act, sim_f_ext, active_contact_indices, sim_q0, sim_v0, frame_num, deformable, boundary_indices, backward_method, backward_opt, dt=0.01, amplitude=0.003, frequency=15, beta=0.5, prepare = 200):

    oscillation_offset = lambda t : amplitude * np.sin(2 * np.pi * frequency * t)
    oscillation_vel = lambda t: 2 * np.pi * frequency * amplitude * np.cos(2 * np.pi * frequency * t)

    ret = stepwise_loss_and_grad(objective,q[-1], v[-1], prepare + frame_num, frame_num)
    grad_q, grad_v = ret[1], ret[2]

    full_ret = trajectory_loss_and_grad(objective,q,v)
    dq_full, dv_full = full_ret[1], full_ret[2]

    grad_q += ndarray(dq_full[-1])
    grad_v += ndarray(dv_full[-1])

    p_array = StdRealVector(p.clone().detach().numpy())
    dofs = deformable.dofs()

    dl_dq_next = np.copy(grad_q)
    dl_dv_next = np.copy(grad_v)
    act_dofs = deformable.act_dofs()
    dl_act = np.zeros((prepare + frame_num, act_dofs))
    dl_df_ext = np.zeros((prepare + frame_num, dofs))
    dl_dp = np.zeros(p.shape[0])
    mat_w_dofs = 2 * deformable.NumOfPdElementEnergies()
    act_w_dofs = deformable.NumOfPdMuscleEnergies()
    state_p_dofs = deformable.NumOfStateForceParameters()
    dl_dmat_w = np.zeros(mat_w_dofs)
    dl_dact_w = np.zeros(act_w_dofs)
    dl_dstate_p = np.zeros(state_p_dofs)

    for i in reversed(range(prepare + frame_num)):

        t = (i + 1) * dt

        vx_offset =  oscillation_offset(t)
        vx_vel = oscillation_vel(t)

        q_cur_array = ndarray(q[i].copy())
        v_cur_array = ndarray(v[i].copy())

        for node_idx in boundary_indices: 

            vx, vy, vz = sim_q0[node_idx : node_idx + 3]
            deformable.SetDirichletBoundaryCondition(node_idx, vx + vx_offset)
            deformable.SetDirichletBoundaryCondition(node_idx + 1, vy)
            deformable.SetDirichletBoundaryCondition(node_idx + 2, vz)

            q_cur_array[node_idx] = vx + vx_offset
            q_cur_array[node_idx + 1] = vy
            q_cur_array[node_idx + 2] = vz
            v_cur_array[node_idx] = vx_vel
            v_cur_array[node_idx + 1] = 0
            v_cur_array[node_idx + 2] = 0

        dl_dq = StdRealVector(dofs)
        dl_dv = StdRealVector(dofs)
        dl_da = StdRealVector(act_dofs)
        dl_df = StdRealVector(dofs)
        dl_dmat_wi = StdRealVector(mat_w_dofs)
        dl_dact_wi = StdRealVector(act_w_dofs)
        dl_dstate_pi = StdRealVector(state_p_dofs)
        dl_dpi = StdRealVector(p.shape[0])

        deformable.PyBackward(backward_method, q_cur_array, v_cur_array, sim_act[i], sim_f_ext[i], p_array, dt,
                q[i + 1], v[i + 1], active_contact_indices[i + 1], dl_dq_next, dl_dv_next,
                backward_opt, dl_dq, dl_dv, dl_da, dl_df, dl_dpi, dl_dmat_wi, dl_dact_wi, dl_dstate_pi)

        sur_grad = 1 / (beta * (1 + np.abs(beta * (p.detach().numpy() - 0.5)))**2)
        dl_dpi = sur_grad * ndarray(dl_dpi)
        
        dl_dq_next = ndarray(dl_dq)
        dl_dv_next = ndarray(dl_dv)

        if i >= prepare:
            dl_dq_next += ndarray(dq_full[i])
            dl_dv_next += ndarray(dv_full[i])

            ret = stepwise_loss_and_grad(objective, q[i], v[i], i, frame_num)
            dqi, dvi = ret[1], ret[2]
            dl_dq_next += ndarray(dqi)
            dl_dv_next += ndarray(dvi)
        
        dl_act[i - prepare] = ndarray(dl_da)
        dl_df_ext[i  - prepare] = ndarray(dl_df)
        dl_dp += ndarray(dl_dpi)
        dl_dmat_w += ndarray(dl_dmat_wi)
        dl_dact_w += ndarray(dl_dact_wi)
        dl_dstate_p += ndarray(dl_dstate_pi)
    
    return np.copy(dl_dp)


def optimize(objective, frame_num, hex_params, num_epochs = 100, lr = 1e3):

    env = HangingCantileverEnv3d(
        42,
        f"{objective}_optimize",
        hex_params,
    )
    deformable = env.deformable()

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.q0.clone() 
    v0 = env.v0.clone() 
    dt = 1e-2

    mesh = deformable.mesh()
    num_elements = mesh.NumOfElements()

    element_pos = []
    for e in range(num_elements):
        element = mesh.py_element(e)
        vertices = np.array([mesh.py_vertex(v) for v in element])
        element_pos.append(vertices.mean(axis=0))

    element_pos = np.array(element_pos)
    
    boundary_indices = []
    for i in range(0,env._dofs,3):
        if env.is_dirichlet_dof(i):
            boundary_indices.append(i)

    params = torch.normal(0.5, 0.3, [num_elements], dtype=torch.float64)

    sim_q0, sim_v0 = env._q0.copy(), env._v0.copy()

    Path(f"{objective}_optimize/checkpoints").mkdir(parents=True, exist_ok=True)
    np.save(f"{objective}_optimize/checkpoints/epoch_0.npy", params.clone().detach().numpy())

    plot_materialMode_views(f"{objective}_optimize/checkpoints/epoch_0.png", params.clone().detach().numpy(), element_pos)
    
    losses = []
    grads = []
    for epoch in tqdm(range(num_epochs)):

        loss, q, v, sim_act, sim_f_ext, active_contact_indices = sim_forward(objective, params, sim_q0, sim_v0, frame_num, deformable, boundary_indices, env.method, env.opt)
        dl_dp = sim_backward(objective, params, q, v, sim_act, sim_f_ext, active_contact_indices, sim_q0, sim_v0, frame_num, deformable, boundary_indices, env.method, env.opt)
        
        losses.append(loss)
        grads.append(dl_dp)

        np.save(f"{objective}_optimize/grads.npy", np.array(grads))
        np.save(f"{objective}_optimize/losses.npy", np.array(losses))
        plot_lossCurve(objective, np.array(losses))

        params -= lr * torch.tensor(dl_dp, dtype=params.dtype)
        params = params.clamp(0,1)
        np.save(f"{objective}_optimize/checkpoints/epoch_{epoch+1}.npy", params.clone().detach().numpy())
        plot_materialMode_views(f"{objective}_optimize/checkpoints/epoch_{epoch+1}.png", params.clone().detach().numpy(), element_pos)

        for node_idx in boundary_indices: 
            deformable.RemoveDirichletBoundaryCondition(node_idx)
            deformable.RemoveDirichletBoundaryCondition(node_idx + 1)
            deformable.RemoveDirichletBoundaryCondition(node_idx + 2)
        
    
    loss, q, v, sim_act, sim_f_ext, active_contact_indices = sim_forward(objective, params, sim_q0, sim_v0, frame_num, deformable, boundary_indices, env.method, env.opt)
    losses.append(loss)
    np.save(f"{objective}_optimize/losses.npy", np.array(losses))
    plot_lossCurve(objective, np.array(losses))


def visualize(objective, frame_num, hex_params, dt=0.01, amplitude=0.003, frequency=15, prepare = 200,  mode="best"):

    hex_params["amplitude"] = amplitude
    hex_params["frequency"] = frequency
    env = HangingCantileverEnv3d(
        42,
        f"{objective}_optimize",
        hex_params,
    )
    deformable = env.deformable()

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.q0.clone() 
    v0 = env.v0.clone() 
    
    mesh = deformable.mesh()
    num_elements = mesh.NumOfElements()

    element_pos = []
    for e in range(num_elements):
        element = mesh.py_element(e)
        vertices = np.array([mesh.py_vertex(v) for v in element])
        element_pos.append(vertices.mean(axis=0))

    element_pos = np.array(element_pos)

    if mode == "best":
        losses = np.load(f'{objective}_optimize/losses.npy')
        min_epoch = np.argmin(losses)   
        params_arr = np.load(f'{objective}_optimize/checkpoints/epoch_{min_epoch}.npy')
    else:
        params_arr = np.load(f'{objective}_optimize/checkpoints/epoch_0.npy')

    plot_materialMode_views(f"{objective}_optimize/{mode}.png", params_arr, element_pos)

    params = torch.tensor(params_arr, dtype=torch.float64)

    q, v = q0.clone(), v0.clone()
    for k in range(int(prepare)):
        t = (k + 1) * dt
        vx_offset, vx_vel = env.update_boundary(t)

        for node_dof in range(0, dofs, 3):
            if env.is_dirichlet_dof(node_dof):
                q[node_dof] = q0[node_dof] + vx_offset
                q[node_dof + 1] = q0[node_dof + 1]
                q[node_dof + 2] = q0[node_dof + 2]
                v[node_dof] = vx_vel
                v[node_dof + 1] = 0
                v[node_dof + 2] = 0
        q, v =  env.forward(q, v, material_mode=params, dt=dt)

    Path(f"{objective}_optimize/visualizations/{mode}").mkdir(parents=True, exist_ok=True)

    env.display_mesh(q, f"{objective}_optimize/visualizations/{mode}/0.png")
    for k in range(int(frame_num)):
        t = (k + 1 + prepare) * dt
        vx_offset, vx_vel = env.update_boundary(t)

        for node_dof in range(0, dofs, 3):
            if env.is_dirichlet_dof(node_dof):
                q[node_dof] = q0[node_dof] + vx_offset
                q[node_dof + 1] = q0[node_dof + 1]
                q[node_dof + 2] = q0[node_dof + 2]
                v[node_dof] = vx_vel
                v[node_dof + 1] = 0
                v[node_dof + 2] = 0
        q, v =  env.forward(q, v, material_mode=params, dt=dt)
        file_name = f"{objective}_optimize/visualizations/{mode}/{k+1}.png"
        env.display_mesh(q, file_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-objective", dest="objective", type=str, default="negative")
    args = parser.parse_args()

    frame_num = 150
    hex_params = {
                "amplitude": 0.003,
                "refinement": 1,
                'density':  6e3,
                'state_force_parameters': [0, 0, -9.80709],
            }
    
    if "maxAmplitude" in args.objective:
        lr = 1e4
    else:
        lr = 1e3 
    optimize(args.objective, frame_num, hex_params, num_epochs = 100, lr=lr)

    Path(f"{args.objective}_optimize/visualizations").mkdir(parents=True, exist_ok=True)

    visualize(args.objective, frame_num, hex_params, mode="best")
    visualize(args.objective, frame_num, hex_params, mode="init")

    generate_video(f"{args.objective}_optimize/checkpoints", "", f"{args.objective}_optimize/epochs.mp4", fps=5)
    generate_video_directory(f"{args.objective}_optimize/visualizations", ["best", "init"], flag="", delete_after=True) 