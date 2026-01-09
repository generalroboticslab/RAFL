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

    plt.savefig(f"optimize_{objective}/losses.png")
    plt.close()

def visualize(objective, hex_params, dt=0.01, amplitude=0.003, frequency=15, frame_num = 150, prepare = 200,  mode="best"):

    hex_params["amplitude"] = amplitude
    hex_params["frequency"] = frequency
    env = HangingCantileverEnv3d(
        42,
        f"optimize_{objective}",
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
        losses = np.load(f'optimize_{objective}/losses.npy')
        min_epoch = np.argmin(losses)   
        params_arr = np.load(f'optimize_{objective}/checkpoints/epoch_{min_epoch}.npy')

        epoch_used = min_epoch

    else:
        params_arr = np.load(f'optimize_{objective}/checkpoints/epoch_0.npy')

        epoch_used = 0

    plot_materialMode_views(f"optimize_{objective}/{mode}.png", params_arr, element_pos)
    np.save(f"optimize_{objective}/{mode}.npy", params_arr)

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
    
    qs = [q.clone()]
    Path(f"optimize_{objective}/visualizations/{mode}").mkdir(parents=True, exist_ok=True)

    env.display_mesh(q, f"optimize_{objective}/visualizations/{mode}/0.png")
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
        file_name = f"optimize_{objective}/visualizations/{mode}/{k+1}.png"
        env.display_mesh(q, file_name)
        qs.append(q.clone())

    q_traj = torch.stack(qs)

    np.save(f"optimize_{objective}/visualizations/{mode}_traj.npy", q_traj.detach().numpy())

    # xs = q_traj.reshape(frame_num + 1, -1, 3)[:,min_z_idx,0].mean(-1)
    # print(f"{mode} model max amplitude: {float((xs.max() - xs.min()).detach().numpy()):.17g}")
    return epoch_used


x_mean = 0.00525
min_z_idx = [  0,  31,  62,  93, 124, 155, 186, 217, 248, 279, 310, 341, 372, 403, 434, 465]

def optimize(objective, hex_params, num_epochs = 100, lr = 1e3, dt=0.01, amplitude=0.003, frequency=15, frame_num = 150, prepare = 200 ):

    hex_params["amplitude"] = amplitude
    hex_params["frequency"] = frequency

    env = HangingCantileverEnv3d(
        42,
        f"optimize_{objective}",
        hex_params,
    )
    deformable = env.deformable()

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.q0.detach().clone() 
    v0 = env.v0.detach().clone() 

    mesh = deformable.mesh()
    num_elements = mesh.NumOfElements()

    element_pos = []
    for e in range(num_elements):
        element = mesh.py_element(e)
        vertices = np.array([mesh.py_vertex(v) for v in element])
        element_pos.append(vertices.mean(axis=0))

    element_pos = np.array(element_pos)
    
    boundary_indices = []
    non_boundary_indices = []
    for i in range(0,env._dofs,3):
        if env.is_dirichlet_dof(i):
            boundary_indices.append(i)
        else:
            non_boundary_indices.extend([i, i+1, i+2])

    params = torch.normal(0.5, 0.3, [num_elements], dtype=torch.float64, requires_grad=True)

    losses = []
    grads = []

    optimizer = torch.optim.Adam([params],lr=lr)

    Path(f"optimize_{objective}/checkpoints").mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(num_epochs + 1)):

        plot_materialMode_views(f"optimize_{objective}/checkpoints/epoch_{epoch}.png", params.clone().detach().numpy(), element_pos)
        np.save(f"optimize_{objective}/checkpoints/epoch_{epoch}.npy", params.clone().detach().numpy())

        optimizer.zero_grad()

        q, v = env.q0.clone(), env.v0.clone()

        for k in range(int(prepare)):
            t = (k + 1) * dt
            vx_offset, vx_vel = env.get_new_boundary(t)

            q_cur, v_cur = torch.zeros_like(q), torch.zeros_like(v)
            for node_dof in boundary_indices:
                q_cur[node_dof] = q0[node_dof] + vx_offset
                q_cur[node_dof + 1] = q0[node_dof + 1]
                q_cur[node_dof + 2] = q0[node_dof + 2]
                v_cur[node_dof] = vx_vel
                v_cur[node_dof + 1] = 0
                v_cur[node_dof + 2] = 0

            q_cur[non_boundary_indices] = q[non_boundary_indices]
            v_cur[non_boundary_indices] = v[non_boundary_indices]
            
            q, v  =  env.forward(q_cur, v_cur, varying_boundary_indices=boundary_indices, material_mode=params, dt=dt)

        qs = [q.clone()]

        for k in range(int(frame_num)):
            t = (k + 1 + prepare) * dt
            vx_offset, vx_vel = env.get_new_boundary(t)

            q_cur, v_cur = torch.zeros_like(q), torch.zeros_like(v)
            for node_dof in boundary_indices:
                q_cur[node_dof] = q0[node_dof] + vx_offset
                q_cur[node_dof + 1] = q0[node_dof + 1]
                q_cur[node_dof + 2] = q0[node_dof + 2]
                v_cur[node_dof] = vx_vel
                v_cur[node_dof + 1] = 0
                v_cur[node_dof + 2] = 0

            q_cur[non_boundary_indices] = q[non_boundary_indices]
            v_cur[non_boundary_indices] = v[non_boundary_indices]

            q, v =  env.forward(q_cur, v_cur, varying_boundary_indices=boundary_indices, material_mode=params, dt=dt)
            
            qs.append(q.clone())
        
        q_traj = torch.stack(qs)
        
        xs = q_traj.reshape(frame_num + 1, -1, 3)[:,min_z_idx,0].mean(-1)

        if objective == "negative":
            # loss = - torch.relu(x_mean - q_traj.reshape(frame_num + 1, -1, 3)[:,:,0]).mean(-1).mean()
            loss = xs.mean() - x_mean

        elif objective == "positive":
            # loss = - torch.relu(q_traj.reshape(frame_num + 1, -1, 3)[:,:,0] - x_mean).mean(-1).mean()
            loss = x_mean - xs.mean()

        elif objective == "minAmplitude":
            # xs = q_traj.reshape(frame_num + 1, -1, 3)[:,min_z_idx,0].mean(-1)
            loss = (xs - xs.mean()).abs().mean()

        elif objective == "maxAmplitude":
            # xs = q_traj.reshape(frame_num + 1, -1, 3)[:,min_z_idx,0].mean(-1)
            loss = -(xs - xs.mean()).abs().mean()

        if epoch < num_epochs:
            loss.backward()

            grads.append(params.grad.clone().detach().cpu().numpy())
            np.save(f"optimize_{objective}/grads.npy", np.array(grads))

            optimizer.step()

            with torch.no_grad():
                params.clamp_(0, 1)

        losses.append(loss.clone().detach().cpu().numpy())
        plot_lossCurve(objective, np.array(losses))
        np.save(f"optimize_{objective}/losses.npy", np.array(losses))
        
        env.remove_boundary()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-objective", dest="objective", type=str, choices=["positive", "negative", "maxAmplitude", "minAmplitude"], default="negative")
    args = parser.parse_args()

    hex_params = {
                "refinement": 1,
                'state_force_parameters': [0, 0, -9.80709],
            }
    
    num_epochs = 100
    lr = 0.1

    optimize(args.objective, hex_params, num_epochs = num_epochs, lr=lr)

    Path(f"optimize_{args.objective}/visualizations").mkdir(parents=True, exist_ok=True)

    init_epoch = visualize(args.objective, hex_params, mode="init")
    best_epoch = visualize(args.objective, hex_params, mode="best")

    generate_video(f"optimize_{args.objective}/checkpoints", "", f"optimize_{args.objective}/epochs.mp4", fps=max((best_epoch + 1) // 10,1), num_frames=best_epoch + 1)
    generate_video_directory(f"optimize_{args.objective}/visualizations", ["best", "init"], flag="", delete_after=True) 
