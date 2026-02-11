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


x_mean = 0.00525
default_location = np.array([0.00525, 0.00525, 0.])
min_z_idx = [  0,  31,  62,  93, 124, 155, 186, 217, 248, 279, 310, 341, 372, 403, 434, 465]

def get_max_amplitude(qs, frequency, dt, prepare, frame_num, boundary_indices):


    qs_arr = np.stack(qs)

    to_idx = lambda t: np.rint(t / dt).astype(int)

    Tperiod = 1.0 / frequency
    t_end = frame_num * dt

    t_shift = (prepare * dt) #/ (2.0 * np.pi * frequency)

    def make_indices(offset_frac):
        # t = t_shift + (offset_frac + k) * Tperiod
        k_min = int(np.floor((0.0 + t_shift) / Tperiod - offset_frac))
        k_max = int(np.ceil((t_end + t_shift) / Tperiod - offset_frac))
        ks = np.arange(k_min, k_max + 1, dtype=float)
        ts = -t_shift + (offset_frac + ks) * Tperiod
        ts = ts[(ts >= 0.0) & (ts <= t_end)]
        idx = to_idx(ts)
        idx = idx[(idx >= 0) & (idx < frame_num)]
        # dedup in case rounding maps two times to same index
        return np.unique(idx)

    pos_idx = make_indices(0.25)
    neg_idx = make_indices(0.75)

    pos_boundary_qs = qs_arr[pos_idx][:,boundary_indices]
    pos_bottom_qs = qs_arr.reshape(frame_num + 1, -1, 3)[pos_idx][:,min_z_idx,:]

    neg_boundary_qs = qs_arr[neg_idx][:,boundary_indices]
    neg_bottom_qs = qs_arr.reshape(frame_num + 1, -1, 3)[neg_idx][:,min_z_idx,:]

    print(pos_boundary_qs.mean(axis=-1))
    print(neg_boundary_qs.mean(axis=-1))

    all_peak_bottom_qs = np.concatenate([pos_bottom_qs, neg_bottom_qs], axis=0)

    max_amplitude = np.linalg.norm(all_peak_bottom_qs.mean(axis=1) - default_location, axis=-1).mean()

    print(max_amplitude)

    return max_amplitude


def run_sim(hex_params,  mode, frequency, amplitude=0.0005, prepare = 1000, frame_num = 1000, dt=0.001):
    hex_params["amplitude"] = amplitude
    hex_params["frequency"] = frequency

    env = HangingCantileverEnv3d(
        42,
        f"freq_response_{mode}",
        hex_params,
    )
    deformable = env.deformable()

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.q0.detach().clone() 
    v0 = env.v0.detach().clone() 

    # default_location = env._q0.reshape(-1,3)[min_z_idx,:].mean(axis=0)
    # print(default_location)
    # exit()

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
    
    if mode == "stiff":
        params_arr = np.zeros(num_elements)
    elif mode == "soft":
        params_arr = np.ones(num_elements)
    elif mode == "topSoft":
        vert_num = mesh.NumOfVertices()
        verts = np.array([np.array(mesh.py_vertex(i)) for i in range(vert_num)])
        mid_z = verts[:,2].mean()
        material_mode = []
        for e in range(num_elements):
            element = mesh.py_element(e)
            vertices = np.array([mesh.py_vertex(v) for v in element])
            if vertices[:,2].mean() > mid_z:
                material_mode.append(1.0)
            else:
                material_mode.append(0.0)
        params_arr = np.array(material_mode)
    elif mode == "topStiff":
        vert_num = mesh.NumOfVertices()
        verts = np.array([np.array(mesh.py_vertex(i)) for i in range(vert_num)])
        mid_z = verts[:,2].mean()
        material_mode = []
        for e in range(num_elements):
            element = mesh.py_element(e)
            vertices = np.array([mesh.py_vertex(v) for v in element])
            if vertices[:,2].mean() > mid_z:
                material_mode.append(0.0)
            else:
                material_mode.append(1.0)
        params_arr = np.array(material_mode)

    params = torch.tensor(params_arr)
        
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
        t = (prepare + k + 1) * dt
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
        
        qs.append(q.clone().detach().numpy())
    
    max_amplitude = get_max_amplitude(qs, frequency, dt, prepare, frame_num, boundary_indices)

    return max_amplitude



if __name__ == "__main__":


    hex_params = {
                "refinement": 1,
                'density':  6e3,
                'state_force_parameters': [0, 0, -9.80709],
            }


    mode = "stiff"
    frequencies = list(range(1,41))
    max_amplitudes = []
    for f in tqdm(frequencies[::-1]):

        max_amplitude = run_sim(hex_params, mode=mode, frequency=f)

        max_amplitudes.append(max_amplitude)
    

    plt.plot(frequencies, max_amplitudes)
    plt.savefig(f"{mode}.png")


