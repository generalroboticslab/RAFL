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
min_z_idx = [  0,  31,  62,  93, 124, 155, 186, 217, 248, 279, 310, 341, 372, 403, 434, 465]

def get_max_amplitude(qs, frequency, dt):


    peaks = (np.pi / 2) * self.frequency
    return 0


def run_sim(hex_params,  mode="soft", amplitude=0.0005, frequency=15, frame_num = 300, prepare = 200, dt=0.01):
    hex_params["amplitude"] = amplitude
    hex_params["frequency"] = frequency

    env = HangingCantileverEnv3d(
        42,
        f"freq_{frequency}",
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
                material_mode.append(0.)
            else:
                material_mode.append(1.)
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
                material_mode.append(1.)
            else:
                material_mode.append(0.)
        params_arr = np.array(material_mode)

    params = torch.tensor(params_arr)
        
    q, v = env.q0.clone(), env.v0.clone()

    # for k in range(int(prepare)):
    #     t = (k + 1) * dt
    #     vx_offset, vx_vel = env.get_new_boundary(t)

    #     q_cur, v_cur = torch.zeros_like(q), torch.zeros_like(v)
    #     for node_dof in boundary_indices:
    #         q_cur[node_dof] = q0[node_dof] + vx_offset
    #         q_cur[node_dof + 1] = q0[node_dof + 1]
    #         q_cur[node_dof + 2] = q0[node_dof + 2]
    #         v_cur[node_dof] = vx_vel
    #         v_cur[node_dof + 1] = 0
    #         v_cur[node_dof + 2] = 0

    #     q_cur[non_boundary_indices] = q[non_boundary_indices]
    #     v_cur[non_boundary_indices] = v[non_boundary_indices]
        
    #     q, v  =  env.forward(q_cur, v_cur, varying_boundary_indices=boundary_indices, material_mode=params, dt=dt)

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
    
    max_amplitude = get_max_amplitude(qs, frequency, dt)

    return max_amplitude



if __name__ == "__main__":


    hex_params = {
                "refinement": 1,
                'state_force_parameters': [0, 0, -9.80709],
            }


    max_amplitudes = []
    for f in tqdm(range(41)):

        max_amplitude = run_sim(hex_params, frequency=f)

        max_amplitudes.append(max_amplitude)
    



