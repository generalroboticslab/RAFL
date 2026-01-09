import sys

sys.path.append("../..")

from pathlib import Path
import numpy as np
import torch
from env_hanging_cantilever import HangingCantileverEnv3d
from video_generation import *
from tqdm import tqdm

def data_generate(sample_num, frame_num, hex_params, vis=True, mode="random"):

    oscillation_amplitudes = [0.003] if sample_num == 1 else np.linspace(0.0005, 0.003, sample_num)
    for idx, amplitude in tqdm(enumerate(oscillation_amplitudes)):

        hex_params["amplitude"] = amplitude
        env = HangingCantileverEnv3d(
            42,
            f"oscillate_{mode}",
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
        params = torch.normal(0.5, 0.3, [num_elements], dtype=torch.float64)

        if mode == "soft":
            params_arr = np.zeros(num_elements)
        elif mode == "stiff":
            params_arr = np.ones(num_elements)
        elif mode == "topStiff":
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
        elif mode == "topSoft":
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
        
        params = torch.tensor(params_arr, dtype=torch.float64)

        prepare = 200 #1500

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

        Path(f"oscillate_{mode}/{idx}").mkdir(parents=True, exist_ok=True)

        qs = [q.clone().detach().numpy()]
        vs = [v.clone().detach().numpy()]
        if vis:
            env.display_mesh(q, f"oscillate/{idx}/0.png")
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
            if vis:
                file_name = f"oscillate/{idx}/{k+1}.png"
                env.display_mesh(q, file_name)
            qs.append(q.clone().detach().numpy())
            vs.append(v.clone().detach().numpy())

        data = {"q": np.stack(qs), "v": np.stack(vs), "amplitude": amplitude}
        data_folder = f'data_real_sim2sim'
        Path(f"{data_folder}").mkdir(parents=True, exist_ok=True)
        np.save(f"{data_folder}/trajectory{idx}.npy", data)
    
    generate_video_directory(f"oscillate", list(range(sample_num)), flag="")


if __name__ == "__main__":
    youngs_modulus = 263824
    sample_num = 1 #8 # 200
    frame_num = 150
    hex_params = {
                "refinement": 1,
                "twist_angle": 0.0,
                'density':  6e3,
                'state_force_parameters': [0, 0, -9.80709],
            }
    data_generate(sample_num, frame_num, hex_params, mode="stiff")
