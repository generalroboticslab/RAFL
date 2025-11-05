import sys

sys.path.append("../..")

from pathlib import Path
import numpy as np
import torch
from env_hanging_cantilever import HangingCantileverEnv3d
from video_generation import *
from tqdm import tqdm

def data_generate(sample_num, frame_num, hex_params, vis=True):

    scale = 0.01 / 0.0035

    oscillation_amplitudes = scale * np.random.uniform(0.0005, 0.003, size=sample_num)
    for idx, amplitude in tqdm(enumerate(oscillation_amplitudes)):

        hex_params["amplitude"] = amplitude
        env = HangingCantileverEnv3d(
            42,
            f"oscillate",
            hex_params,
        )
        deformable = env.deformable()

        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = env.q0.clone() 
        v0 = env.v0.clone() 
        dt = 1e-2
        
        prepare = 100 #1500

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
            q, v =  env.forward(q, v, dt=dt)

        Path(f"oscillate/{idx}").mkdir(parents=True, exist_ok=True)

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
            q, v =  env.forward(q, v, dt=dt)
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
    sample_num = 20 # 200
    frame_num = 100
    hex_params = {
                "refinement": 1,
                "youngs_modulus": youngs_modulus,
                "twist_angle": 0.0,
                'density': 1.07e3,
                'poissons_ratio': 0.499,
                'state_force_parameters': [0, 0, -9.80709],
            }
    data_generate(sample_num, frame_num, hex_params)
