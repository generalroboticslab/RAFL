import sys

sys.path.append("../..")

from pathlib import Path
import numpy as np
import torch
from env_underwater_cantilever import UnderwaterCantileverEnv3d
from video_generation import *
from tqdm import tqdm

def data_generate(sample_num, frame_num, hex_params, vis=True):

    oscillation_amplitudes = np.random.uniform(0.0005, 0.002, size=sample_num)
    for idx, amplitude in tqdm(enumerate(oscillation_amplitudes)):

        hex_params["amplitude"] = amplitude
        env = UnderwaterCantileverEnv3d(
            42,
            "oscillate_liquid",
            hex_params,
        )
        deformable = env.deformable()

        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = env.q0.clone() 
        v0 = env.v0.clone() 
        dt = 1e-2

        Path(f"{folder}/{idx}").mkdir(parents=True, exist_ok=True)

        qs = [q0.clone().detach().numpy()]
        vs = [v0.clone().detach().numpy()]
        q, v = q0.clone(), v0.clone()
        if vis:
            env.display_mesh(q, f"{folder}/{idx}/0.png")
        for k in range(int(frame_num)):
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
            if vis:
                file_name = f"{folder}/{idx}/{k+1}.png"
                env.display_mesh(q, file_name)
            qs.append(q.clone().detach().numpy())
            vs.append(v.clone().detach().numpy())

        data = {"q": np.stack(qs), "v": np.stack(vs), "amplitude": amplitude}
        data_folder = 'data_real_liquid'
        Path(f"{data_folder}").mkdir(parents=True, exist_ok=True)
        np.save(f"{data_folder}/trajectory{idx}.npy", data)
    
    generate_video_directory(folder, list(range(sample_num)), flag="")


if __name__ == "__main__":
    # youngs_modulus = 215856
    # poissons_ratio = 0.45
    youngs_modulus = 263824
    poissons_ratio = 0.499
    sample_num = 20 # 200
    frame_num = 100
    hex_params = {
                "refinement": 1,
                "youngs_modulus": youngs_modulus,
                "twist_angle": 0.0,
                'density': 1.07e3,
                'poissons_ratio': poissons_ratio,
            }
    data_generate(sample_num, frame_num, hex_params)
