import sys

sys.path.append("../..")

from pathlib import Path
import numpy as np
import torch
from env_hanging_cantilever_scaled import HangingCantileverScaledEnv3d
from video_generation import *
from tqdm import tqdm

def data_generate(sample_num, frame_num, hex_params, vis=True, target_dx=0.01):

    for idx in tqdm(range(sample_num)):

        orig_data = np.load(f'../sim2sim_hanging_beam_model/data_real_solid/trajectory{idx}.npy', allow_pickle=True).item()
        amplitude = orig_data['amplitude']

        hex_params["amplitude"] = amplitude
        env = HangingCantileverScaledEnv3d(
            42,
            f"oscillate_liquid_{target_dx}",
            hex_params,
            target_dx=target_dx
        )
        deformable = env.deformable()

        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = env.q0.clone() 
        v0 = env.v0.clone() 
        dt = 1e-2

        Path(f"oscillate_liquid_{target_dx}/{idx}").mkdir(parents=True, exist_ok=True)

        qs = [q0.clone().detach().numpy()]
        vs = [v0.clone().detach().numpy()]
        q, v = q0.clone(), v0.clone()
        if vis:
            env.display_mesh(q, f"oscillate_liquid_{target_dx}/{idx}/0.png")
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
                file_name = f"oscillate_liquid_{target_dx}/{idx}/{k+1}.png"
                env.display_mesh(q, file_name)
            qs.append(q.clone().detach().numpy())
            vs.append(v.clone().detach().numpy())

        data = {"q": np.stack(qs), "v": np.stack(vs), "amplitude": amplitude}
        data_folder = f'data_real_liquid_{target_dx}'
        Path(f"{data_folder}").mkdir(parents=True, exist_ok=True)
        np.save(f"{data_folder}/trajectory{idx}.npy", data)
    
    generate_video_directory(f"oscillate_liquid_{target_dx}", list(range(sample_num)), flag="")


if __name__ == "__main__":
    # youngs_modulus = 215856
    # poissons_ratio = 0.45
    youngs_modulus = 500000 #9000000 
    poissons_ratio = 0.42 
    sample_num = 20 # 200
    frame_num = 100
    hex_params = {
                "refinement": 1,
                "youngs_modulus": youngs_modulus,
                "twist_angle": 0.0,
                'density': 6e3,
                'poissons_ratio': poissons_ratio,
            }
    data_generate(sample_num, frame_num, hex_params, target_dx=0.01)
