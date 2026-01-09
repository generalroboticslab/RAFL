import sys
sys.path.append('../')
sys.path.append('../../..')
from pathlib import Path
import numpy as np
import torch

from env_fishTailMesh import FishTailMeshEnv3d
from argparse import ArgumentParser
from tqdm import tqdm 
from video_generation import generate_video_directory

args = ArgumentParser()
args.add_argument("-t", dest="save_folder", required=False)


if __name__ == '__main__':
    youngs_modulus = 263824
    poissons_ratio = 0.499
    density = 1.07e3
    state_force = [0, 0, -9.80709]
    hex_params = {
        'density': density,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'state_force_parameters': state_force,
        'mesh_type': 'tet',
        'refinement': 1,
    }
    weights = [0.05, 0.06, 0.07, 0.1, 0.09, 0.08, 0.11, 0.12, 0.15, 0.09, 0.13, 0.14, 0.16, 0.17, 0.2,0.18,0.22,0.21]
    prepare = 200
    save_folder = 'fishTailMesh_data_real' if args.parse_args().save_folder is None else args.parse_args().save_folder
    cantilever = FishTailMeshEnv3d(42, 'fishTailMesh', hex_params)
    q = torch.from_numpy(cantilever._q0)
    v = torch.zeros_like(q)

    Path(f"{save_folder}/visualizations").mkdir(parents=True, exist_ok=True)
    for id in range(len(weights)):
        Path(f"{save_folder}/visualizations/{id}").mkdir(parents=True, exist_ok=True)
        for j in tqdm(range(prepare)):
                weight = weights[id] * 9.80709
                res_force = torch.zeros(q.shape, dtype=torch.float64)

                res_force = res_force.reshape(-1,3)
                res_force[cantilever.force_nodes, 2] = - weight / len(cantilever.force_nodes)
                res_force = res_force.flatten()

                q, v = cantilever.forward(q, v, f_ext=res_force, dt=0.01)
        q_trajectory = [q.detach().numpy()]
        v_trajectory = [v.detach().numpy()]
        cantilever.display_mesh(q.detach(), f"{save_folder}/visualizations/{id}/0.png")
        for k in tqdm(range(150)):
            q, v = cantilever.forward(q, v, f_ext=torch.zeros_like(q), dt=0.01)
            q_trajectory.append(q.detach().numpy())
            v_trajectory.append(v.detach().numpy())
            cantilever.display_mesh(q.detach(), f"{save_folder}/visualizations/{id}/{k+1}.png")
        data = {"q" : np.stack(q_trajectory), "v" : np.stack(v_trajectory)}
        np.save(f'{save_folder}/trajectory{id}.npy', data)
        print("Finish trajectory", id)

    generate_video_directory(f"{save_folder}/visualizations", list(range(len(weights))), flag="")




    