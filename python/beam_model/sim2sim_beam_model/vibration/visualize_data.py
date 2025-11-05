import sys
sys.path.append('../')
sys.path.append('../../..')
from pathlib import Path
import numpy as np
import torch

from video_generation import *
from env_cantilever import CantileverEnv3d
from argparse import ArgumentParser
from tqdm import tqdm

args = ArgumentParser()
args.add_argument("-t", dest="save_folder", required=False)


if __name__ == '__main__':
    youngs_modulus = 215856
    poissons_ratio = 0.45
    density = 1.07e3
    state_force = [0, 0, -9.80709]
    hex_params = {
        'density': density,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'state_force_parameters': state_force,
        'mesh_type': 'hex',
        'refinement': 1,
    }
    
    save_folder = 'data_real' if args.parse_args().save_folder is None else args.parse_args().save_folder
    cantilever = CantileverEnv3d(42, save_folder, hex_params)

    cantilever_base = CantileverEnv3d(42, "cantilever_data_sim2sim", hex_params)
    
    dt = 0.01
    for idx in tqdm(range(18)):
        
        data = np.load(f'{save_folder}/trajectory{idx}.npy', allow_pickle=True).item()

        q_trajectory = data['q']

        Path(f"{save_folder}/visualize/{idx}").mkdir(parents=True, exist_ok=True)
        for i in range(q_trajectory.shape[0]):

            q = q_trajectory[i]

            cantilever.display_mesh(q, f"{save_folder}/visualize/{idx}/{i}.png")
        
        target_trajectory = torch.from_numpy(data['q'])
        target_trajectory_v = torch.from_numpy(data['v'])
        q0 = target_trajectory[0]
        v0 = target_trajectory_v[0]
        q, v = q0.clone(), v0.clone()
        Path(f"cantilever_data_sim2sim/visualize/{idx}").mkdir(parents=True, exist_ok=True)
        cantilever_base.display_mesh(q, f"cantilever_data_sim2sim/visualize/{idx}/0.png")
        for k in range(int(q_trajectory.shape[0]) - 1):
            
            q, v =  cantilever_base.forward(q, v, dt=dt)
            cantilever_base.display_mesh(q, f"cantilever_data_sim2sim/visualize/{idx}/{k+1}.png")

    generate_video_directory(f"{save_folder}/visualize/", list(range(18)), flag="")
    generate_video_directory(f"cantilever_data_sim2sim/visualize/", list(range(18)), flag="")







    