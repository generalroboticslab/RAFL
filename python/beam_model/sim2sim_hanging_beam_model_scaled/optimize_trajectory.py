import sys
sys.path.append('../..')
from video_generation import *
from pathlib import Path
import time
import numpy as np
import torch

from env_hanging_cantilever_scaled import HangingCantileverScaledEnv3d

def optimize_trajectoryfull( hex_params, num_frames, num_epochs, dt, sample=0, suffix = "solid", target_dx=0.01):

    target_data = np.load(f"data_real_{suffix}_{target_dx}/trajectory{sample}.npy", allow_pickle=True)[()]
    hex_params["amplitude"] = target_data["amplitude"]
    
    cantilever = HangingCantileverScaledEnv3d(42, f'oscillate_base_{target_dx}', hex_params, target_dx=target_dx)
    
    q_init = cantilever.q0.clone() 
    
    target_trajectory = torch.from_numpy(target_data['q'])
    target_trajectory_v = torch.from_numpy(target_data['v'])
    q0 = target_trajectory[0]
    v0 = target_trajectory_v[0]

    prepare = 100

    if suffix == 'solid':
        Path(f"oscillate_base_{target_dx}/{sample}").mkdir(parents=True, exist_ok=True)

        q, v = q0.clone(), v0.clone()
        cantilever.display_mesh(q, f"oscillate_base_{target_dx}/{sample}/0.png")
        for k in range(int(num_frames)):
            t = (k + 1 + prepare) * dt
            vx_offset, vx_vel = cantilever.update_boundary(t)

            for node_dof in range(0, cantilever._dofs, 3):
                if cantilever.is_dirichlet_dof(node_dof):
                    q[node_dof] = q_init[node_dof] + vx_offset
                    q[node_dof + 1] = q_init[node_dof + 1]
                    q[node_dof + 2] = q_init[node_dof + 2]
                    v[node_dof] = vx_vel
                    v[node_dof + 1] = 0
                    v[node_dof + 2] = 0
            q, v =  cantilever.forward(q, v, dt=dt)
            file_name = f"oscillate_base_{target_dx}/{sample}/{k+1}.png"
            cantilever.display_mesh(q, file_name)

    force_nodes_num = cantilever._dofs
    params = torch.normal(0, 1e-4, [force_nodes_num, num_frames], dtype=torch.float64, requires_grad=True)

    ### Define target location
    q_all = []
    v_all = []
    pressure_forces  = []
    loss_history_init = []
    loss_history_optimized = []
    # q_all.append(q0)
    # v_all.append(np.zeros_like(q0))

    q_last_epoch = q0.clone()
    v_last_epoch = v0.clone()

    for frame_i in range(1, num_frames):
        start_time = time.time()
        t = (frame_i + prepare)  * dt
        print("t: ", t)
        with torch.no_grad():
            params[:, frame_i] = params[:, frame_i - 1]
            vx_offset, vx_vel = cantilever.update_boundary(t)
            print("amplitude: ", hex_params["amplitude"])
            print("offset: ", vx_offset)
            for node_dof in range(0, cantilever._dofs, 3):
                if cantilever.is_dirichlet_dof(node_dof):
                    q_last_epoch[node_dof] = q_init[node_dof] + vx_offset
                    q_last_epoch[node_dof + 1] = q_init[node_dof + 1]
                    q_last_epoch[node_dof + 2] = q_init[node_dof + 2]
                    v_last_epoch[node_dof] = vx_vel
                    v_last_epoch[node_dof + 1] = 0
                    v_last_epoch[node_dof + 2] = 0

        init_loss = ((target_trajectory[frame_i] - q_last_epoch)**2).sum().item()
        loss_history_init.append(init_loss)
        optimizer = torch.optim.Adam([params],lr=1e-3)
        loss_last = 0
        f = np.array(cantilever.sim.deformable.PyForwardStateForce(q_last_epoch.detach().clone().numpy(), v_last_epoch.detach().clone().numpy()))

        v_all.append(v_last_epoch.detach().numpy())
        q_all.append(q_last_epoch.detach().numpy())
        pressure_forces.append(f)

        q_sim, v_sim = cantilever.forward(q_last_epoch, v_last_epoch, f_ext=torch.zeros_like(q_last_epoch), dt=dt)
        params_prev_step = params[:, frame_i - 1].detach().clone()
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            q, v = q_last_epoch.detach().clone(), v_last_epoch.detach().clone()
            f_ext = torch.zeros(cantilever._dofs, dtype=torch.float64)
            f_ext[-force_nodes_num:] += params[:, frame_i]

            q, v = cantilever.forward(q, v, f_ext=f_ext, dt=dt)
            data_loss = ((target_trajectory[frame_i] - q)**2).sum()
            loss = data_loss + 1e-4 * (params[:,frame_i]**2).sum()
            if (torch.abs(loss - loss_last)/loss < 1e-6):
                break
            loss_last = loss
            # Backward gradients so we know which direction to update parameters
            loss.backward()
            # Actually update parameters
            optimizer.step()
        loss_history_optimized.append(loss.item())
        v_last_epoch = v
        q_last_epoch = q

        print("Params Prev Step Udated: ", torch.norm(params_prev_step - params[:, frame_i - 1]).item())
        print(f"Error sim and real: {((q.detach().numpy() - q_sim.detach().numpy())**2).sum()}")
    
        with np.printoptions(precision=3):
            print(f"Frame [{frame_i}/{num_frames-1}]/ Epoch {epoch}: {(time.time()-start_time):.2f}s,- Loss {data_loss.item():.4e}, init_loss {init_loss} ")

    v_all.append(v_last_epoch.detach().numpy())
    q_all.append(q_last_epoch.detach().numpy())

    ### Plotting Loss History
    print("length of q_all", len(q_all))
    print("length of v_all", len(v_all))
    optimized_data_save = {
        "q_trajectory": np.stack(q_all),
        "v_trajectory": np.stack(v_all),
        "pressure_forces": np.stack(pressure_forces),
        "optimized_forces": params.detach().numpy(),
        "amplitude": hex_params["amplitude"]
    }
    Path(f"cantilever_data_sim2sim_{suffix}_{target_dx}").mkdir(parents=True,exist_ok=True)
    np.save(f"cantilever_data_sim2sim_{suffix}_{target_dx}/optimized_data_{sample}.npy", optimized_data_save)

if __name__ == '__main__':
    prepare = 100
    forward = 0
    forward_times = []
    transformed_data = []
    idxs = list(range(20))
    for idx in idxs:

        youngs_modulus = 1500000 #97350 
        poissons_ratio = 0.4
        density = 6e3 
        hex_params = {
            'density': density,
            'youngs_modulus': youngs_modulus,
            'poissons_ratio': poissons_ratio,
            'mesh_type': 'hex',
            'refinement': 1,
        }

        ########################################
        num_frames = 100
        num_epochs = 300
        #optimize_trajectoryfull(hex_params, num_frames, num_epochs, 1e-2, sample=idx, suffix="liquid")
        optimize_trajectoryfull(hex_params, num_frames, num_epochs, 1e-2, sample=idx, suffix="solid", target_dx=0.01)
    
    generate_video_directory("oscillate_base_0.01", idxs, flag="")