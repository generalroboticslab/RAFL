import sys
sys.path.append('../..')
# sys.path.append('../../..')

import time
from pathlib import Path
import numpy as np
import torch
from env_cantilever_shorter import ShorterCantileverEnv3d

def optimize_init_force(
    num_epochs,
    num_frames,
    target,
    cantilever: ShorterCantileverEnv3d,
    dt,
    id=0,
    res_force=None,
    suffix="fix_registration",
):
    force_nodes_num = 16
    params = torch.normal(
        0, 1e-4, [force_nodes_num, num_frames], dtype=torch.float64, requires_grad=True
    )
    with torch.no_grad():
        if res_force is not None:
            for i in range(num_frames):
                params[:, i] = res_force[-46::3]

    optimizer = torch.optim.Adam([params], lr=1e-3)
    loss_history = []
    target = torch.from_numpy(target)

    for epoch in range(num_epochs):
        start_time = time.time()

        def closure():
            optimizer.zero_grad()
            q0 = cantilever._q0
            q, v = torch.from_numpy(q0), torch.zeros(q0.shape).double()
            loss = 0
            for i in range(num_frames):
                f_ext = torch.zeros(cantilever.dofs, dtype=torch.float64)
                f_ext[-force_nodes_num * 3 + 2 :: 3] += params[:, i]
                q, v = cantilever.forward(q, v, f_ext=f_ext, dt=dt)
                qx = q.reshape(-1, 3)
                qx_marker = cantilever.get_markers_3d(qx)
                loss += ((target - qx_marker) ** 2).sum()

            loss /= num_frames

            ### Additionally add condition on f_ext change to not be too sudden
            threshold = 100
            loss += 1e-7 * (params**2).sum()
            # Backward gradients so we know which direction to update parameters
            loss.backward()

            # with torch.no_grad():
            return loss

        # Actually update parameters
        loss = optimizer.step(closure)
        loss_history.append(loss.item())
        rel_loss = abs(loss.item() - loss_history[-2]) / loss.item() if epoch > 10 else 1
        if rel_loss < 1e-3:
            break
        with np.printoptions(precision=3):
            print(
                f"Epoch [{epoch+1}/{num_epochs}]: {(time.time()-start_time):.2f}s - Loss {loss.item():.4e} "
            )  # - f_ext: {params[0].detach().cpu().numpy():.6f} - grad: {params.grad[0].detach().cpu().numpy():.2e} ")# - Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        ### Early stopping
        if epoch > 0 and abs(loss_history[-1] - loss_history[-2]) < 1e-8:
            break

    q_all = []
    print(f"test{id}")
    path_vis = Path(f"cantilever_data_shorter_{suffix}/test{id}/init_beam")
    path_vis.mkdir(exist_ok=True, parents=True)
    with torch.no_grad():
        q, v = cantilever._q0, torch.zeros(cantilever.dofs, dtype=torch.float64)
        q_all.append(q)
        q = torch.from_numpy(q)

        for i in range(num_frames):
            start_time = time.time()

            f_ext = torch.zeros(cantilever.dofs, dtype=torch.float64)
            f_ext[-force_nodes_num * 3 + 2 :: 3] += params[:, i]
            end_vis = time.time()
            q, v = cantilever.forward(q, v, f_ext=f_ext, dt=dt)
            q_all.append(q.detach().numpy())

            qx_marker = torch.zeros(target.shape).double()
            qx = q.reshape(-1, 3)
            qx_marker = cantilever.get_markers_3d(qx)

            # Time including visualization
            print(
                f"Frame [{i+1}/{num_frames}]: {1000*(time.time()-end_vis):.2f}ms (+ {1000*(end_vis-start_time):.2f}ms for visualization)"
            )
        cantilever.vis_dynamic_sim2real_markers(
            f"init_beam",
            q.detach().numpy(),
            qx_marker.detach().numpy(),
            target.detach().numpy(),
            frame=i,
        )
        Path(f"cantilever_data_shorter_{suffix}").mkdir(exist_ok=True, parents=True)
        np.savez(f"cantilever_data_shorter_{suffix}/q_force_opt{id}_reorder.npz", *q_all)
        np.save(f"cantilever_data_shorter_{suffix}/q_force_opt{id}_init.npy", params.detach().cpu().numpy())
        print("-----------finish visualization---------------")

if __name__ == '__main__':
    weights = [0.05, 0.06, 0.07, 0.1, 0.09, 0.08, 0.11, 0.12, 0.15, 0.09, 0.13, 0.14, 0.16, 0.17, 0.2,0.18,0.22,0.21]
    
    prepare = 100
    forward = 0
    forward_times = []
    transformed_data = []
    idxs = [0,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17]
    for idx in idxs:
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
        cantilever = ShorterCantileverEnv3d(42, f'cantilever_data_shorter_straight/test{idx}', hex_params)
        q_init = cantilever._q0
        q0 = torch.from_numpy(cantilever._q0)
        q_ = q0.reshape(-1, 3)
        v0 = torch.zeros(q0.shape, dtype=torch.float64)
        weight = weights[idx] * 9.80709
        res_force = torch.zeros(q0.shape, dtype=torch.float64)
        res_force[-16::3] = - weight / 16
        qs_real = np.load(f"data_shorter/q{idx}.npy")
        steady_state = qs_real[0] * 1e-3

        R, t = cantilever.fit_realframe(steady_state)

        ######################################
        steady_state_transformed = steady_state @ R.T + t

        ########################################
        target = qs_real[1,:,:] * 1e-3
        target = target @ R.T + t
        cantilever.interpolate_markers_3d(q_.detach().numpy(), steady_state_transformed)
        num_epochs = 300
        num_frames = 150
        optimize_init_force(num_epochs, num_frames, target, cantilever, 0.01, id=idx, res_force=res_force, suffix="straight")
        # target_data = qs_real[1:,:,:] * 1e-3
        # target_data_flatten = np.zeros((target_data.shape[1] * target_data.shape[2], target_data.shape[0]))
        # for i in range(num_frames):
        #     target_data_tmp = target_data[i,:,:]
        #     target_data_tmp = target_data_tmp @ R.T + t
        #     target_data_flatten[:, i] = target_data_tmp.flatten()
        
        # num_frames = 150
        # num_epochs = 300
        # optimize_trajectoryfull(f'test{idx}', cantilever, num_frames, num_epochs, target_data_flatten, 0.01, idx, suffix="straight")