import sys
sys.path.append("../")
sys.path.append("../..")
sys.path.append("../../..")
import time
import os
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from _utils import CantileverDataset
from _visualization import plot_trajectory, plot_forces_norm
from env_cantilever_new import CantileverEnv3d
from env_cantilever_shorter_scaled import ShorterCantileverEnv3d
from residual_physics.network import ResMLPResidual2, MLPResidual
from residual_physics.element_force_update import ElementResidual as UpdatedElementResidual
from residual_physics.element_force import ElementResidual as OldElementResidual
from py_diff_pd.common.common import ndarray
from py_diff_pd.common.hex_mesh import get_boundary_face


def get_test_params(save_folder):

    if 'longer_scaled' in save_folder:
        return 296147.7, 0.3000
    elif 'longer' in save_folder:
        return 219394.2, 0.4848
    elif 'shorter_scaled' in save_folder:
        return 174513.7, 0.3432
    elif 'shorter' in save_folder:
        return 178143.5, 0.3142
    elif 'thicker_scaled' in save_folder:
        return 215828.9, 0.3407
    elif 'thicker' in save_folder:
        return 191847.5, 0.3591
    elif 'thinner_scaled' in save_folder:
        return 284503.8, 0.4976
    elif 'thinner' in save_folder:
        return 309542.5, 0.4982
    else:
        return 233756.7, 0.4760

def test_trajectory(
    cantilever:CantileverEnv3d, save_folder, test_data_idx, transformed_markers, start_frame=0, end_frame=150, cantilever_sim=None, default_cantilever=None
):
    if cantilever_sim is None:
        cantilever_sim = cantilever
    training_options = yaml.safe_load(open(f"{save_folder}/config.yaml"))
    
    device = torch.device("cpu")
    print(f"{save_folder}/residual_network.pth")
    model = torch.load(f"{save_folder}/residual_network.pth", map_location=device)

    # create new OrderedDict that does not contain `module.`
    if 'module' in list(model['model'].keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model['model'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model['model'] = new_state_dict

    if 'element' in training_options['model']:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model['model'].items():
            if 'unmodelled_nn' in k:
                name = k 
                new_state_dict[name] = v
        model['model'] = new_state_dict

    dofs = cantilever._dofs
    if training_options["model"] == "skip_connection":
        residual_network = ResMLPResidual2(dofs * 2, dofs, hidden_size=training_options['hidden_size'], num_mlp_blocks=training_options['num_mlp_blocks'], num_block_layer=training_options['num_hidden_layer'])
    elif training_options['model'] == 'element':
        g = training_options['state_force_parameters']
        poissons_ratio = 0.45
        youngs_modulus = 215856

        la = (
                cantilever._youngs_modulus
                * cantilever._poissons_ratio
                / ((1 + cantilever._poissons_ratio) * (1 - 2 * cantilever._poissons_ratio))
            )
        m = cantilever._youngs_modulus / (2 * (1 + cantilever._poissons_ratio))

        mesh = cantilever._deformable.mesh()

        elements = []
        mu = []
        lam = []
        rho = []
        num_elements = mesh.NumOfElements()
        for e in range(num_elements):
            elements.append(mesh.py_element(e))
            mu.append(m)
            lam.append(la)
            rho.append(cantilever._deformable.density())
        
        surface_faces = get_boundary_face(mesh)
        elements = ndarray(elements)
        mu = ndarray(mu)
        lam = ndarray(lam)
        rho = ndarray(rho)

        residual_network = UpdatedElementResidual(cantilever._dofs, 
                                            torch.tensor(elements), 
                                            torch.tensor(surface_faces), 
                                            torch.tensor(mu), 
                                            torch.tensor(lam), 
                                            torch.tensor(rho),
                                            cantilever._q0, 
                                            [0.008,0.01,0.01],
                                            hidden_size=training_options['hidden_size'],
                                            num_hidden_layer=training_options['num_hidden_layer'],
                                            actuated=training_options['actuated'],
                                            normalize_inputs=training_options['normalize_inputs'] if 'normalize_inputs' in training_options else True,
                                            multi_shape='all' in save_folder,
                                            separated=training_options['separated'] if 'separated' in training_options else True,
                                            conditioned=training_options['conditioned'] if 'conditioned' in training_options else True
                                            )
    elif training_options['model'] == 'element_old':
        g = training_options['state_force_parameters']
        poissons_ratio = 0.45
        youngs_modulus = 215856

        la = (
                cantilever._youngs_modulus
                * cantilever._poissons_ratio
                / ((1 + cantilever._poissons_ratio) * (1 - 2 * cantilever._poissons_ratio))
            )
        m = cantilever._youngs_modulus / (2 * (1 + cantilever._poissons_ratio))

        mesh = cantilever._deformable.mesh()

        elements = []
        mu = []
        lam = []
        rho = []
        num_elements = mesh.NumOfElements()
        for e in range(num_elements):
            elements.append(mesh.py_element(e))
            mu.append(m)
            lam.append(la)
            rho.append(cantilever._deformable.density())
        
        elements = ndarray(elements)
        mu = ndarray(mu)
        lam = ndarray(lam)
        rho = ndarray(rho)

        residual_network = OldElementResidual(cantilever._dofs, 
                                            torch.tensor(elements), 
                                            torch.tensor(mu), 
                                            torch.tensor(lam), 
                                            torch.tensor(rho),
                                            cantilever._q0, 
                                            0.01,
                                            hidden_size=training_options['hidden_size'],
                                            num_hidden_layer=training_options['num_hidden_layer'],
                                            actuated=training_options['actuated']
                                            )

    residual_network.load_state_dict(model["model"], strict=False)
    print("The model saves at epoch", model["epoch"])
    print(residual_network.count_parameters())
    residual_network.eval()

    # ground_truth = np.load(
    #     f"cantilever_data_straight/optimized_data_{test_data_idx}.npy",
    #     allow_pickle=True,
    # )[()]
    # f_optimized = torch.from_numpy(ground_truth["optimized_forces"]).t()[1:]
    loss_fn = torch.nn.MSELoss(reduction="mean")

    training_set = CantileverDataset(
    training_options["training_set"],
    default_cantilever._q0,
    f"cantilever_data_new_straight",
    start_frame=training_options["start_frame"],
    end_frame=training_options["end_frame"],
    )

    q0 = torch.from_numpy(np.load(f"cantilever_data_shorter_scaled_straight/q_force_opt{test_data_idx}_reorder.npz")['arr_%d' % (len(np.load(f"cantilever_data_shorter_scaled_straight/q_force_opt{test_data_idx}_reorder.npz"))-1)])
    v0 = torch.zeros_like(q0)
    q_sim = q0.clone()
    v_sim = v0.clone()
    q_res = q0.clone()
    v_res = v0.clone()
    q_origin = q0.clone()
    v_origin = v0.clone()
    frame_i = 0

    qs_sim = []
    vs_sim = []
    qs_sim.append(q_sim.detach().numpy())
    vs_sim.append(v_sim.detach().numpy())
    qs_res = []
    vs_res = []
    qs_res.append(q_res.detach().numpy())
    vs_res.append(v_res.detach().numpy())
    qs_origin = []
    vs_origin = []
    qs_origin.append(q_origin.detach().numpy())
    vs_origin.append(v_origin.detach().numpy())
    # res_force_errors = []
    # predicted_residual_force_norms = []
    # ground_truth_residual_force_norms = []
    normalize = training_options["normalize"]
    cantilever.vis_dynamic_sim2real_markers(f"{save_folder.replace('training/', '')}/res_{test_data_idx}", q_res.detach().numpy(), cantilever.get_markers_3d(q_res.reshape(-1,3)).detach().numpy(), transformed_markers[0], frame=0)
    time_sim = 0
    time_res = 0
    time_origin = 0
    time_network = 0
    #f_mean, f_std = torch.mean(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten(), torch.std(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
    f_mean, f_std = torch.zeros(training_set.q_init.shape[0]).flatten(), torch.std(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
    #f_mean, f_std = torch.zeros(training_set.q_init.shape[0]).flatten(), torch.max(training_set.fs.view(-1,3).norm(dim=1), dim=0)[0].expand(training_set.q_init.shape[0])
    #f_mean, f_std = torch.zeros(training_set.q_init.shape[0]).flatten(), torch.mean(torch.abs(training_set.fs.view(-1,3)), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
    # print(f_std[:3])
    # exit()

    test_input = torch.cat([q_init, v0])

    test_output = residual_network(test_input)

    print(torch.mean(torch.abs(training_set.fs.view(-1,3)), dim=0))
    print(torch.abs(test_output.reshape(-1,3)).mean(dim=0))

    #print(residual_network.unmodelled_nn(torch.zeros(13, dtype=torch.float64), torch.zeros(14, dtype=torch.float64), torch.zeros(4, dtype=torch.float64), torch.zeros(11, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)))
    for frame_i in range(1, end_frame):
        if normalize:
            
            if 'element' in training_options['model']:
                ti = time.time()
                res_force_normalized = residual_network(
                    torch.cat((q_res, v_res), dim=0)
                )[0]
                res_force = training_set.denormalize(f=res_force_normalized, normalization_params=(None, None, None, None, f_mean, f_std))[0]
                ti_end = time.time()
                time_network += ti_end - ti
                time_res += ti_end - ti
            else:
                ti = time.time()
                (
                    q_res_normalized,
                    v_res_normalized,
                ) = training_set.normalize(q=q_res - q_init, v=v_res)
                res_force_normalized = residual_network(
                    torch.cat(
                        (q_res_normalized, v_res_normalized),
                        dim=0,
                    ).expand(1, -1)
                )[0]
                res_force = training_set.denormalize(f=res_force_normalized)[0]
                ti_end = time.time()
                time_network += ti_end - ti
                time_res += ti_end - ti
        else:
            ti = time.time()
            res_force_normalized = residual_network(
                torch.cat((q_res, v_res), dim=0)
            )[0]
            res_force = res_force_normalized
            ti_end = time.time()
            time_network += ti_end - ti
            time_res += ti_end - ti
        # res_force_error = torch.norm((res_force - f_optimized[frame_i - 1, :]).reshape(-1,3), dim=-1)
        # predicted_residual_force_norms.append(torch.norm(res_force).item())
        # res_force_errors.append(res_force_error)
        # ground_truth_residual_force_norms.append(
        #     torch.norm(f_optimized[frame_i - 1, :]).item()
        # )

        # res_force = f_optimized[frame_i - 1, :]
        try:
            t0 = time.time()
            q_res, v_res = cantilever.forward(
                q_res, v_res, f_ext=res_force, dt=0.01
            )
            time_res += time.time() - t0
            t1 = time.time()
            q_sim, v_sim = cantilever_sim.forward(q_sim, v_sim, f_ext=torch.zeros_like(q_sim), dt=0.01)
            time_sim += time.time() - t1
            t1 = time.time()
            q_origin, v_origin = cantilever.forward(q_origin, v_origin, f_ext=torch.zeros_like(q_origin), dt=0.01)
            time_origin += time.time() - t1
        except:
            print("Solver fails at frame", frame_i)
            break
        # if frame_i % 10 == 0:
        #     cantilever.vis_dynamic_sim2real_markers(f"{save_folder.replace('training/', '')}/qs_res_{test_data_idx}", q_res.detach().numpy(), cantilever.get_markers_3d(q_res.reshape(-1,3)).detach().numpy(), transformed_markers[frame_i], frame=frame_i)
        #     cantilever.vis_dynamic_sim2real_markers(f"{save_folder.replace('training/', '')}/qs_sim_{test_data_idx}", q_sim.detach().numpy(), cantilever.get_markers_3d(q_sim.reshape(-1,3)).detach().numpy(), transformed_markers[frame_i], frame=frame_i)
        qs_sim.append(q_sim.detach().numpy())
        qs_res.append(q_res.detach().numpy())
        vs_sim.append(v_sim.detach().numpy())
        vs_res.append(v_res.detach().numpy())
        qs_origin.append(q_origin.detach().numpy())
        vs_origin.append(v_origin.detach().numpy())
    np.save(f"{save_folder}/shorter_scaled/qs_sim_{test_data_idx}.npy", qs_sim)
    np.save(f"{save_folder}/shorter_scaled/qs_res_{test_data_idx}.npy", qs_res)
    np.save(f"{save_folder}/shorter_scaled/vs_sim_{test_data_idx}.npy", vs_sim)
    np.save(f"{save_folder}/shorter_scaled/vs_res_{test_data_idx}.npy", vs_res)
    np.save(f"{save_folder}/shorter_scaled/qs_origin_{test_data_idx}.npy", qs_origin)
    np.save(f"{save_folder}/shorter_scaled/vs_origin_{test_data_idx}.npy", vs_origin)
    qs_sim = np.array(qs_sim)
    qs_res = np.array(qs_res)
    qs_origin = np.array(qs_origin)
    

    pairs = [[0, 5], [1, 4], [2, 2], [3, 0], [4, 1], [5, 3]]
    vis_1d_folder = f"shorter_scaled_2dplots_displacement/{save_folder.replace('training/', '')}_residual_network"
    os.makedirs(vis_1d_folder, exist_ok=True)
    mm = 1.5 / 25.4
    sim_markers = []
    res_markers = []
    origin_markers = []
    sim_frames = qs_sim.shape[0]
    dt = 0.01
    times = np.linspace(0, sim_frames * dt, sim_frames + 1)[:-1]
    for frame_i in range(sim_frames):
        sim_markers.append(
            cantilever.get_markers_3d(
                torch.from_numpy(qs_sim[frame_i].reshape(-1, 3))
            )
            .detach()
            .numpy()
        )
        res_markers.append(
            cantilever.get_markers_3d(
                torch.from_numpy(qs_res[frame_i].reshape(-1, 3))
            )
            .detach()
            .numpy()
        )
        origin_markers.append(
            cantilever.get_markers_3d(
                torch.from_numpy(qs_origin[frame_i].reshape(-1, 3))
            )
            .detach()
            .numpy()
        )
    sim_markers = np.array(sim_markers)
    res_markers = np.array(res_markers)
    origin_markers = np.array(origin_markers)
    real_frames = res_markers.shape[0]
    print(sim_markers.shape)
    sim_markers_error = np.linalg.norm(sim_markers[:real_frames] - transformed_markers[:real_frames], axis=-1)
    res_markers_error = np.linalg.norm(res_markers[:real_frames] - transformed_markers[:real_frames], axis=-1)
    origin_markers_error = np.linalg.norm(origin_markers[:real_frames] - transformed_markers[:real_frames], axis=-1)
    # predicted_residual_force_norms = np.array(predicted_residual_force_norms)
    # res_force_errors = torch.stack(res_force_errors,dim=0).detach().numpy()
    print(
        f"test id {test_data_idx} sim error {sim_markers_error.mean()*1e3:.3f}mm +-  {sim_markers_error.mean(-1).std()*1e3:.3f} mm"
    )
    print(
        f"res error {test_data_idx} {res_markers_error.mean()*1e3:.3f}mm +-  {res_markers_error.mean(-1).std()*1e3:.3f} mm"
    )
    # print(
    #     f"res force error {test_data_idx} {res_force_errors.mean():.5f}N +-  {res_force_errors.mean(-1).std():.5f}N"
    # )
    print(f"origin error {origin_markers_error.mean()*1e3:.3f}mm +-  {origin_markers_error.mean(-1).std()*1e3:.3f} mm")
    print(f"Sim time {time_sim:.3f} s, Res time {time_res:.3f} s, origin time {time_origin:.3f} s, Network time {time_network:.3f} s")
    figsize = (44 * mm, 30 * mm)
    plot_trajectory(
        vis_1d_folder,
        figsize,
        transformed_markers,
        sim_markers,
        res_markers,
        origin_markers,
        test_data_idx,
        real_frames,
        dt,
    )
    # plot_forces_norm(
    #     vis_1d_folder,
    #     test_data_idx,
    #     figsize,
    #     predicted_residual_force_norms,
    #     ground_truth_residual_force_norms,
    #     dt,
    # )

    return sim_markers_error, res_markers_error, time_sim, time_res, time_network, origin_markers_error, time_origin

if __name__ == "__main__":

    weights = [0.05, 0.06, 0.07, 0.1, 0.09, 0.08, 0.11, 0.12, 0.15, 0.09, 0.13, 0.14, 0.16, 0.17, 0.2,0.18,0.22,0.21]


    save_folder =  f"training/test_refactor_element_nonWeighted_try_unseparated_unconditioned_direct"
    sim_errors = []
    res_errors = []
    origin_errors = []
    time_sim_total = 0
    time_res_total = 0
    time_origin_total = 0
    time_network_total = 0
    os.makedirs(f"beam_shorter/{save_folder.replace('training/', '')}", exist_ok=True)
    os.makedirs(f"beam_shorter/{save_folder.replace('training/', '')}", exist_ok=True)
    os.makedirs(f"{save_folder}/shorter_scaled", exist_ok=True)
    for test_i in [2,7,11,14,16]: 

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

        cantilever = ShorterCantileverEnv3d(42, 'beam_shorter', hex_params)
        # hex_params['youngs_modulus'] = 74523.2 #174513.7 
        # hex_params['poissons_ratio'] = 0.499 #0.3432 
        # hex_params['youngs_modulus'] = 233756.7   #116260.6 #233756.7  
        # hex_params['poissons_ratio'] = 0.4760 #0.499 #0.4760 

        hex_params['youngs_modulus'], hex_params['poissons_ratio'] = get_test_params(save_folder)

        cantilever_sim = ShorterCantileverEnv3d(42, 'beam', hex_params)
        q_init = torch.from_numpy(cantilever._q0)
        q0 = torch.from_numpy(cantilever._q0)
        q_ = q0.reshape(-1, 3)
        v0 = torch.zeros(q0.shape, dtype=torch.float64)

        default_cantilever = CantileverEnv3d(42, 'beam', hex_params)

        # qs_real_ = np.load("weight_data_ordered/q_data_reorder.npz")
        # steady_state = qs_real_[f'arr_0'][:, :, -1] * 1e-3
        
        qs_real = np.load(f"data_shorter/q{test_i}.npy")
        steady_state = qs_real[0] * 1e-3
        
        R, t = cantilever.fit_realframe(steady_state)
        # print(R,t)

        # R = np.eye(3)
        # t = np.zeros(3)
        steady_state_transformed = steady_state @ R.T + t

        cantilever.interpolate_markers_3d(q_.detach().numpy(), steady_state_transformed)


        print(f"test id {test_i}")
        #real_markers = np.load(f"weight_data_ordered/qs_real{test_i}_reorder.npy") * 1e-3
        real_markers = qs_real[1:,:,:] * 1e-3
        transformed_markers = np.zeros((150, 32,3))
        for i in range(150):
            transformed_markers[i] = real_markers[i, :, :] @ R.T + t
        sim_error, res_error, time_sim, time_res, time_network, origin_error, time_origin = test_trajectory(
            cantilever, save_folder, test_i, transformed_markers, end_frame=140, cantilever_sim=cantilever_sim, default_cantilever=default_cantilever
        )
        time_res_total += time_res
        time_sim_total += time_sim
        time_origin_total += time_origin
        time_network_total += time_network
        sim_errors.append(sim_error)
        res_errors.append(res_error)
        origin_errors.append(origin_error)
    sim_errors = np.array(sim_errors)
    res_errors = np.array(res_errors)
    origin_errors = np.array(origin_errors)
    sim_error_mean = sim_errors.mean(axis=-1).mean(axis=-1)
    res_error_mean = res_errors.mean(axis=-1).mean(axis=-1)
    origin_error_mean = origin_errors.mean(axis=-1).mean(axis=-1)
    print(f"sim error {sim_error_mean.mean() *1000 :.3f}mm +-  {sim_error_mean.std() * 1000:.3f} mm")
    print(f"res error {res_error_mean.mean() * 1000:.3f}mm +-  {res_error_mean.std() * 1000:.3f} mm")
    time_res_mean = time_res_total / 6
    time_sim_mean = time_sim_total / 6
    time_origin_mean = time_origin_total / 6
    time_network_mean = time_network_total / 6
    print(f"Total Sim time {time_sim_mean:.3f} s, Total Res time {time_res_mean:.3f} s, Total Network time {time_network_mean:.3f} s, Total origin time {time_origin_mean:.3f} s")
    print("Frame error: \n")
    print(f"sim error {sim_errors.mean(-1).flatten().mean() *1000 :.3f}mm +-  {sim_errors.mean(-1).flatten().std() * 1000:.3f} mm")
    print(f"res error {res_errors.mean(-1).flatten().mean() * 1000:.3f}mm +-  {res_errors.mean(-1).flatten().std() * 1000:.3f} mm")
    print(f"origin error {origin_error_mean.mean() * 1000:.3f}mm +-  {origin_errors.mean(-1).flatten().std() * 1000:.3f} mm")
    np.save(f"{save_folder}/shorter_scaled/sim_errors_residual_network.npy", sim_errors)
    np.save(f"{save_folder}/shorter_scaled/res_errors_residual_network.npy", res_errors)
