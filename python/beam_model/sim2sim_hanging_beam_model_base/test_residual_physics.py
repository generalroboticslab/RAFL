import sys

sys.path.append("../")
sys.path.append("../../")
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from _utils import UnderwaterCantileverDataset
from _visualization import plot_trajectory, plot_forces_norm
from env_hanging_cantilever import HangingCantileverEnv3d
from residual_physics.network import ResMLPResidual2, MLPResidual
from residual_physics.element_force import ElementResidual
from residual_physics.implicit_force import ImplicitResidual
from py_diff_pd.common.common import ndarray
from video_generation import *

args = argparse.ArgumentParser()
args.add_argument("-model", dest="model", required=False)

def test_trajectory(
    hex_params, save_folder, test_data_idx,  start_frame=0, end_frame=150,
):

    target_data = np.load(f"data_real_sim2sim/trajectory{test_data_idx}.npy", allow_pickle=True)[()]
    hex_params["amplitude"] = target_data["amplitude"]
    print(hex_params)
    cantilever = HangingCantileverEnv3d(42, 'oscillate_base', hex_params)
    cantilever_sim = HangingCantileverEnv3d(42, 'oscillate_base', hex_params)

    boundary_indices = []
    nonboundary_indices = []
    for i in range(0,cantilever._dofs,3):
        if cantilever.is_dirichlet_dof(i):
            boundary_indices.extend([i,i+1,i+2])
        else:
            nonboundary_indices.extend([i,i+1,i+2])

    training_options = yaml.safe_load(open(f"{save_folder}/config.yaml"))
    dofs = cantilever._dofs

    assert(dofs == cantilever_sim._dofs)
    if training_options["model"] == "skip_connection":
        residual_network = ResMLPResidual2(dofs * 2, dofs, hidden_size=training_options['hidden_size'], num_mlp_blocks=training_options['num_mlp_blocks'], num_block_layer=training_options['num_hidden_layer'])
    elif training_options["model"] == "MLP":
        residual_network = MLPResidual(dofs*2, dofs)
    elif training_options['model'] == 'element':
        youngs_modulus = hex_params['youngs_modulus']
        poissons_ratio = hex_params['poissons_ratio']
        la = (
            youngs_modulus
            * poissons_ratio
            / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        )
        m = youngs_modulus / (2 * (1 + poissons_ratio))

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

        residual_network = ElementResidual(cantilever._dofs, 
                                            torch.tensor(elements), 
                                            torch.tensor(mu), 
                                            torch.tensor(lam), 
                                            torch.tensor(rho),
                                            cantilever._q0, 
                                            mesh.dx(),
                                            hidden_size=training_options['hidden_size'],
                                            num_hidden_layer=training_options['num_hidden_layer'],
                                            actuated=training_options['actuated'],
                                            gravity=True,
                                            changing_boundary_indices=boundary_indices
                                            )

        # vert_num = mesh.NumOfVertices()
        # for i in range(vert_num):
        #     print("mesh: ", ndarray(mesh.py_vertex(i)))
        #     print("env: ", cantilever._q0[3*i: 3* i + 3])
    elif training_options["model"] == "implicit":

            youngs_modulus = cantilever._youngs_modulus
            poissons_ratio = cantilever._poissons_ratio 
            la = (
                youngs_modulus
                * poissons_ratio
                / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
            )
            m = youngs_modulus / (2 * (1 + poissons_ratio))

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

            residual_network = ImplicitResidual(cantilever._dofs, 
                                                    torch.tensor(elements), 
                                                    torch.tensor(rho), 
                                                    cantilever.q0, 
                                                    mesh.dx(),
                                                    hidden_size=training_options['hidden_size'],
                                                    num_hidden_layer=training_options['num_hidden_layer'],
                                                    actuated=training_options['actuated'],
                                                    gravity = True,
                                                    )

    model_input = args.parse_args().model
    model_input = f"residual_network"
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{save_folder}/{model_input}.pth")
    model = torch.load(f"{save_folder}/{model_input}.pth", map_location=device)

    # create new OrderedDict that does not contain `module.`
    if 'module' in list(model['model'].keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model['model'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model['model'] = new_state_dict

    residual_network.load_state_dict(model["model"])
    print("The model saves at epoch", model["epoch"])
    residual_network.eval()

    
    loss_fn = torch.nn.MSELoss(reduction="mean")

    training_set = UnderwaterCantileverDataset(
    training_options["training_set"],
    cantilever._q0,
    f"cantilever_data_sim2sim",
    start_frame=training_options["start_frame"],
    end_frame=training_options["end_frame"],
    )

    ground_truth = np.load(
        f"cantilever_data_sim2sim/optimized_data_{test_data_idx}.npy",
        allow_pickle=True,
    )[()]

    print("amplitude: ", ground_truth['amplitude'])

    f_optimized = torch.from_numpy(ground_truth["optimized_forces"]).t()[1:]

    target_trajectory = torch.from_numpy(target_data['q'])
    target_trajectory_v = torch.from_numpy(target_data['v'])
    q0 = target_trajectory[0]
    v0 = target_trajectory_v[0]
    q_init = cantilever.q0.clone()
    # q0 = cantilever.q0.clone()
    # v0 = cantilever.v0.clone()
    q_sim = q0.detach().clone()
    v_sim = v0.detach().clone()
    q_res = q0.detach().clone()
    v_res = v0.detach().clone()
    dt = 0.01
    prepare = 100

    Path(f"{save_folder}/visualize/{test_data_idx}").mkdir(parents=True, exist_ok=True)

    qs_sim = []
    vs_sim = []
    qs_sim.append(q_sim.detach().numpy())
    vs_sim.append(v_sim.detach().numpy())
    qs_res = []
    vs_res = []
    qs_res.append(q_res.detach().numpy())
    vs_res.append(v_res.detach().numpy())
    res_force_errors = []
    predicted_residual_force_norms = []
    ground_truth_residual_force_norms = []
    normalize = training_options["normalize"]
    #f_mean, f_std = torch.mean(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten(), torch.std(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
    f_mean, f_std = torch.zeros(training_set.q_init.shape[0]).flatten(), torch.std(training_set.fs[:,:,nonboundary_indices].view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
    #f_mean, f_std = torch.zeros(training_set.q_init.shape[0]).flatten(), torch.mean(torch.abs(training_set.fs.view(-1,3)), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
    
    #cantilever.display_mesh(q_res.detach().numpy(), f"{save_folder}/visualize/{test_data_idx}/0.png")

    print(torch.mean(training_set.fs.view(-1,3), dim=0))
    print(f_std[:3])
    last_frame = end_frame
    for frame_i in range(1, end_frame):

        t = (prepare + frame_i) * dt

        # q_res_optimized, v_res_optimized = ground_truth["q_trajectory"][frame_i - 1], ground_truth["v_trajectory"][frame_i - 1]
        # q_res_original, v_res_original = torch.tensor(target_data['q'][frame_i]), torch.tensor(target_data['v'][frame_i])

        with torch.no_grad():
            vx_offset_sim, vx_vel_sim = cantilever_sim.update_boundary(t)
            for node_dof in range(0, cantilever_sim._dofs, 3):
                if cantilever_sim.is_dirichlet_dof(node_dof):
                    q_sim[node_dof] = q_init[node_dof] + vx_offset_sim
                    q_sim[node_dof + 1] = q_init[node_dof + 1]
                    q_sim[node_dof + 2] = q_init[node_dof + 2]
                    v_sim[node_dof] = vx_vel_sim
                    v_sim[node_dof + 1] = 0
                    v_sim[node_dof + 2] = 0

            vx_offset_res, vx_vel_res = cantilever.update_boundary(t)
            for node_dof in range(0, cantilever._dofs, 3):
                if cantilever.is_dirichlet_dof(node_dof):
                    q_res[node_dof] = q_init[node_dof] + vx_offset_res
                    q_res[node_dof + 1] = q_init[node_dof + 1]
                    q_res[node_dof + 2] = q_init[node_dof + 2]
                    v_res[node_dof] = vx_vel_res
                    v_res[node_dof + 1] = 0
                    v_res[node_dof + 2] = 0
        
        # qs_sim.append(q_sim.detach().numpy())
        # vs_sim.append(v_sim.detach().numpy())
        
        # qs_res.append(q_res.detach().numpy())
        # vs_res.append(v_res.detach().numpy())

        # q_res, v_res = torch.tensor(ground_truth["q_trajectory"][frame_i - 1]), torch.tensor(ground_truth["v_trajectory"][frame_i - 1])
        # q_sim, v_sim = torch.tensor(ground_truth["q_trajectory"][frame_i - 1]), torch.tensor(ground_truth["v_trajectory"][frame_i - 1])

        #print("q diff: ", np.linalg.norm(q_res[boundary_indices].clone().detach().numpy().reshape(-1,3) - q_res_optimized[boundary_indices].reshape(-1,3), axis=-1).mean())
        # print("v diff: ", np.linalg.norm(v_res.clone().detach().numpy().reshape(-1,3) - v_res_optimized.reshape(-1,3), axis=-1).mean())

        p = np.array(cantilever._deformable.PyForwardStateForce(q_res.clone().detach().numpy(), v_res.clone().detach().numpy()))
        p_res = torch.tensor(p, dtype=q_res.dtype)
        if normalize:
            (
                q_res_normalized,
                v_res_normalized,
                p_res_normalized,
            ) = training_set.normalize(q=q_res - q_init, v=v_res, p=p_res)
            res_force_normalized = residual_network(
                torch.cat(
                    (q_res_normalized, v_res_normalized), #, p_res_normalized),
                    dim=0,
                ).expand(1, -1)
            )[0]
            res_force = training_set.denormalize(f=res_force_normalized)[0]

        else:
            res_force_normalized = residual_network(
                torch.cat((q_res, v_res), dim=0) #, p_res), dim=0)
            )[0]
            res_force = training_set.denormalize(f=res_force_normalized, normalization_params=(None, None, None, None, None, None, f_mean, f_std))[0]

        res_force = f_optimized[frame_i - 1, :].clone()
        res_force[boundary_indices] = 0
        res_force_error = torch.norm(res_force - f_optimized[frame_i - 1, :])
        res_force_errors.append(res_force_error.item())
        predicted_residual_force_norms.append(torch.norm(res_force.reshape(-1,3), dim=-1).mean().item())
        ground_truth_residual_force_norms.append(torch.norm(f_optimized[frame_i - 1].reshape(-1,3), dim=-1).mean().item())
        # print("ground truth: ", f_optimized[frame_i - 1].reshape(-1,3).mean(dim=0))
        # print("learned: ", res_force.reshape(-1,3).mean(dim=0))

        
        # for node_dof in range(0, cantilever._dofs, 3):
        #     if cantilever.is_dirichlet_dof(node_dof):
        #         #print("res force boundary before: ", res_force[node_dof: node_dof + 3])
        #         res_force[node_dof] = 0
        #         res_force[node_dof + 1] = 0
        #         res_force[node_dof + 2] = 0
        #         #print("res force boundary after: ", res_force[node_dof: node_dof + 3])

        try:
            q_sim, v_sim = cantilever_sim.forward(q_sim, v_sim, f_ext=torch.zeros_like(q_sim), dt=dt)

            # q_res, v_res = cantilever.forward(
            #     q_res, v_res, f_ext=f_optimized[frame_i - 1, :], dt=dt
            # )
            q_res, v_res = cantilever.forward(
                q_res, v_res, f_ext=res_force, dt=dt
            )
            #cantilever.display_mesh(q_res.detach().numpy(), f"{save_folder}/visualize/{test_data_idx}/{frame_i}.png")
        except:
            print("Solver fails at frame", frame_i)
            last_frame = frame_i
            break
        
        qs_sim.append(q_sim.detach().numpy())
        vs_sim.append(v_sim.detach().numpy())
        
        qs_res.append(q_res.detach().numpy())
        vs_res.append(v_res.detach().numpy())

         
    # t = (prepare + last_frame) * dt

    # vx_offset_res, vx_vel_res = cantilever.update_boundary(t)
    # for node_dof in range(0, cantilever._dofs, 3):
    #     if cantilever.is_dirichlet_dof(node_dof):
    #         q_res[node_dof] = q_init[node_dof] + vx_offset_res
    #         q_res[node_dof + 1] = q_init[node_dof + 1]
    #         q_res[node_dof + 2] = q_init[node_dof + 2]
    #         v_res[node_dof] = vx_vel_res
    #         v_res[node_dof + 1] = 0
    #         v_res[node_dof + 2] = 0

    # if normalize:
    #     (
    #         q_res_normalized,
    #         v_res_normalized,
    #         p_res_normalized,
    #     ) = training_set.normalize(q=q_res - q_init, v=v_res, p=p_res)
    #     res_force_normalized = residual_network(
    #         torch.cat(
    #             (q_res_normalized, v_res_normalized), #, p_res_normalized),
    #             dim=0,
    #         ).expand(1, -1)
    #     )[0]
    #     res_force = training_set.denormalize(f=res_force_normalized)[0]

    # else:
    #     res_force_normalized = residual_network(
    #         torch.cat((q_res, v_res), dim=0) #, p_res), dim=0)
    #     )[0]
    #     res_force = training_set.denormalize(f=res_force_normalized, normalization_params=(None, None, None, None, None, None, f_mean, f_std))[0]

    # try:
    #     q_res, v_res = cantilever.forward(
    #         q_res, v_res, f_ext=res_force, dt=dt
    #     )
    #     cantilever.display_mesh(q_res.detach().numpy(), f"{save_folder}/visualize/{test_data_idx}/{end_frame}.png")

    # except:
    #     pass
    # qs_sim.append(q_sim.detach().numpy())
    # vs_sim.append(v_sim.detach().numpy())
    
    # qs_res.append(q_res.detach().numpy())
    # vs_res.append(v_res.detach().numpy())

    np.save(f"{save_folder}/qs_sim_{test_data_idx}.npy", qs_sim)
    np.save(f"{save_folder}/qs_res_{test_data_idx}.npy", qs_res)
    np.save(f"{save_folder}/vs_sim_{test_data_idx}.npy", vs_sim)
    np.save(f"{save_folder}/vs_res_{test_data_idx}.npy", vs_res)
    qs_sim = np.array(qs_sim)
    qs_res = np.array(qs_res)
    qs_ground_truth = np.array(target_data['q']) #np.array(ground_truth["q_trajectory"])

    # print(qs_res.shape)
    # print(qs_ground_truth.shape)

    pairs = [[0, 5], [1, 4], [2, 2], [3, 0], [4, 1], [5, 3]]
    vis_1d_folder = f"sim2sim/{save_folder.replace('training/', '')}_{model_input}"
    os.makedirs(vis_1d_folder, exist_ok=True)
    mm = 1.5 / 25.4
    sim_frames = qs_sim.shape[0] 
    real_frames = sim_frames 
    dt = 0.01
    times = np.linspace(0, sim_frames * dt, sim_frames + 1)[:-1]
    qs_sim = qs_sim.reshape(qs_sim.shape[0], -1,3)
    qs_res = qs_res.reshape(qs_res.shape[0], -1,3)
    qs_ground_truth = qs_ground_truth.reshape(qs_ground_truth.shape[0], -1,3)
    sim_error = np.linalg.norm(qs_sim[:real_frames] - qs_ground_truth[:real_frames], axis=-1)
    res_error = np.linalg.norm(qs_res[:real_frames] - qs_ground_truth[:real_frames], axis=-1)
    predicted_residual_force_norms = np.array(predicted_residual_force_norms)
    res_force_errors = np.array(res_force_errors)
    print(
        f"test id {test_data_idx} sim error {sim_error.mean()*1e3:.3f}mm +-  {sim_error.mean(-1).std()*1e3:.3f} mm"
    )
    print(
        f"res error {test_data_idx} {res_error.mean()*1e3:.3f}mm +-  {res_error.mean(-1).std()*1e3:.3f} mm"
    )
    figsize = (88 * mm, 60 * mm)
    plot_trajectory(
        vis_1d_folder,
        figsize,
        qs_ground_truth,
        None,
        qs_res,
        qs_sim,
        test_data_idx,
        real_frames,
        dt,
        dim=0
    )
    plot_forces_norm(
        vis_1d_folder,
        test_data_idx,
        figsize,
        predicted_residual_force_norms,
        ground_truth_residual_force_norms,
        dt,
    )

    return sim_error, res_error


if __name__ == "__main__":

    save_folder = f"training/test_refactor_element"
    sim_errors = []
    res_errors = []
    for test_i in range(12,20):
        youngs_modulus = 215856 #97350 
        poissons_ratio = 0.45
        density = 1.07e3 
        hex_params = {
            'density': density,
            'youngs_modulus': youngs_modulus,
            'poissons_ratio': poissons_ratio,
            'mesh_type': 'hex',
            'refinement': 1,
        }

        print(f"test id {test_i}")
        sim_error, res_error = test_trajectory(
            hex_params, save_folder, test_i, end_frame=100
        )
        sim_errors.append(sim_error)
        res_errors.append(res_error)
    
    min_length = 100
    for s in sim_errors:
        #print(s.shape)
        if min_length > s.shape[0]:
            min_length = s.shape[0]

    sim_errors = np.array([s[:min_length] for s in sim_errors])
    res_errors = np.array([r[:min_length] for r in res_errors])
    #print(sim_errors.shape)
    sim_error_mean = sim_errors.mean(axis=-1).mean(axis=-1)
    res_error_mean = res_errors.mean(axis=-1).mean(axis=-1)
    print(f"sim error {sim_error_mean.mean() * 1000 :.3f}mm +-  {sim_error_mean.std() * 1000:.3f} mm")
    print(f"res error {res_error_mean.mean() * 1000:.3f}mm +-  {res_error_mean.std() * 1000:.3f} mm")
    print("Frame error: \n")
    print(f"sim error {sim_errors.mean(-1).flatten().mean() * 1000 :.3f}mm +-  {sim_errors.mean(-1).flatten().std() * 1000:.3f} mm")
    print(f"res error {res_errors.mean(-1).flatten().mean() * 1000:.3f}mm +-  {res_errors.mean(-1).flatten().std() * 1000:.3f} mm")
    np.save(f"{save_folder}/sim_errors_residual_network.npy", sim_errors)
    np.save(f"{save_folder}/res_errors_residual_network.npy", res_errors)

    #generate_video_directory(f"{save_folder}/visualize", list(range(12,20)), flag="")