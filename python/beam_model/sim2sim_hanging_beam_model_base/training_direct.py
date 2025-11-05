import sys

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../..")
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from _utils import CantileverDataset
from _visualization import plot_trajectory, plot_forces_norm
from env_hanging_cantilever import HangingCantileverEnv3d
from residual_physics.network import ResMLPResidual2, MLPResidual
from residual_physics.element_force import ElementResidual
from py_diff_pd.common.common import ndarray
from tqdm import tqdm

def fine_tune(
    hex_params, save_folder, start_frame=0, end_frame=150, num_epochs=100, dt=0.01
):

    target_data = np.load(f"data_real_sim2sim/trajectory0.npy", allow_pickle=True)[()]
    hex_params["amplitude"] = target_data["amplitude"]
    print(hex_params)
    cantilever = HangingCantileverEnv3d(42, 'oscillate_base', hex_params)

    boundary_indices = []
    for i in range(0,cantilever._dofs,3):
        if cantilever.is_dirichlet_dof(i):
            boundary_indices.extend([i,i+1,i+2])
    non_boundary_indices = []
    for i in range(0,cantilever._dofs,3):
        if not cantilever.is_dirichlet_dof(i):
            non_boundary_indices.extend([i,i+1,i+2])

    training_options = yaml.safe_load(open(f"{save_folder}/config.yaml"))
    dofs = cantilever._dofs
    if training_options["model"] == "skip_connection":
        residual_network = ResMLPResidual2(dofs * 2, dofs, hidden_size=training_options['hidden_size'], num_mlp_blocks=training_options['num_mlp_blocks'], num_block_layer=training_options['num_hidden_layer'])
    elif training_options["model"] == "MLP":
        residual_network = MLPResidual(dofs*2, dofs)
    elif training_options['model'] == 'element':
        youngs_modulus = 215856
        poissons_ratio = 0.45
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
                                            gravity=True
                                            )

    model_input = f"residual_network"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"{save_folder}/{model_input}.pth")
    # model = torch.load(f"{save_folder}/{model_input}.pth", map_location=device)

    # # create new OrderedDict that does not contain `module.`
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in model['model'].items():
    #     if 'unmodelled_nn' in k:
    #         name = k 
    #         new_state_dict[name] = v
    # model['model'] = new_state_dict

    # residual_network.load_state_dict(model["model"], strict=False)
    # print("The model saves at epoch", model["epoch"])

    residual_network.to(device)

    residual_network.train()

    # for m in residual_network.modules():
    #     if isinstance(m, torch.nn.BatchNorm1d): # Adjust for BatchNorm type
    #         m.weight.requires_grad_(False)
    #         m.bias.requires_grad_(False)
    #         m.eval() # This freezes running_mean and running_var

    training_set = CantileverDataset(
    training_options["training_set"],
    cantilever._q0,
    f"cantilever_data_sim2sim",
    start_frame=training_options["start_frame"],
    end_frame=training_options["end_frame"],
    )
    
    trainable_params = [p for p in residual_network.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
            trainable_params,
            lr= 0.1 * training_options["learning_rate"],
            weight_decay=training_options["weight_decay"],
        )


    normalize = training_options["normalize"]
    #f_mean, f_std = torch.mean(training_set.fs.view(-1,3), dim=0).expand(dofs // 3, 3).flatten(), torch.std(training_set.fs.view(-1,3), dim=0).expand(dofs // 3, 3).flatten()
    f_mean, f_std = torch.zeros(training_set.q_init.shape[0]).flatten(), torch.std(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
        
    total_loss_history = []
    val_loss_history = []
    with tqdm(num_epochs) as qbar:
        for epoch in range(num_epochs):
            train_loss = 0
            for data_idx in training_options["training_set"]:
                target_trajectory_q = torch.from_numpy(np.load(f"data_real_sim2sim/trajectory{data_idx}.npy", allow_pickle=True)[()]['q'])
                target_trajectory_v = torch.from_numpy(np.load(f"data_real_sim2sim/trajectory{data_idx}.npy", allow_pickle=True)[()]['v'])

                hex_params["amplitude"] = np.load(f"data_real_sim2sim/trajectory{data_idx}.npy", allow_pickle=True)[()]["amplitude"]
                #print(hex_params)
                cantilever = HangingCantileverEnv3d(42, 'oscillate_base', hex_params)

                q0 = target_trajectory_q[0].detach().clone()

                q = target_trajectory_q[0].detach().clone()
                v = target_trajectory_v[0].detach().clone()
                optimizer.zero_grad()
                total_loss = []
                for frame_i in range(1, end_frame):

                    t = frame_i * dt

                    # q_res_optimized, v_res_optimized = ground_truth["q_trajectory"][frame_i - 1], ground_truth["v_trajectory"][frame_i - 1]
                    # q_res_original, v_res_original = torch.tensor(target_data['q'][frame_i]), torch.tensor(target_data['v'][frame_i])

                    with torch.no_grad():

                        q_combined, v_combined = q.clone(), v.clone()

                        vx_offset_res, vx_vel_res = cantilever.update_boundary(t)
                        for node_dof in range(0, cantilever._dofs, 3):
                            if cantilever.is_dirichlet_dof(node_dof):
                                q_combined[node_dof] = q0[node_dof] + vx_offset_res
                                q_combined[node_dof + 1] = q0[node_dof + 1]
                                q_combined[node_dof + 2] = q0[node_dof + 2]
                                v_combined[node_dof] = vx_vel_res
                                v_combined[node_dof + 1] = 0
                                v_combined[node_dof + 2] = 0

                        q_combined[non_boundary_indices] = q[non_boundary_indices]
                        v_combined[non_boundary_indices] = v[non_boundary_indices]

                    try:
                        if normalize:
                            (
                                q_res_normalized,
                                v_res_normalized,
                            ) = training_set.normalize(q=q_combined - q_init, v=v)
                            res_force_normalized = residual_network(
                                torch.cat(
                                    (q_res_normalized, v_res_normalized),
                                    dim=0,
                                ).expand(1, -1).to(device)
                            )[0]
                            res_force = training_set.denormalize(f=res_force_normalized.cpu())[0]
                        else:
                            # res_force_normalized = residual_network(
                            #     torch.cat((q_combined, v_combined), dim=0).to(device)
                            # )[0]
                            # res_force = training_set.denormalize(f=res_force_normalized.cpu(), normalization_params=(None, None, None, None, f_mean, f_std))[0]

                            res_force = residual_network(
                                    torch.cat((q_combined, v_combined), dim=0).to(device)
                                )[0]
                                
                        res_force[boundary_indices] = 0

                        q_combined, v_combined = cantilever.forward(q_combined, v_combined, f_ext=res_force, dt=0.01)
                        cantilever.remove_boundary()
                        q[non_boundary_indices], v[non_boundary_indices] = q_combined[non_boundary_indices], v_combined[non_boundary_indices]
                        data_loss = ((target_trajectory_q[frame_i][non_boundary_indices] - q[non_boundary_indices])**2).sum()
                        loss = data_loss 
                        total_loss.append(loss)
                    except:
                        print("Failed full trajectory")
                        break

                total_loss = torch.stack(total_loss).mean() 
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item() 
            
            train_loss /= len(training_options["training_set"])
            total_loss_history.append(train_loss)

            with torch.no_grad():
                val_loss = 0
                for data_idx in training_options["validate_set"]:
                    target_trajectory_q = torch.from_numpy(np.load(f"data_real_sim2sim/trajectory{data_idx}.npy", allow_pickle=True)[()]['q'])
                    target_trajectory_v = torch.from_numpy(np.load(f"data_real_sim2sim/trajectory{data_idx}.npy", allow_pickle=True)[()]['v'])
                    
                    hex_params["amplitude"] = np.load(f"data_real_sim2sim/trajectory{data_idx}.npy", allow_pickle=True)[()]["amplitude"]
                    #print(hex_params)
                    cantilever = HangingCantileverEnv3d(42, 'oscillate_base', hex_params)

                    q0 = target_trajectory_q[0].detach().clone()

                    q = target_trajectory_q[0].detach().clone()
                    v = target_trajectory_v[0].detach().clone()

                    total_loss = []
                    for frame_i in range(1, end_frame):
                        
                        t = frame_i * dt

                        q_combined, v_combined = torch.zeros_like(q), torch.zeros_like(v)

                        vx_offset_res, vx_vel_res = cantilever.update_boundary(t)
                        for node_dof in range(0, cantilever._dofs, 3):
                            if cantilever.is_dirichlet_dof(node_dof):
                                q_combined[node_dof] = q0[node_dof] + vx_offset_res
                                q_combined[node_dof + 1] = q0[node_dof + 1]
                                q_combined[node_dof + 2] = q0[node_dof + 2]
                                v_combined[node_dof] = vx_vel_res
                                v_combined[node_dof + 1] = 0
                                v_combined[node_dof + 2] = 0

                        q_combined[non_boundary_indices] = q[non_boundary_indices]
                        v_combined[non_boundary_indices] = v[non_boundary_indices]

                        try:

                            if normalize:
                                (
                                    q_res_normalized,
                                    v_res_normalized,
                                ) = training_set.normalize(q=q_combined - q_init, v=v)
                                res_force_normalized = residual_network(
                                    torch.cat(
                                        (q_res_normalized, v_res_normalized),
                                        dim=0,
                                    ).expand(1, -1).to(device)
                                )[0]
                                res_force = training_set.denormalize(f=res_force_normalized.cpu())[0]
                            else:
                                # res_force_normalized = residual_network(
                                #     torch.cat((q_combined, v_combined), dim=0).to(device)
                                # )[0]
                                # res_force = training_set.denormalize(f=res_force_normalized.cpu(), normalization_params=(None, None, None, None, f_mean, f_std))[0]
                                res_force = residual_network(
                                    torch.cat((q_combined, v_combined), dim=0).to(device)
                                )[0]

                            res_force[boundary_indices] = 0

                            q_combined, v_combined = cantilever.forward(q_combined, v_combined, f_ext=res_force, dt=0.01)
                            q[non_boundary_indices], v[non_boundary_indices] = q_combined[non_boundary_indices], v_combined[non_boundary_indices]
                            data_loss = ((target_trajectory_q[frame_i][non_boundary_indices] - q[non_boundary_indices])**2).sum()
                            loss = data_loss 
                            total_loss.append(loss)
                        except:
                            break

                    total_loss = torch.stack(total_loss).mean() 
                    val_loss += total_loss.item() 
            
                val_loss /= len(training_options["validate_set"])
                val_loss_history.append(val_loss)
                if val_loss == np.min(val_loss_history):
                    torch.save(
                        {
                            "model": residual_network.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "total_loss_history": total_loss_history,
                            "val_loss_history": val_loss_history,
                            "epoch": epoch,
                        },
                        f"{save_folder}/finetune/residual_network.pth",
                    )
            qbar.set_description(
                    f"Epoch {epoch}, Train Loss: {train_loss:.3E}, Val Loss: {val_loss:.3E}"
                )
            qbar.update(1)


if __name__ == "__main__":

    config = {}
    config["seed"] = 42
    config["epochs"] = 1000
    config["batch_size"] = 256
    config["learning_rate"] = 1e-3
    config["optimizer"] = "adam"
    config["start_frame"] = 0
    config["end_frame"] = 100
    config["training_set"] = list(range(10))
    config["validate_set"] = [10,11]

    config["cuda"] = 0
    config["normalize"] = False 
    config["Inialization"] = 1e-3
    config["scale"] = 1#e6
    config["data_type"] = "optimized"
    config["weight_decay"] = 1e-5
    config["fit"] = "forces"
    # config["fit"] = "SITL"
    config["model"] = "element" #"skip_connection"
    # config["model"] = "MLP"
    config["num_mlp_blocks"] = 5
    config["hidden_size"] = 64
    config["actuated"] = True
    config["num_hidden_layer"] = 4
    # save_folder = f"training/sim2simResMLP5"
    save_folder = f"training/test_refactor_element_direct"
    os.makedirs(f"{save_folder}", exist_ok=True)


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

    torch.manual_seed(config["seed"])
    with open(f'{save_folder}/config.yaml', 'w') as f:
        yaml.dump(hex_params, f)
        yaml.dump(config, f)

    cantilever = HangingCantileverEnv3d(42, 'beam', hex_params)
    q_init = torch.from_numpy(cantilever._q0)

    
    fine_tune(
        hex_params, save_folder, end_frame=100
    )