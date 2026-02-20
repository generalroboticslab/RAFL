import sys

sys.path.append("../")
import time
import os
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from tqdm import tqdm
from markermatch import init_realdata, init_simenv
from _utils import ArmDataset
from residual_physics.network import MLPResidual, ResMLPResidual2
from residual_physics.element_force_update import ElementResidual
from py_diff_pd.common.common import ndarray
from py_diff_pd.common.tet_mesh import get_boundary_face
from sopra_residual_physics import SoPrAResidualPhysics

def train(
    sopra_env, save_folder, transformed_markers, real_p, start_frame=0, end_frame=999, num_epochs=100,
):

    training_options = yaml.safe_load(open(f"{save_folder}/config.yaml"))
    if training_options['model'] == "skip_connection":
        residual_network = ResMLPResidual2(sopra_env._dofs * 3, sopra_env._dofs, num_mlp_blocks=training_options['num_mlp_blocks'], num_block_layer=training_options['num_block_layer'])
    elif training_options['model'] == "mlp":
        residual_network = MLPResidual(sopra_env._dofs * 3, sopra_env._dofs, hidden_sizes=training_options['hidden_sizes'])
    elif training_options['model'] == 'element':
        g = training_options['state_force_parameters']

        la = (
                sopra_env._youngs_modulus
                * sopra_env._poissons_ratio
                / ((1 + sopra_env._poissons_ratio) * (1 - 2 * sopra_env._poissons_ratio))
            )
        m = sopra_env._youngs_modulus / (2 * (1 + sopra_env._poissons_ratio))

        mesh = sopra_env._deformable.mesh()

        elements = []
        mu = []
        lam = []
        rho = []
        num_elements = mesh.NumOfElements()
        for e in range(num_elements):
            elements.append(mesh.py_element(e))
            mu.append(m)
            lam.append(la)
            rho.append(sopra_env._deformable.density())
        
        surface_faces = sopra_env._get_boundary_ordered()
        elements = ndarray(elements)
        mu = ndarray(mu)
        lam = ndarray(lam)
        rho = ndarray(rho)
        
        inner_faces = np.concatenate(sopra_env._inner_faces)
        force_nodes = []
        for face in inner_faces:
            force_nodes.extend(face)

        force_nodes = np.array(list(set(force_nodes)))


        residual_network = ElementResidual(sopra_env._dofs, 
                                                torch.tensor(elements), 
                                                torch.tensor(surface_faces),
                                                torch.tensor(mu), 
                                                torch.tensor(lam), 
                                                torch.tensor(rho), 
                                                sopra_env._q0, 
                                                None,
                                                hidden_size=training_options['hidden_size'],
                                                num_hidden_layer=training_options['num_hidden_layer'],
                                                actuated=training_options['actuated'],
                                                force_nodes=torch.tensor(force_nodes),
                                                normalize_inputs=training_options['normalize_inputs'],
                                                separated=training_options['separated'] if 'separated' in training_options else True,
                                                conditioned=training_options['conditioned'] if 'conditioned' in training_options else True,
                                                stress=training_options['stress'] if 'stress' in training_options else False
                                                )

    residual_network.train()

    pairs = [[0, 5], [1, 4], [2, 2], [3, 0], [4, 1], [5, 3]]
    real_chambers = [pairs[i][1] for i in range(len(pairs))]
    data_folder = "augmented_dataset_smaller_tol"
    loss_fn = torch.nn.MSELoss(reduction="mean")

    training_set = ArmDataset(
    training_options["training_set"],
    sopra_env,
    f"../preprocess_data/{data_folder}",
    start_frame=training_options["start_frame"],
    end_frame=training_options["end_frame"],
    )
    
    trainable_params = [p for p in residual_network.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
            trainable_params,
            lr=training_options["learning_rate"],
            weight_decay=training_options["weight_decay"],
        )



    normalize = training_options["normalize"]
    #f_mean, f_std = torch.mean(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten(), torch.std(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
    f_mean, f_std = torch.zeros(training_set.q_init.shape[0]).flatten(), torch.std(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()


    total_loss_history = []
    val_loss_history = []

    with tqdm(num_epochs) as qbar:
        for epoch in range(num_epochs):
            train_loss = 0
            for data_idx in tqdm(training_options["training_set"]):
                
                optimizer.zero_grad()
                total_loss = []

                q_res = torch.from_numpy(sopra_env._q0)
                v_res = torch.zeros_like(q_res)
                
                for frame_i in range(1, end_frame):
                    
                    pressure = real_p[data_idx, frame_i - 1]
                    
                    f_ext_res = sopra_env.apply_inner_pressure(
                        pressure, q_res.detach().numpy(), chambers=real_chambers
                    )
                    
                    f_ext_res = torch.from_numpy(f_ext_res)
                    
                    if normalize:
                        (
                            q_res_normalized,
                            v_res_normalized,
                            f_ext_res_normalized,
                        ) = training_set.normalize(q=q_res - q_init, v=v_res, p=f_ext_res)
                        res_force_normalized = residual_network(
                            torch.cat(
                                (q_res_normalized, v_res_normalized, f_ext_res_normalized),
                                dim=0,
                            ).expand(1, -1)
                        )[0]
                        res_force = training_set.denormalize(f=res_force_normalized)[0]
                    else:
                        res_force_normalized = residual_network(
                            torch.cat((q_res, v_res, f_ext_res), dim=0)
                        )[0]
                        
                        res_force = res_force_normalized
                    
                    try:
                        
                        q_res, v_res = sopra_env.forward(
                            q_res, v_res, f_ext=f_ext_res + res_force, dt=0.01
                        )
                    except:
                        print("Solver fails at frame", frame_i)
                        break

                    qx_marker = sopra_env.return_simulated_markers(q_res.reshape(-1, 3))

                    data_loss = ((-qx_marker.flatten() + transformed_markers[data_idx, frame_i].flatten())**2).sum()
                    loss = data_loss + 1e-4 * (res_force**2).sum()
                    total_loss.append(loss)
                
                print(torch.stack(total_loss).mean())
                total_loss = training_options["scale"] * torch.stack(total_loss).mean() 
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item() 
            
            train_loss /= len(training_options["training_set"])
            total_loss_history.append(train_loss)

            with torch.no_grad():
                val_loss = 0
                for data_idx in tqdm(training_options["validate_set"]):
                    
                    total_loss = []

                    q_res = torch.from_numpy(sopra_env._q0)
                    v_res = torch.zeros_like(q_res)
                    
                    for frame_i in range(1, end_frame):
                        
                        pressure = real_p[data_idx, frame_i - 1]
                        
                        f_ext_res = sopra_env.apply_inner_pressure(
                            pressure, q_res.detach().numpy(), chambers=real_chambers
                        )
                        
                        f_ext_res = torch.from_numpy(f_ext_res)
                        
                        if normalize:
                            (
                                q_res_normalized,
                                v_res_normalized,
                                f_ext_res_normalized,
                            ) = training_set.normalize(q=q_res - q_init, v=v_res, p=f_ext_res)
                            res_force_normalized = residual_network(
                                torch.cat(
                                    (q_res_normalized, v_res_normalized, f_ext_res_normalized),
                                    dim=0,
                                ).expand(1, -1)
                            )[0]
                            res_force = training_set.denormalize(f=res_force_normalized)[0]
                        else:
                            res_force_normalized = residual_network(
                                torch.cat((q_res, v_res, f_ext_res), dim=0)
                            )[0]
                            
                            res_force = res_force_normalized
                        
                        try:
                            
                            q_res, v_res = sopra_env.forward(
                                q_res, v_res, f_ext=f_ext_res + res_force, dt=0.01
                            )
                        except:
                            print("Solver fails at frame", frame_i)
                            break

                        qx_marker = sopra_env.return_simulated_markers(q_res.reshape(-1, 3))

                        data_loss = ((-qx_marker.flatten() + transformed_markers[data_idx, frame_i].flatten())**2).sum()
                        loss = data_loss 
                        total_loss.append(loss)
                    
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
                        f"{save_folder}/residual_network.pth",
                    )
            qbar.set_description(
                    f"Epoch {epoch}, Train Loss: {train_loss:.3E}, Val Loss: {val_loss:.3E}"
                )
            qbar.update(1)

if __name__ == "__main__":


    poissons_ratio = 0.45
    youngs_modulus = 215856
    density = 1.07e3
    state_force = [0, 0, -9.80709]
    model_name = "sopra_494"
    model = f"../sopra_model/{model_name}.vtk"
    params = {
        "density": density,
        "youngs_modulus": youngs_modulus,
        "poissons_ratio": poissons_ratio,
        "state_force_parameters": state_force,
        "mesh_type": "tet",
        "refinement": 1,
        "arm_file": model,
    }

    config = {}
    config["seed"] = 42
    config["epochs"] = 1000
    config["batch_size"] = 256
    config["learning_rate"] = 8e-3
    config["optimizer"] = "adam"
    config["start_frame"] = 0
    config["end_frame"] = 999
    config["training_set"] = list(range(60)) #160))
    config["validate_set"] = list(range(160,180))
    config["cuda"] = 1
    config["normalize"] = False 
    config["Inialization"] = 1e-3
    config["scale"] = 1e3 #e6
    config["data_type"] = "optimized"
    config["weight_decay"] = 1e-5
    # config["validate_physics"] = True
    # config["validate_epochs"] = 20
    # config["transfer_learning_model"] = f'training/fit_500_batch_sopra_4942/residual_network.pth'
    config["fit"] = "forces"
    # config["fit"] = "SITL"  
    # config["model"] = "skip_connection"
    config["model"] = "element"
    config["tolerance"] = 121
    config["num_mlp_blocks"] = 5
    config["hidden_size"] = 64
    config["num_hidden_layer"] = 4
    config["actuated"] = True
    config["normalize_inputs"] = False
    config["separated"] = False
    config["conditioned"] = False
    config['stress'] = False

    # save_folder = f"training/test_refactor"
    save_folder = f"training/test_refactor_element_transformer_direct"
    config["data_folder"] = save_folder.replace("training/", "")

    sopra_residual = SoPrAResidualPhysics(config, save_folder, params)

    torch.manual_seed(config["seed"])
    with open(f'{save_folder}/config.yaml', 'w') as f:
        yaml.dump(params, f)
        yaml.dump(config, f)

    max_pressure = 200
    # max_pressure = 350
    real_p, base_q = init_realdata(
        f"../arm_data_sep_4/captured_data_200traj_1000timesteps_{max_pressure}pressure.npy"
    )
    arm_folder = model_name = "sopra_494"
    model = f"../sopra_model/{model_name}.vtk"


    options = {}
    options["poissons_ratio"] = 0.45
    options["youngs_modulus"] = 215856
    sopra_env, method, opt = init_simenv(model, arm_folder, options)

    sopra_env.set_measured_markers()
    measured_markers = sopra_env.get_measured_markers()
    steady_state = base_q[0, 0]
    R, t = sopra_env.fit_realframe(steady_state)
    real_markers = steady_state @ R.T + t

    sopra_env.compute_interpolation_coeff(real_markers[3:])
    transformed_markers = base_q[:, :, 3:] @ R.T + t

    train(sopra_env, save_folder, torch.from_numpy(transformed_markers), real_p, "force", end_frame=999)