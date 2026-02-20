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
from env_cantilever import CantileverEnv3d
from residual_physics.network import ResMLPResidual2, MLPResidual
from residual_physics.element_force_update import ElementResidual
from py_diff_pd.common.common import ndarray
from tqdm import tqdm
from py_diff_pd.common.hex_mesh import get_boundary_face

def train(
    cantilever:CantileverEnv3d, save_folder, start_frame=0, end_frame=150, num_epochs=100, cantilever_sim=None
):

    if cantilever_sim is None:
        cantilever_sim = cantilever
    training_options = yaml.safe_load(open(f"{save_folder}/config.yaml"))
    # with open(f"{save_folder}/config.yaml", 'w') as f:
    #     yaml.dump(training_options, f)
    dofs = cantilever._dofs
    if training_options["model"] == "skip_connection":
        residual_network = ResMLPResidual2(dofs * 2, dofs, hidden_size=training_options['hidden_size'], num_mlp_blocks=training_options['num_mlp_blocks'], num_block_layer=training_options['num_hidden_layer'])
    elif training_options["model"] == "MLP":
        residual_network = MLPResidual(dofs*2, dofs)
    elif training_options['model'] == 'element':
        g = training_options['state_force_parameters']
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
        
        surface_faces = get_boundary_face(mesh)
        elements = ndarray(elements)
        mu = ndarray(mu)
        lam = ndarray(lam)
        rho = ndarray(rho)

        residual_network = ElementResidual(cantilever._dofs, 
                                            torch.tensor(elements), 
                                            torch.tensor(surface_faces),
                                            torch.tensor(mu), 
                                            torch.tensor(lam), 
                                            torch.tensor(rho),
                                            cantilever._q0, 
                                            0.01,
                                            hidden_size=training_options['hidden_size'],
                                            num_hidden_layer=training_options['num_hidden_layer'],
                                            actuated=training_options['actuated'],
                                            normalize_inputs=training_options['normalize_inputs'],
                                            separated=training_options['separated'] if 'separated' in training_options else True,
                                            conditioned=training_options['conditioned'] if 'conditioned' in training_options else True,
                                            stress=training_options['stress'] if 'stress' in training_options else False
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
            lr= training_options["learning_rate"],
            weight_decay=training_options["weight_decay"],
        )


    normalize = training_options["normalize"]
    #f_mean, f_std = torch.mean(training_set.fs.view(-1,3), dim=0).expand(dofs // 3, 3).flatten(), torch.std(training_set.fs.view(-1,3), dim=0).expand(dofs // 3, 3).flatten()
    f_mean, f_std = torch.zeros(training_set.q_init.shape[0]).flatten(), torch.std(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
    
    decay_rate_cycle = np.linspace(0.91, 1.0, 10)
    
    total_loss_history = []
    val_loss_history = []
    with tqdm(num_epochs) as qbar:
        for epoch in range(num_epochs):
            train_loss = 0
            for data_idx in training_options["training_set"]:
                target_trajectory_q = torch.from_numpy(np.load(f"data_real/trajectory{data_idx}.npy", allow_pickle=True)[()]['q'])
                target_trajectory_v = torch.from_numpy(np.load(f"data_real/trajectory{data_idx}.npy", allow_pickle=True)[()]['v'])

                q = target_trajectory_q[0].detach().clone()
                v = target_trajectory_v[0].detach().clone()
                optimizer.zero_grad()
                total_loss = []

                for frame_i in range(1, end_frame):
                    # if epoch < 10:
                    #     q = target_trajectory_q[frame_i - 1].detach().clone()
                    #     v = target_trajectory_v[frame_i - 1].detach().clone()
                    try:
                        if normalize:
                            if 'element' in training_options['model']:
                                res_force_normalized = residual_network(
                                    torch.cat((q, v), dim=0).to(device)
                                )[0]
                                res_force = training_set.denormalize(f=res_force_normalized.cpu(), normalization_params=(None, None, None, None, f_mean, f_std))[0]
                            else:
                                (
                                    q_res_normalized,
                                    v_res_normalized,
                                ) = training_set.normalize(q=q - q_init, v=v)
                                res_force_normalized = residual_network(
                                    torch.cat(
                                        (q_res_normalized, v_res_normalized),
                                        dim=0,
                                    ).expand(1, -1).to(device)
                                )[0]
                                res_force = training_set.denormalize(f=res_force_normalized.cpu())[0]
                        else:
                            if 'element' in training_options['model']:
                                res_force_normalized = residual_network(
                                    torch.cat((q, v), dim=0).to(device)
                                )[0]
                            else:
                                res_force_normalized = residual_network(
                                    torch.cat((q - q_init, v), dim=0).expand(1, -1).to(device)
                                )[0]
                            # res_force = training_set.denormalize(f=res_force_normalized.cpu(), normalization_params=(None, None, None, None, f_mean, f_std))[0]
                            res_force = res_force_normalized.cpu()

                        q, v = cantilever.forward(q, v, f_ext=res_force, dt=0.01)
                        data_loss = ((target_trajectory_q[frame_i] - q)**2).sum()
                        loss = data_loss + 1e-4 * (res_force**2).sum()
                        total_loss.append(loss)
                    except Exception as e:
                        # Get the exception type name and message
                        exception_type = type(e).__name__
                        exception_message = str(e)
                        print(f"An error of type '{exception_type}' occurred: {exception_message}")
                        # Output: An error of type 'NameError' occurred: name 'x' is not defined
                        print("Failed full trajectory")
                        break

                # total_loss = (torch.pow(decay_rate_cycle[epoch % 10], torch.arange(len(total_loss))) * torch.stack(total_loss)).mean() 

                total_loss = training_options["scale"] * torch.stack(total_loss).mean() 
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item() 
            
            train_loss /= len(training_options["training_set"])
            total_loss_history.append(train_loss)

            with torch.no_grad():
                val_loss = 0
                for data_idx in training_options["validate_set"]:
                    target_trajectory_q = torch.from_numpy(np.load(f"data_real/trajectory{data_idx}.npy", allow_pickle=True)[()]['q'])
                    target_trajectory_v = torch.from_numpy(np.load(f"data_real/trajectory{data_idx}.npy", allow_pickle=True)[()]['v'])
                    
                    q = target_trajectory_q[0].detach().clone()
                    v = target_trajectory_v[0].detach().clone()

                    total_loss = []
                    for frame_i in range(1, end_frame):

                        # q = target_trajectory_q[frame_i - 1].detach().clone()
                        # v = target_trajectory_v[frame_i - 1].detach().clone()
                        try:

                            if normalize:
                                if 'element' in training_options['model']:
                                    res_force_normalized = residual_network(
                                        torch.cat((q, v), dim=0).to(device)
                                    )[0]
                                    res_force = training_set.denormalize(f=res_force_normalized.cpu(), normalization_params=(None, None, None, None, f_mean, f_std))[0]
                                else:
                                    (
                                        q_res_normalized,
                                        v_res_normalized,
                                    ) = training_set.normalize(q=q - q_init, v=v)
                                    res_force_normalized = residual_network(
                                        torch.cat(
                                            (q_res_normalized, v_res_normalized),
                                            dim=0,
                                        ).expand(1, -1).to(device)
                                    )[0]
                                    res_force = training_set.denormalize(f=res_force_normalized.cpu())[0]
                            else:
                                if 'element' in training_options['model']:
                                    res_force_normalized = residual_network(
                                        torch.cat((q, v), dim=0).to(device)
                                    )[0]
                                else:
                                    res_force_normalized = residual_network(
                                        torch.cat((q - q_init, v), dim=0).expand(1, -1).to(device)
                                    )[0]
                                # res_force = training_set.denormalize(f=res_force_normalized.cpu(), normalization_params=(None, None, None, None, f_mean, f_std))[0]
                                res_force = res_force_normalized.cpu()
                            q, v = cantilever.forward(q, v, f_ext=res_force, dt=0.01)
                            data_loss = ((target_trajectory_q[frame_i] - q)**2).sum()
                            loss = data_loss 
                            total_loss.append(loss)
                        except:
                            print("Failed")
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
                        f"{save_folder}/residual_network.pth",
                    )
                np.save(f"{save_folder}/val_loss.npy", np.array(val_loss_history))
            qbar.set_description(
                    f"Epoch {epoch}, Train Loss: {train_loss:.3E}, Val Loss: {val_loss:.3E}"
                )
            qbar.update(1)


if __name__ == "__main__":

    config = {}
    config["seed"] = 42
    config["epochs"] = 1000
    config["batch_size"] = 256
    config["learning_rate"] = 1e-3#5e-6
    config["optimizer"] = "adam"
    config["start_frame"] = 0
    config["end_frame"] = 150
    config["training_set"] = [0, 3, 4, 6, 8, 10, 12, 13, 17]
    config["validate_set"] = [5, 15]

    config["cuda"] = 1
    config["normalize"] = False 
    config["Inialization"] = 1e-3
    config["scale"] = 1e3#e6
    config["data_type"] = "optimized"
    config["weight_decay"] = 1e-5
    config["fit"] = "forces"
    # config["fit"] = "SITL"
    # config["model"] = "element"
    config["model"] = "element"
    config["num_mlp_blocks"] = 5
    config["hidden_size"] = 64
    config["num_hidden_layer"] = 4
    config["actuated"] = True
    config["normalize_inputs"] = False
    config["separated"] = False
    config["conditioned"] = False
    config['stress'] = False

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
    cantilever = CantileverEnv3d(42, 'beam', hex_params)
    q_init = torch.from_numpy(cantilever._q0)

    save_folder = f"training/test_refactor_element_direct"
    os.makedirs(f"{save_folder}", exist_ok=True)

    config["data_folder"] = save_folder.replace("training/", "")
    torch.manual_seed(config["seed"])
    with open(f'{save_folder}/config.yaml', 'w') as f:
        yaml.dump(hex_params, f)
        yaml.dump(config, f)

    train(
        cantilever, save_folder, end_frame=150, cantilever_sim=cantilever
    )