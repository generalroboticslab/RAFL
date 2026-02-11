import sys
sys.path.append("../")
sys.path.append("../..")
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
from env_cantilever_new import CantileverEnv3d as CantileverNewEnv3d
from env_cantilever_longer_scaled import LongerCantileverEnv3d
from env_cantilever_shorter_scaled import ShorterCantileverEnv3d
from env_cantilever_thicker_scaled import ThickerCantileverEnv3d
from env_cantilever_thinner_scaled import ThinnerCantileverEnv3d
from residual_physics.network import ResMLPResidual2, MLPResidual
from residual_physics.element_force_update import ElementResidual
from py_diff_pd.common.common import ndarray
from tqdm import tqdm
from py_diff_pd.common.hex_mesh import get_boundary_face

def train(
    hex_params, save_folder, start_frame=0, end_frame=150, num_epochs=100,
):

    training_options = yaml.safe_load(open(f"{save_folder}/config.yaml"))
    with open(f"{save_folder}/config.yaml", 'w') as f:
        yaml.dump(training_options, f)
   
    cantilever = CantileverEnv3d(42, 'beam', hex_params)

    dofs = cantilever._dofs
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
                                        .01,
                                        hidden_size=training_options['hidden_size'],
                                        num_hidden_layer=training_options['num_hidden_layer'],
                                        actuated=training_options['actuated'],
                                        normalize_inputs=training_options['normalize_inputs'],
                                        multi_shape="all" in save_folder
                                        )
        

    model_input = f"residual_network"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residual_network.to(device)

    residual_network.train()

    # training_set = CantileverDataset(
    # training_options["training_set"],
    # cantilever._q0,
    # f"cantilever_data_longer_scaled_straight",
    # start_frame=training_options["start_frame"],
    # end_frame=training_options["end_frame"],
    # )

    cantilever_base = CantileverEnv3d(42, 'beam', hex_params)
    cantilever_new = CantileverNewEnv3d(42, 'beam_new', hex_params)
    cantilever_longer = LongerCantileverEnv3d(42, 'beam_longer', hex_params)
    cantilever_shorter = ShorterCantileverEnv3d(42, 'beam_shorter', hex_params)
    cantilever_thicker = ThickerCantileverEnv3d(42, 'beam_thicker', hex_params)
    cantilever_thinner = ThinnerCantileverEnv3d(42, 'beam_thinner', hex_params)


    trainable_params = [p for p in residual_network.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
            trainable_params,
            lr=training_options["learning_rate"],
            weight_decay=training_options["weight_decay"],
        )


    normalize = training_options["normalize"]
    # f_mean, f_std = torch.zeros(training_set.q_init.shape[0]).flatten(), torch.std(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
    
    R,t = None, None

    shapes = np.array(training_options['shape_list'])

    rng = np.random.default_rng(42)

    # shape_loss_scale = {shape : 1 for shape in shapes}

    R_shapes = {}
    t_shapes = {}
    base_loss_shape = {shape: 0 for shape in shapes}

    for shape in shapes:

        if shape == '':
            cantilever = cantilever_base
            dx = 0.01
            shape_fullname = ''
        elif shape == '_new':
            cantilever = cantilever_new
            dx = 0.01
            shape_fullname = '_new'
        elif shape == '_longer':
            cantilever = cantilever_longer
            dx = [0.012, 0.01, 0.01]
            shape_fullname = '_longer_scaled'
        elif shape == '_shorter':
            cantilever = cantilever_shorter
            dx = [0.008,0.01,0.01]
            shape_fullname = '_shorter_scaled'
        elif shape == '_thicker':
            cantilever = cantilever_thicker
            dx = [0.01,0.01333333,0.01333333]
            shape_fullname = '_thicker_scaled'
        elif shape == '_thinner':
            cantilever = cantilever_thinner
            dx = [0.01, 0.0066667, 0.0066667]
            shape_fullname = '_thinner_scaled'

        for data_idx in training_options["validate_set"]:

            q_init = torch.from_numpy(cantilever._q0)
            
            qs_real = np.load(f'data{shape}/q{data_idx}.npy' )

            if data_idx == training_options['validate_set'][0]:
                initial_real_marker = qs_real[0,:,:]*1e-3
                R, t = cantilever.fit_realframe(initial_real_marker)
                R_shapes[shape] = R 
                t_shapes[shape] = t
                real_markers_init = initial_real_marker @ R.T + t
                cantilever.interpolate_markers_3d(cantilever._q0.reshape(-1,3), real_markers_init)

            real_markers_old = qs_real[1:,:,:] * 1e-3
            real_markers = np.zeros((real_markers_old.shape[0],real_markers_old.shape[1],real_markers_old.shape[2]),dtype=np.float64)
            for i in range(real_markers.shape[0]):
                real_markers[i] = real_markers_old[i,:,:] @ R.T + t

            target_data = torch.from_numpy(real_markers)

            q_arr = np.load(f"cantilever_data{shape_fullname}_straight/q_force_opt{data_idx}_reorder.npz")['arr_%d' % (len(np.load(f"cantilever_data{shape_fullname}_straight/q_force_opt{data_idx}_reorder.npz"))-1)]
            q = torch.from_numpy(q_arr)
            v = torch.zeros(cantilever.dofs, dtype=torch.float64)

            total_loss = []
            for frame_i in range(1, end_frame):

                # q = target_trajectory_q[frame_i - 1].detach().clone()
                # v = target_trajectory_v[frame_i - 1].detach().clone()
                
                q, v = cantilever.forward(q, v, f_ext=torch.zeros(dofs, dtype=torch.float64), dt=0.01)
                qx = q.reshape(-1,3)

                qx_marker = cantilever.get_markers_3d(qx)
                data_loss = ((-qx_marker.flatten() + target_data[frame_i].flatten())**2).sum()
                loss = data_loss 
                total_loss.append(loss)


            total_loss = torch.stack(total_loss).mean() 
            base_loss_shape[shape] += total_loss.item() 
    
    for shape in shapes:
        base_loss_shape[shape] /= len(training_options['validate_set'])
    
    total_loss_history = []
    val_loss_history = []
    with tqdm(num_epochs) as qbar:
        for epoch in range(num_epochs):
            train_loss = 0
            val_loss = 0
            
            shape_val_loss = {shape: 0 for shape in shapes}

            rng.shuffle(shapes)
            print(shapes)
            # print(shape_loss_scale)

            for data_idx in training_options["training_set"]:

                for shape in shapes:
                
                    if shape == '':
                        cantilever = cantilever_base
                        dx = 0.01
                        shape_fullname = ''
                    elif shape == '_new':
                        cantilever = cantilever_new
                        dx = 0.01
                        shape_fullname = '_new'
                    elif shape == '_longer':
                        cantilever = cantilever_longer
                        dx = [0.012, 0.01, 0.01]
                        shape_fullname = '_longer_scaled'
                    elif shape == '_shorter':
                        cantilever = cantilever_shorter
                        dx = [0.008,0.01,0.01]
                        shape_fullname = '_shorter_scaled'
                    elif shape == '_thicker':
                        cantilever = cantilever_thicker
                        dx = [0.01,0.01333333,0.01333333]
                        shape_fullname = '_thicker_scaled'
                    elif shape == '_thinner':
                        cantilever = cantilever_thinner
                        dx = [0.01, 0.0066667, 0.0066667]
                        shape_fullname = '_thinner_scaled'

                    q_init = torch.from_numpy(cantilever._q0)
                    
                    mesh = cantilever._deformable.mesh()

                    elements = []
                    num_elements = mesh.NumOfElements()
                    for e in range(num_elements):
                        elements.append(mesh.py_element(e))

                    surface_faces = get_boundary_face(mesh)
                    elements = ndarray(elements)

                    residual_network.reset_mesh(cantilever._q0, torch.tensor(elements), torch.tensor(surface_faces), dx)
                    qs_real = np.load(f'data{shape}/q{data_idx}.npy' )

                    # initial_real_marker = qs_real[0,:,:]*1e-3
                    R, t = R_shapes[shape], t_shapes[shape]
                    # real_markers_init = initial_real_marker @ R.T + t
                    # cantilever.interpolate_markers_3d(cantilever._q0.reshape(-1,3), real_markers_init)
                    
                    real_markers_old = qs_real[1:,:,:] * 1e-3
                    real_markers = np.zeros((real_markers_old.shape[0],real_markers_old.shape[1],real_markers_old.shape[2]),dtype=np.float64)
                    for i in range(real_markers.shape[0]):
                        real_markers[i] = real_markers_old[i,:,:] @ R.T + t

                    target_data = torch.tensor(real_markers, dtype=torch.float64)

                    q_arr = np.load(f"cantilever_data{shape_fullname}_straight/q_force_opt{data_idx}_reorder.npz")['arr_%d' % (len(np.load(f"cantilever_data{shape_fullname}_straight/q_force_opt{data_idx}_reorder.npz"))-1)]
                    q = torch.from_numpy(q_arr)
                    v = torch.zeros(cantilever.dofs, dtype=torch.float64)

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
                                res_force_normalized = residual_network(
                                    torch.cat((q, v), dim=0).to(device)
                                )[0]
                                # res_force = training_set.denormalize(f=res_force_normalized.cpu(), normalization_params=(None, None, None, None, f_mean, f_std))[0]
                                res_force = res_force_normalized.cpu()

                            q, v = cantilever.forward(q, v, f_ext=res_force, dt=0.01)

                            qx = q.reshape(-1,3)

                            qx_marker = cantilever.get_markers_3d(qx)

                            data_loss = ((-qx_marker.flatten() + target_data[frame_i].flatten())**2).sum()
                            loss = data_loss 
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

            with torch.no_grad():
                for data_idx in training_options["validate_set"]:

                    for shape in shapes:
                
                        if shape == '':
                            cantilever = cantilever_base
                            dx = 0.01
                            shape_fullname = ''
                        elif shape == '_new':
                            cantilever = cantilever_new
                            dx = 0.01
                            shape_fullname = '_new'
                        elif shape == '_longer':
                            cantilever = cantilever_longer
                            dx = [0.012, 0.01, 0.01]
                            shape_fullname = '_longer_scaled'
                        elif shape == '_shorter':
                            cantilever = cantilever_shorter
                            dx = [0.008,0.01,0.01]
                            shape_fullname = '_shorter_scaled'
                        elif shape == '_thicker':
                            cantilever = cantilever_thicker
                            dx = [0.01,0.01333333,0.01333333]
                            shape_fullname = '_thicker_scaled'
                        elif shape == '_thinner':
                            cantilever = cantilever_thinner
                            dx = [0.01, 0.0066667, 0.0066667]
                            shape_fullname = '_thinner_scaled'

                        q_init = torch.from_numpy(cantilever._q0)
                        
                        mesh = cantilever._deformable.mesh()

                        elements = []
                        num_elements = mesh.NumOfElements()
                        for e in range(num_elements):
                            elements.append(mesh.py_element(e))

                        surface_faces = get_boundary_face(mesh)
                        elements = ndarray(elements)

                        residual_network.reset_mesh(cantilever._q0, torch.tensor(elements), torch.tensor(surface_faces), dx)
                        qs_real = np.load(f'data{shape}/q{data_idx}.npy' )

                        # initial_real_marker = qs_real[0,:,:]*1e-3
                        R, t = R_shapes[shape], t_shapes[shape]
                        # real_markers_init = initial_real_marker @ R.T + t
                        # cantilever.interpolate_markers_3d(cantilever._q0.reshape(-1,3), real_markers_init)

                        real_markers_old = qs_real[1:,:,:] * 1e-3
                        real_markers = np.zeros((real_markers_old.shape[0],real_markers_old.shape[1],real_markers_old.shape[2]),dtype=np.float64)
                        for i in range(real_markers.shape[0]):
                            real_markers[i] = real_markers_old[i,:,:] @ R.T + t

                        target_data = torch.from_numpy(real_markers)

                        q_arr = np.load(f"cantilever_data{shape_fullname}_straight/q_force_opt{data_idx}_reorder.npz")['arr_%d' % (len(np.load(f"cantilever_data{shape_fullname}_straight/q_force_opt{data_idx}_reorder.npz"))-1)]
                        q = torch.from_numpy(q_arr)
                        v = torch.zeros(cantilever.dofs, dtype=torch.float64)

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
                                    res_force_normalized = residual_network(
                                        torch.cat((q, v), dim=0).to(device)
                                    )[0]
                                    # res_force = training_set.denormalize(f=res_force_normalized.cpu(), normalization_params=(None, None, None, None, f_mean, f_std))[0]
                                    res_force = res_force_normalized.cpu()
                                q, v = cantilever.forward(q, v, f_ext=res_force, dt=0.01)
                                qx = q.reshape(-1,3)

                                qx_marker = cantilever.get_markers_3d(qx)
                                data_loss = ((-qx_marker.flatten() + target_data[frame_i].flatten())**2).sum()
                                loss = data_loss 
                                total_loss.append(loss)
                            except Exception as e:
                                # Get the exception type name and message
                                exception_type = type(e).__name__
                                exception_message = str(e)
                                print(f"An error of type '{exception_type}' occurred: {exception_message}")
                                # Output: An error of type 'NameError' occurred: name 'x' is not defined
                                print("Failed")
                                break

                        total_loss = torch.stack(total_loss).mean() 
                        val_loss += total_loss.item() 
                        shape_val_loss[shape] += total_loss.item()

            train_loss /= len(shapes) * len(training_options["training_set"])
            total_loss_history.append(train_loss)

            val_loss /= len(shapes) * len(training_options["validate_set"])

            val_loss_improvement = 0
            for shape in shapes:
                shape_val_loss[shape] /= len(training_options["validate_set"])
                val_loss_improvement += shape_val_loss[shape] / base_loss_shape[shape]

            val_loss_improvement /= len(shapes)

            val_loss_history.append(val_loss_improvement)
            
            if val_loss_improvement == np.min(val_loss_history):
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
                        f"Epoch {epoch}, Train Loss: {train_loss:.3E}, Val Loss: {val_loss:.3E}, Val Loss Improvement: {val_loss_improvement:.3E}"
                    )
            qbar.update(1)


if __name__ == "__main__":
    poissons_ratio = 0.45
    youngs_modulus = 215856
    density = 1.07e3
    state_force = [0, 0, -9.80709]
    params = {
        "density": density,
        "youngs_modulus": youngs_modulus,
        "poissons_ratio": poissons_ratio,
        "state_force_parameters": state_force,
        "mesh_type": "tet",
        "refinement": 1,
    }
    
    config = {}
    config["seed"] = 42
    config["epochs"] = 1000 
    config["batch_size"] = 256
    config["learning_rate"] = 1e-3
    config["optimizer"] = "adam"
    config["start_frame"] = 0
    config["end_frame"] = 140
    config["training_set"] = [0, 3, 4, 6, 8, 10, 12, 13, 17]
    config["validate_set"] = [5, 15]

    config["cuda"] = 5
    config["normalize"] = False
    config["Inialization"] = 1e-3
    config["scale"] = 1e3
    config["data_type"] = "optimized"
    config["weight_decay"] = 1e-5
    config["fit"] = "forces"
    config["model"] = "element"
    config["hidden_size"] = 64
    config["actuated"] = True
    config["num_hidden_layer"] = 4
    config["num_mlp_blocks"] = 5
    config["normalize_inputs"] = False
    # config['shape_list'] = ['', '_longer', '_shorter', '_thicker', '_thinner']
    # config['shape_list'] = ['_longer', '_shorter', '_thicker', '_thinner']
    # config['shape_list'] = ['_thicker', '_thinner']
    # config['shape_list'] = ['_longer', '_shorter']
    config['shape_list'] = ['_thicker', '_longer']
    # config['shape_list'] = ['_thinner', '_shorter']
    # config['shape_list'] = ['_new', '_thicker', '_thinner']
    # config['shape_list'] = ['_new', '_longer', '_shorter']
    # config['shape_list'] = ['_new', '_thinner', '_shorter']
    # config['shape_list'] = ['_new', '_thicker', '_longer']

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

    # save_folder = f"training/test_refactor_element_zero_transformer_all_plus_mixed_direct"
    # save_folder = f"training/test_refactor_element_zero_transformer_all_mixed_direct"
    # save_folder = f"training/test_refactor_element_zero_transformer_thickness_mixed_direct"
    # save_folder = f"training/test_refactor_element_zero_transformer_length_mixed_direct"
    # save_folder = f"training/test_refactor_element_zero_transformer_ratio_mixed_direct"
    save_folder = f"training/test_refactor_element_zero_transformer_all_ratioInverse_mixed_direct"
    # save_folder = f"training/test_refactor_element_zero_transformer_all_thickness_new_mixed_direct"
    # save_folder = f"training/test_refactor_element_zero_transformer_all_length_new_mixed_direct"
    # save_folder = f"training/test_refactor_element_zero_transformer_all_ratio_new_mixed_direct"
    # save_folder = f"training/test_refactor_element_zero_transformer_ratioInverse_new_mixed_direct"
    os.makedirs(f"{save_folder}", exist_ok=True)
    
    config["data_folder"] = save_folder.replace("training/", "")
    torch.manual_seed(config["seed"])
    with open(f'{save_folder}/config.yaml', 'w') as f:
        yaml.dump(hex_params, f)
        yaml.dump(config, f)
    
    train(
    hex_params, save_folder, end_frame=140, 
    )
