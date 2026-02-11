import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import time
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from _utils import CantileverDataset
from env_cantilever import CantileverEnv3d
from residual_physics.residual_physics import ResidualPhysicsBase
from residual_physics.network import ResMLPResidual2, MLPResidual
from residual_physics.element_force_update import ElementResidual as UpdatedElementResidual
from residual_physics.element_force import ElementResidual as OldElementResidual
from py_diff_pd.common.common import ndarray
from py_diff_pd.common.hex_mesh import get_boundary_face

class CantileverResidualPhysics(ResidualPhysicsBase):
    def __init__(self, config, folder, options):
        super().__init__(config)
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.diffpd_model = CantileverEnv3d(config['seed'], folder, options)

    def train(self, data_path, config):
        diffpd_model = self.diffpd_model


        self.boundary_indices = []
        for i in range(0,diffpd_model._dofs,3):
            if diffpd_model.is_dirichlet_dof(i):
                self.boundary_indices.extend([i,i+1,i+2])

        if config["model"] == "skip_connection":
            self.model_type = 'full'
            self.residual_network = ResMLPResidual2(diffpd_model._dofs * 2, diffpd_model._dofs, hidden_size=config['hidden_size'], num_mlp_blocks=config['num_mlp_blocks'], num_block_layer=config['num_hidden_layer'])
        elif config["model"] == "MLP":
            self.model_type = 'full'
            self.residual_network = MLPResidual(diffpd_model._dofs*2, diffpd_model._dofs)
        elif config["model"] == "element":
            self.model_type = 'element'
            youngs_modulus = 215856
            poissons_ratio = 0.45
            la = (
                youngs_modulus
                * poissons_ratio
                / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
            )
            m = youngs_modulus / (2 * (1 + poissons_ratio))

            mesh = diffpd_model._deformable.mesh()

            elements = []
            mu = []
            lam = []
            rho = []
            num_elements = mesh.NumOfElements()
            for e in range(num_elements):
                elements.append(mesh.py_element(e))
                mu.append(m)
                lam.append(la)
                rho.append(diffpd_model._deformable.density())
            
            surface_faces = get_boundary_face(mesh)
            elements = ndarray(elements)
            mu = ndarray(mu)
            lam = ndarray(lam)
            rho = ndarray(rho)

            self.scaling = len(elements) * 8

            self.residual_network = UpdatedElementResidual(diffpd_model._dofs, 
                                                    torch.tensor(elements), 
                                                    torch.tensor(surface_faces),
                                                    torch.tensor(mu), 
                                                    torch.tensor(lam), 
                                                    torch.tensor(rho), 
                                                    diffpd_model._q0, 
                                                    0.01,
                                                    hidden_size=config['hidden_size'],
                                                    num_hidden_layer=config['num_hidden_layer'],
                                                    actuated=config['actuated'],
                                                    normalize_inputs=config['normalize_inputs'] if 'normalize_inputs' in config else True,
                                                    separated=config['separated'] if 'separated' in config else True,
                                                    conditioned=config['conditioned'] if 'conditioned' in config else True,
                                                    )
        elif config["model"] == "element_old":
            self.model_type = 'element'
            youngs_modulus = 215856
            poissons_ratio = 0.45
            la = (
                youngs_modulus
                * poissons_ratio
                / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
            )
            m = youngs_modulus / (2 * (1 + poissons_ratio))

            mesh = diffpd_model._deformable.mesh()

            elements = []
            mu = []
            lam = []
            rho = []
            num_elements = mesh.NumOfElements()
            for e in range(num_elements):
                elements.append(mesh.py_element(e))
                mu.append(m)
                lam.append(la)
                rho.append(diffpd_model._deformable.density())
            
            elements = ndarray(elements)
            mu = ndarray(mu)
            lam = ndarray(lam)
            rho = ndarray(rho)

            self.residual_network = OldElementResidual(diffpd_model._dofs, 
                                                    torch.tensor(elements), 
                                                    torch.tensor(mu), 
                                                    torch.tensor(lam), 
                                                    torch.tensor(rho), 
                                                    diffpd_model._q0, 
                                                    0.01,
                                                    hidden_size=config['hidden_size'],
                                                    num_hidden_layer=config['num_hidden_layer'],
                                                    actuated=config['actuated'],
                                                    )
        # Initialize dataset
        training_set_index = config["training_set"]
        assert config["fit"] in ["SITL", "forces"]
        training_set = CantileverDataset(
            training_set_index,
            diffpd_model.q0,
            data_path,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
        )
        training_loader = DataLoader(
            training_set, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        
        if self.validation:
            validation_set_index = config["validate_set"]
            self.validation_set_index = validation_set_index
            validation_set = CantileverDataset(
                validation_set_index,
                diffpd_model.q0,
                data_path,
                start_frame=self.start_frame,
                end_frame=self.end_frame,
            )
            batch_size = 1 if self.validate_physics else config["batch_size"]
            validation_loader = DataLoader(
                validation_set, batch_size=batch_size, shuffle=False
            )
        else:
            validation_set = None
            validation_loader = None

        # Load model
        self.residual_network.to(self.device)
        self.residual_network.train()
        if self.transfer_learning_model is not None:
            self.load_model()
        assert self.optimizer in ["adam"]
        self.initialize_optimizer(weight_decay=config["weight_decay"])
        # Fit model
        assert config["fit"] in ["SITL", "forces"]
        self.fit_residual_physics(
                training_set,
                training_loader,
                validation_set,
                validation_loader,
            )
    
    def fit_residual_physics(
        self,
        training_set,
        training_loader,
        validation_set,
        validation_loader,
    ):
        #f_mean, f_std = torch.mean(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten(), torch.std(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
        f_mean, f_std = torch.zeros(training_set.q_init.shape[0]).flatten(), torch.std(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
        
        device = self.device
        with tqdm(total=self.epochs) as qbar:
            start_time = time.time()
            self.total_loss_history = []
            self.batch_loss_history = []
            self.f_ext_loss = []
            self.validation_loss_history = []
            for epoch in range(self.epochs):
                train_loss = 0
                batch_iter = 0
                self.residual_network.train()
                for (
                    q_start_batch,
                    q_target_batch,
                    v_start_batch,
                    v_target_batch,
                    f_optimized,
                ) in training_loader:
                    batch_size = q_start_batch.shape[0]
                    if self.normalize:
                        if self.model_type == 'element':
                            (
                                f_optimized,
                            ) = training_set.normalize(
                                None,None,
                                f_optimized, (None, None, None, None, f_mean, f_std)
                            )
                            q_start_batch = q_start_batch + training_set.q_init.unsqueeze(0)
                        else:
                            (
                                q_start_batch,
                                v_start_batch,
                                f_optimized,
                            ) = training_set.normalize(
                                q_start_batch,
                                v_start_batch,
                                f_optimized,
                            )
                    else:
                        # (
                        #     f_optimized,
                        # ) = training_set.normalize(
                        #     None,None,
                        #     f_optimized, (None, None, None, None, f_mean, f_std)
                        # )
                        q_start_batch = q_start_batch + training_set.q_init.unsqueeze(0)
                    q_start_batch = q_start_batch.to(device)
                    v_start_batch = v_start_batch.to(device)
                    f_optimized = f_optimized.to(device)
                    batch_iter += 1
                    self.optimizer.zero_grad()
                    residual_forces = self.residual_network(
                        torch.cat(
                            (q_start_batch, v_start_batch), dim=1
                        )
                    )

                    # if epoch > 30:
                    #     print(torch.linalg.norm(residual_forces, dim=-1).mean(), torch.linalg.norm(f_optimized, dim=-1).mean())
                    loss = self.loss_fn(residual_forces, f_optimized) * self.scaling
                    loss.backward()
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.residual_network.parameters(), self.grad_clip
                        )
                    self.batch_loss_history.append(loss.item())
                    self.optimizer.step()
                    train_loss += loss.item() * batch_size
                grad_norm = 0
                for p in self.residual_network.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm() ** 2
                grad_norm = grad_norm**0.5
                train_loss /= len(training_set)
                self.total_loss_history.append(train_loss)
                np.save(
                    f"{self.diffpd_model._folder}/total_loss_history.npy",
                    np.array(self.total_loss_history),
                )
                ####Validation#####
                val_loss = 0
                if self.validation:
                    self.residual_network.eval()
                    val_iter = 0
                    for (
                        q_start_batch,
                        q_target_batch,
                        v_start_batch,
                        v_target_batch,
                        f_optimized,
                    ) in validation_loader:
                        val_iter += 1
                        batch_size = q_start_batch.shape[0]
                        if self.normalize:
                            if self.model_type == 'element':
                                (
                                    f_optimized,
                                ) = training_set.normalize(
                                    None,None,
                                    f_optimized, (None, None, None, None, f_mean, f_std)
                                )
                                q_start_batch = q_start_batch + training_set.q_init.unsqueeze(0)
                            else:
                                (
                                    q_start_batch,
                                    v_start_batch,
                                    f_optimized,
                                ) = training_set.normalize(
                                    q_start_batch,
                                    v_start_batch,
                                    f_optimized,
                                )
                        else:
                            # (
                            #     f_optimized,
                            # ) = training_set.normalize(
                            #     None,None,
                            #     f_optimized, (None, None, None, None, f_mean, f_std)
                            # )
                            q_start_batch = q_start_batch + training_set.q_init.unsqueeze(0)
                        q_start_batch = q_start_batch.to(device)
                        v_start_batch = v_start_batch.to(device)
                        f_optimized = f_optimized.to(device)
                        residual_forces = self.residual_network(
                            torch.cat(
                                (q_start_batch, v_start_batch),
                                dim=1,
                            )
                        )
                        f_ext_loss = self.loss_fn(residual_forces, f_optimized) * self.scaling
                        val_loss += f_ext_loss.item() * batch_size
                    val_loss /= len(validation_set)
                else:
                    val_loss = self.validation_loss_history[-1] if len(self.validation_loss_history) != 0 else 1e10
                self.save_training_history(val_loss, epoch, start_time)
                qbar.set_description(
                    f"Epoch {epoch+1}, Loss: {train_loss:.3E}, Validation Loss: {val_loss:.3E}"
                )
                qbar.update(1)
