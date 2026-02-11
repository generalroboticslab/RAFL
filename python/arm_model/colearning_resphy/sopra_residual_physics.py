import sys

sys.path.append("../..")
sys.path.append("..")
import time
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from arm_model.env_arm import ArmEnv
from arm_model.colearning_resphy._utils import (
    ArmDataset,
)
from residual_physics.network import MLPResidual, ResMLPResidual2
from model import SupervisedLearningForward, PhysicsForward, LearningFoward
from validate_residual_physics import main as val_main
from residual_physics.residual_physics import ResidualPhysicsBase
from residual_physics.element_force_update import ElementResidual
from py_diff_pd.common.common import ndarray
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def ddp_is_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if ddp_is_initialized() else 0

def is_main_process():
    return get_rank() == 0


class SoPrAResidualPhysics(ResidualPhysicsBase):
    def __init__(self, config, save_folder, params):
        super().__init__(config)
        self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.diffpd_model = ArmEnv(config['seed'], save_folder, params)

    def train(self, data_path, config):
        diffpd_model = self.diffpd_model
        assert config["model"] in ["skip_connection", "MLP", "element"]
        assert config["fit"] in ["SITL", "forces"]

        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self.device = f"cuda:{local_rank}" 

        if config["model"] == "skip_connection":
            self.residual_network = ResMLPResidual2(diffpd_model._dofs * 3, diffpd_model._dofs, num_mlp_blocks=config['num_mlp_blocks'],num_block_layer=config['num_block_layer'], hidden_size=config['hidden_size'], act_fn=self.fn_act)
        elif config["model"] == "MLP":
            self.residual_network = MLPResidual(diffpd_model._dofs * 3, diffpd_model._dofs, hidden_sizes=self.hidden_sizes, act_fn=self.fn_act)
        elif config["model"] == "element":

            la = (
                diffpd_model._youngs_modulus
                * diffpd_model._poissons_ratio
                / ((1 + diffpd_model._poissons_ratio) * (1 - 2 * diffpd_model._poissons_ratio))
            )
            m = diffpd_model._youngs_modulus / (2 * (1 + diffpd_model._poissons_ratio))

            #mesh = diffpd_model._deformable.mesh()

            elements = []
            mu = []
            lam = []
            rho = []
            for e in diffpd_model._elements:
                elements.append(e)
                mu.append(m)
                lam.append(la)
                rho.append(diffpd_model._deformable.density())
            
            surface_faces = diffpd_model._get_boundary_ordered()
            elements = ndarray(elements)
            mu = ndarray(mu)
            lam = ndarray(lam)
            rho = ndarray(rho)
            
            inner_faces = np.concatenate(diffpd_model._inner_faces)
            force_nodes = []
            for face in inner_faces:
                force_nodes.extend(face)

            force_nodes = np.array(list(set(force_nodes)))


            self.residual_network = ElementResidual(diffpd_model._dofs, 
                                                    torch.tensor(elements), 
                                                    torch.tensor(surface_faces),
                                                    torch.tensor(mu), 
                                                    torch.tensor(lam), 
                                                    torch.tensor(rho), 
                                                    diffpd_model._q0, 
                                                    None,
                                                    hidden_size=config['hidden_size'],
                                                    num_hidden_layer=config['num_hidden_layer'],
                                                    actuated=config['actuated'],
                                                    force_nodes=torch.tensor(force_nodes)
                                                    )
        
        
        # Initialize dataset
        training_set_index = config["training_set"]
        training_set = ArmDataset(
            training_set_index,
            diffpd_model,
            data_path,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
        )

        num_samples_per_rank = len(training_set) // 8

        train_sampler = DistributedSampler(training_set, shuffle=True, drop_last=True)
        
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            drop_last=True,
            sampler=train_sampler,
            num_workers=7,
            pin_memory=True,
            persistent_workers=True,
        )

        if self.validation:
            validation_set_index = config["validate_set"]
            self.validation_set_index = validation_set_index
            validation_set = ArmDataset(
                validation_set_index,
                diffpd_model,
                data_path,
                start_frame=self.start_frame,
                end_frame=self.end_frame,
            )
            val_sampler = DistributedSampler(validation_set, shuffle=False, drop_last=False)
            validation_loader = DataLoader(
                validation_set,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=val_sampler,   
                num_workers=2,
                pin_memory=True,
                persistent_workers=True,
            )
        else:
            validation_set = None
            validation_loader = None

        
        self.residual_network.to(self.device)

        local_rank = int(os.environ["LOCAL_RANK"])
        self.residual_network = DDP(
            self.residual_network,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True
        )

        self.residual_network.train()

        if is_main_process():
            print(f"Number of parameters: {self.residual_network.module.count_parameters() if ddp_is_initialized() else self.residual_network.count_parameters()}")

        #config["hidden_sizes"] = self.residual_network.hidden_sizes
        # Initialize optimizer
        self.initialize_optimizer(weight_decay=config["weight_decay"])

        
        self.fit_residual_physics(
                training_set,
                training_loader,
                validation_set,
                validation_loader,
            )
        
        if ddp_is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def fit_residual_physics(
        self,
        training_set,
        training_loader,
        validation_set,
        validation_loader,
    ):
        # Initialize auxiliary variables
        #f_mean, f_std = torch.mean(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten(), torch.std(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
        f_mean, f_std = torch.zeros(training_set.q_init.shape[0]).flatten(), torch.std(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
        #print(f_mean[:3], f_std[:3])
        device = self.device
        qbar = tqdm(total=self.epochs) if is_main_process() else None

        start_time = time.time()
        self.total_loss_history = []
        #self.batch_loss_history = []
        self.f_ext_loss = []
        self.validation_loss_history = []
        for epoch in range(self.epochs):
            if ddp_is_initialized() and isinstance(training_loader.sampler, DistributedSampler):
                training_loader.sampler.set_epoch(epoch)

            self.residual_network.train()

            local_sum = 0.0     
            local_count = 0     

            for (
                q_start_batch,
                q_target_batch,
                v_start_batch,
                v_target_batch,
                pressure_forces_batch,
                f_optimized,
            ) in training_loader:

                batch_size = q_start_batch.shape[0]
                if self.normalize:
                    (
                        q_start_batch,
                        v_start_batch,
                        pressure_forces_batch,
                        f_optimized,
                    ) = training_set.normalize(
                        q_start_batch,
                        v_start_batch,
                        pressure_forces_batch,
                        f_optimized,
                    )
                else:
                    # (
                    #     f_optimized,
                    # ) = training_set.normalize(
                    #     None,None,None,
                    #     f_optimized, (None, None, None, None, None, None, f_mean, f_std)
                    # )
                    q_start_batch = q_start_batch + training_set.q_init.unsqueeze(0)
                q_start_batch = q_start_batch.to(device)
                v_start_batch = v_start_batch.to(device)
                pressure_forces_batch = pressure_forces_batch.to(device)
                f_optimized = f_optimized.to(device)
                self.optimizer.zero_grad()
                residual_forces = self.residual_network(
                    torch.cat((q_start_batch, v_start_batch, pressure_forces_batch), dim=1)
                )
                loss = self.loss_fn(residual_forces, f_optimized) * self.scaling   # scalar per-rank

                loss.backward()
                self.optimizer.step()

                local_sum += float(loss.item()) * batch_size
                local_count += batch_size

            # --- reduce across ranks (use tensors; no Python-object broadcast) ---
            t = torch.tensor([local_sum, float(local_count)],
                            device=self.device, dtype=torch.float64)
            if ddp_is_initialized():
                dist.all_reduce(t, op=dist.ReduceOp.SUM)   # sums over all ranks

            global_sum, global_count = t[0].item(), t[1].item()
            train_loss = global_sum / max(1.0, global_count)   # global mean loss

            if is_main_process():
                self.total_loss_history.append(train_loss)
                np.save(f"{self.diffpd_model._folder}/total_loss_history.npy", np.array(self.total_loss_history))
            ####Validation#####
            if self.validation and (not self.validate_physics):
                val_loss = self.validate_residual_forces(epoch, training_set, validation_set, validation_loader)

                if is_main_process():
                    self.save_training_history(val_loss, epoch, start_time)
            if is_main_process():
                qbar.set_description(
                    f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.3E}, Validation Loss: {val_loss:.3E}"
                )
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break
                qbar.update(1)

    def validate_residual_forces(self, epoch, training_set, validation_set, validation_loader):
        device = self.device
        model_ref = self.residual_network.module if ddp_is_initialized() else self.residual_network
        model_ref.eval()

        # precompute stats used in normalize
        # f_mean = torch.mean(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
        # f_std  = torch.std(training_set.fs.view(-1,3),  dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
        f_mean, f_std = torch.zeros(training_set.q_init.shape[0]).flatten(), torch.std(training_set.fs.view(-1,3), dim=0).expand(training_set.q_init.shape[0] // 3, 3).flatten()
        local_sum = 0.0
        local_count = 0

        with torch.no_grad():
            for (q_start_batch, q_target_batch, v_start_batch, v_target_batch,
                pressure_forces_batch, f_optimized) in validation_loader:

                batch_size = q_start_batch.shape[0]
                if self.normalize:
                    (q_start_batch, v_start_batch, pressure_forces_batch, f_optimized) = training_set.normalize(
                        q_start_batch, v_start_batch, pressure_forces_batch, f_optimized
                    )
                else:
                    # (f_optimized,) = training_set.normalize(
                    #     None, None, None, f_optimized,
                    #     (None, None, None, None, None, None, f_mean, f_std)
                    # )
                    q_start_batch = q_start_batch + training_set.q_init.unsqueeze(0)

                q_start_batch = q_start_batch.to(device)
                v_start_batch = v_start_batch.to(device)
                pressure_forces_batch = pressure_forces_batch.to(device)
                f_optimized = f_optimized.to(device)

                pred = model_ref(torch.cat((q_start_batch, v_start_batch, pressure_forces_batch), dim=1))
                loss = self.loss_fn(pred, f_optimized)  * self.scaling

                local_sum += float(loss.item()) * batch_size
                local_count += batch_size

        # Reduce across ranks using CUDA tensors (no Python object broadcast)
        t = torch.tensor([local_sum, float(local_count)], device=device, dtype=torch.float64)
        if ddp_is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        global_sum, global_count = t[0].item(), t[1].item()
        global_mean = global_sum / max(1.0, global_count)

        model_ref.train()  # restore
        return global_mean
        