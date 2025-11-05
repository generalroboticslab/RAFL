from pathlib import Path
import numpy as np
import torch
import os
import sys

from env_base import EnvBase
from py_diff_pd.common.sim import Sim
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path

from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.hex_mesh import generate_hex_mesh, get_boundary_face
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable

class HangingCantileverEnv3d (EnvBase):
    # Refinement is an integer controlling the resolution of the mesh.
    def __init__(self, seed, folder, options, target_dx=0.01):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)
        self.folder = folder

        refinement = options["refinement"] if "refinement" in options else 2
        youngs_modulus = (
            options["youngs_modulus"] if "youngs_modulus" in options else 1e6
        )
        poissons_ratio = (
            options["poissons_ratio"] if "poissons_ratio" in options else 0.45
        )

        state_force_parameters = (
            options["state_force_parameters"]
            if "state_force_parameters" in options
            else ndarray([0.0, 0.0, -9.81])
        )

        v_rho = (
            options["v_rho"] if "v_rho" in options else 1e3
        )

        v_water = (
            options["v_water"] if "v_water" in options else ndarray([0,0,0])
        )

        Cd_points = ( 
            options["Cd_points"] if "Cd_points" in options 
            else ndarray([[0.0, 0.05], [0.4, 0.05], [0.7, 1.85], [1.0, 2.05]])
        )

        Ct_points = ( 
            options["Ct_points"] if "Ct_points" in options 
            else ndarray([[-1, -0.8], [-0.3, -0.5], [0.3, 0.1], [1, 2.5]])
        )

        max_thrust = (
            options["max_thrust"] if "max_thrust" in options else 0.1
        )

        density = options["density"] if "density" in options else 5e3
        self.amplitude = options["amplitude"] if "amplitude" in options else 0
        self.frequency = options["frequency"] if "frequency" in options else 15

        ### Mesh Parameters
        # Cantilever is 0.03m long, 0.03m wide, and 0.3 m tall
        dx = 0.01
        cell_nums = (3, 3, 30)
        node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
        origin = ndarray([0.0, 0.0, 0.0])

        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
    

        bin_file_name = folder + "/mesh.bin"
        bin_file_name = Path(bin_file_name)
        voxels = np.ones(cell_nums)
        generate_hex_mesh(voxels, dx, origin, bin_file_name)

        mesh = HexMesh3d()
        mesh.Initialize(str(bin_file_name))
        deformable = HexDeformable()
        deformable.Initialize(str(bin_file_name), density, "none", youngs_modulus, poissons_ratio)
        os.remove(bin_file_name)

        ### Boundary Conditions
        self.__node_nums = node_nums
        vert_num = mesh.NumOfVertices()
        verts = ndarray([ndarray(mesh.py_vertex(i)) for i in range(vert_num)])
        
        self.oscillation_offset = lambda t : self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        self.oscillation_vel = lambda t: 2 * np.pi * self.frequency * self.amplitude * np.cos(2 * np.pi * self.frequency * t)
        for i in range(node_nums[0]):
            for j in range(node_nums[1]):
                node_idx = i * node_nums[1] * node_nums[2] + j * node_nums[2] + node_nums[2] - 1
                vx, vy, vz = verts[node_idx]
                deformable.SetDirichletBoundaryCondition(3 * node_idx, vx)
                deformable.SetDirichletBoundaryCondition(3 * node_idx + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * node_idx + 2, vz)

        
        # Elasticity.
                ### Material Parameters
        la = (
            youngs_modulus
            * poissons_ratio
            / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        )
        mu = youngs_modulus / (2 * (1 + poissons_ratio))

        deformable.AddPdEnergy("corotated", [2 * mu, ], [])
        deformable.AddPdEnergy("volume", [la,], [])

        deformable.AddStateForce("gravity", state_force_parameters)

        # # State-based forces.
        # surface_faces = get_boundary_face(mesh)
        # deformable.AddStateForce("hydrodynamics", np.concatenate([[v_rho,], v_water, Cd_points.ravel(), Ct_points.ravel(), [max_thrust,], ndarray(surface_faces).ravel()]))

        ### Simulation Parameters
        self.method = 'pd_eigen'
        self.opt = {'max_pd_iter': 10000, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-7, 'verbose': 0, 'thread_ct': 16, 'use_bfgs': 1, 'bfgs_history_size': 10}
        #create_folder(f"{folder}/{self.method}", exist_ok=False)

        dofs = deformable.dofs()
        self._dofs = dofs
        act_dofs = deformable.act_dofs()
        self.act_dofs = act_dofs

        self._deformable = deformable
        self.sim = Sim(deformable)

        self.q0 = torch.tensor(verts.flatten())
        self.v0 = torch.zeros_like(self.q0)
        self.f_ext = torch.zeros_like(self.q0)

        self._q0 = self.q0.clone().detach().numpy()
        self._v0 = self.v0.clone().detach().numpy()
        self.t = 0
        

    def is_dirichlet_dof(self, dof):

        node_idx = dof // 3
        k = node_idx % self.__node_nums[2]
        return k == self.__node_nums[2] - 1


    def update_boundary(self, t):
        vx_offset =  self.oscillation_offset(t)
        vx_vel = self.oscillation_vel(t)
        for i in range(self.__node_nums[0]):
            for j in range(self.__node_nums[1]):
                node_idx = i * self.__node_nums[1] * self.__node_nums[2] + j * self.__node_nums[2] + self.__node_nums[2] - 1
                vx, vy, vz = self._q0[3 * node_idx : 3 * node_idx + 3] 
                self.sim.deformable.SetDirichletBoundaryCondition(3 * node_idx, vx + vx_offset)
                self.sim.deformable.SetDirichletBoundaryCondition(3 * node_idx + 1, vy)
                self.sim.deformable.SetDirichletBoundaryCondition(3 * node_idx + 2, vz)
        return vx_offset, vx_vel

    def remove_boundary(self,):

        for i in range(self.__node_nums[0]):
            for j in range(self.__node_nums[1]):
                node_idx = i * self.__node_nums[1] * self.__node_nums[2] + j * self.__node_nums[2] + self.__node_nums[2] - 1
                vx, vy, vz = self._q0[3 * node_idx : 3 * node_idx + 3] 
                self.sim.deformable.RemoveDirichletBoundaryCondition(3 * node_idx)
                self.sim.deformable.RemoveDirichletBoundaryCondition(3 * node_idx + 1)
                self.sim.deformable.RemoveDirichletBoundaryCondition(3 * node_idx + 2)
 
        
    def forward (self, q, v, act=None, f_ext=None, dt=0.01):
        if f_ext is None:
            f_ext = self.f_ext
        if act is None:
            act = torch.zeros(self.act_dofs)

        q, v = self.sim(self._dofs, self.act_dofs, self.method, q, v, act, f_ext, dt, self.opt)

        return q, v


    def display_mesh (self, q, file_name, extra_points=None):
        """
        Allow to get images of the simulation
        """
        options = {
            "parent_dir": self.folder,
            "file_name": file_name,
            "light_map": "uffizi-large.exr",
            "sample": 8,
            "max_depth": 2,
            "camera_pos": (0, -1, 0.4),  # Position of camera
            "camera_lookat": (0, 0, 0.3),  # Position that camera looks at
        }
        renderer = PbrtRenderer(options)
        transforms = [("s", 1), ("t", [-0.0126, -0.2, 0.2])]

        tmp_bin_file_name = f'{self.folder}/.tmp.bin'
        self._deformable.PySaveToMeshFile(ndarray(q), tmp_bin_file_name)

        mesh = HexMesh3d()
        mesh.Initialize(tmp_bin_file_name)
        os.remove(tmp_bin_file_name)

        if extra_points is not None:
            for q_v in extra_points:
                renderer.add_shape_mesh(
                    {"name": "sphere", "center": ndarray((q_v)), "radius": 0.0025},
                    color="ff3025", #"2aaa8a",  # green
                    transforms=transforms,
                )

        renderer.add_hex_mesh(
            mesh, transforms=transforms, render_voxel_edge=True, color="0096c7"
        )
        renderer.add_tri_mesh(
            Path(root_path) / "asset/mesh/curved_ground.obj",
            texture_img="chkbd_24_0.7",
            transforms=[("s", 2)],
        )

        renderer.render()
