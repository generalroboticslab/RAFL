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
from py_diff_pd.common.hex_mesh import generate_hex_mesh
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable


def generate_fish_tail(nx, ny, nz,
                       base_half_width=1.0,
                       peduncle_fraction=0.5,
                       peduncle_width_ratio=1.0,
                       tail_width_ratio=1.667,
                       curvature_power=1.0,
                       base_thickness=1.0,
                       thickness_ratio=0.667):
    """
    Generate a 3D voxel array of a fish body segment that:
      - has a straight vertical base at x = 0
      - curves and narrows toward +x
      - ends in a straight rectangular peduncle

    Axes:
      axis 0 -> x (0 at body, increasing toward tail)
      axis 1 -> y (lateral, symmetric about 0)
      axis 2 -> z (thickness, small)

    Parameters
    ----------
    nx, ny, nz : int
        Grid resolution in x, y, z.
    base_half_width : float
        Half-width at the base (x = 0) in normalized y-coordinates.
    peduncle_fraction : float in (0,1)
        Fraction of length occupied by the peduncle at the tail end.
        (e.g. 0.2 means last 20% of x is a straight rectangle.)
    peduncle_width_ratio : float in (0,1)
        Half-width of the peduncle as a fraction of base_half_width.
    curvature_power : float
        Controls how the body width tapers from base to peduncle.
    thickness_ratio : float in (0,1]
        Fraction of z-slices that are filled.

    Returns
    -------
    voxels : np.ndarray, shape (nx, ny, nz), dtype=np.uint8
        1 = inside shape, 0 = empty.
    """

    # Normalized coordinates
    x = np.linspace(0.0, 1.0, nx)          # 0 = base, 1 = tail end
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    z = np.linspace(-1.0, 1.0, nz)
    _, Z = np.meshgrid(x, z, indexing='ij')

    # Where the peduncle starts in x
    ped_start = peduncle_fraction
    ped_half_width = base_half_width * peduncle_width_ratio
    tail_width = base_half_width * tail_width_ratio

    # y Width as a function of x (1D)
    wy = np.empty_like(x)

    # z Width as a function of x (1D)
    wz = np.empty_like(x)

    # Body region: 0 <= x < ped_start
    body_mask_1d = x < ped_start
    xb = x[body_mask_1d]
    xp = x[~body_mask_1d]
    t_body = np.linspace(0, 1.0, len(xb))
    t_ped = np.linspace(0, 1.0, nx - len(xb))


    # Smooth taper from base_half_width down to ped_half_width
    wy[body_mask_1d] = base_half_width - (base_half_width - ped_half_width) * t_body

    # Peduncle region: straight increase to tail width
    wy[~body_mask_1d] = ped_half_width + (tail_width -  ped_half_width) * t_ped**curvature_power

    wy = wy / wy.max()

    # Thickness: taper throughout entire length
    wz[body_mask_1d] = thickness_ratio * base_thickness + (1 - thickness_ratio) * base_thickness * (1 - t_body)**curvature_power

    wz[~body_mask_1d] = thickness_ratio * base_thickness

    wz = wz / wz.max()

    # Broadcast width to 2D
    Wy = wy[:, None]  # shape (nx, 1)

    # 2D occupancy in xy plane
    mask2d = np.abs(Y) <= Wy

    # Thickness z
    Wz = wz[:, None]  # shape (nx, 1)
    maskz = np.abs(Z) <= Wz

    # print(np.broadcast_to(mask2d[:,:,None], (nx,ny,nz)))
    # print(np.broadcast_to(maskz[:,None,:], (nx,ny,nz)))
    # Combine two masks
    voxels = np.broadcast_to(mask2d[:,:,None], (nx,ny,nz)) & np.broadcast_to(maskz[:,None,:], (nx,ny,nz))     # (nx, ny, nz)
    voxels = voxels.astype(np.uint8)

    # voxels[:, :, maskz[:,1]] = mask2d[..., None].astype(np.uint8)
    # exit()
    return voxels

class FishTailEnv3d (EnvBase):
    # Refinement is an integer controlling the resolution of the mesh.
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        self.folder = folder
        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

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
        density = options["density"] if "density" in options else 5e3
        twist_angle = options["twist_angle"] if "twist_angle" in options else 0

        ### Material Parameters
        la = (
            youngs_modulus
            * poissons_ratio
            / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        )
        mu = youngs_modulus / (2 * (1 + poissons_ratio))

        ### Mesh Parameters
        # Cantilever is 0.12m long, 0.03m wide, and 0.03m tall
        dx = .01
        cell_nums = (int(round(0.10 / dx)), int(round(0.08 / dx)), int(round(0.03 / dx)))
        assert cell_nums[0] * dx == 0.10 and cell_nums[1] * dx == 0.08 and cell_nums[2] * dx == 0.03, "Refinement does not properly divide the cantilever dimensions!" 
        origin = ndarray([0.0, 0.0, 0.0])

        bin_file_name = folder + "mesh.bin"
        bin_file_name = Path(bin_file_name)
        voxels = generate_fish_tail(nx=cell_nums[0], ny=cell_nums[1], nz=cell_nums[2])
        generate_hex_mesh(voxels, dx, origin, bin_file_name)

        mesh = HexMesh3d()
        mesh.Initialize(str(bin_file_name))
        deformable = HexDeformable()
        deformable.Initialize(str(bin_file_name), density, "none", youngs_modulus, poissons_ratio)
        # os.remove(bin_file_name)

        vert_num = mesh.NumOfVertices()
        verts = ndarray([ndarray(mesh.py_vertex(i)) for i in range(vert_num)])

        min_corner = np.min(verts, axis=0)
        max_corner = np.max(verts, axis=0)
        self._obj_center = (max_corner - min_corner) / 2


        ### Boundary Conditions
         ### Boundary Conditions
        self.force_nodes = []
        self.boundary_indices = []
        for i in range(vert_num):
            vx, vy, vz = verts[i]
            if abs(vx - min_corner[0]) < dx:
                self.boundary_indices.append(3*i)
                deformable.SetDirichletBoundaryCondition(3 * i, vx)
                deformable.SetDirichletBoundaryCondition(3 * i + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)

            # Forces are applied on the right edge of the cantilever
            elif abs(vx - max_corner[0]) < dx:
                self.force_nodes.append(i)

        # State-based forces.
        deformable.AddStateForce("gravity", state_force_parameters)
        # Elasticity.
        deformable.AddPdEnergy("corotated", [2 * mu, ], [])
        deformable.AddPdEnergy("volume", [la,], [])

        ### Twist and Rotate
        q0 = ndarray(mesh.py_vertices())

        node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
        max_theta = twist_angle

        x_min, x_max = q0[2::3].min(), q0[2::3].max()
        x_range = x_max - x_min

        print(len(q0)//3)
        
        for idx in range(len(q0)//3):

            v = ndarray(mesh.py_vertex(idx))
            theta = max_theta * (v[0] - x_min) / x_range

            c, s = np.cos(theta), np.sin(theta)
            R = ndarray([[1, 0, 0], [0, c, -s], [0, s, c]])
            center = (
                ndarray([v[0], cell_nums[1] / 2 * dx, cell_nums[2] / 2 * dx]) + origin
            )
            q0[3 * idx : 3 * idx + 3] = R @ (v - center) + center

        self.twisted_q0 = q0

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

        # Reload _q0, _v0 for env base to run simulation for twisting beam
        self._q0 = self.q0.clone().detach().numpy()
        self._v0 = self.v0.clone().detach().numpy()
    
    def is_dirichlet_dof(self, i):
        return i in self.boundary_indices


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
            "camera_pos": (0.5, -1, 0.5),  # Position of camera
            "camera_lookat": (0, 0, 0.2),  # Position that camera looks at
        }
        renderer = PbrtRenderer(options)
        transforms = [("s", 2.4), ("t", [0.0, -0.2, 0.25])]

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
