### Code Structure Sim-to-real beam model

    .
    ├── beam_sys_all.py # Run system identification for canonical beam (optimize Young's modulus and Poisson's ratio) based on the same training set as residual physics frameworks.
    ├── beam_sys_all_longer.py # Run system identification for longer beam (non-scaled) (optimize Young's modulus and Poisson's ratio) based on the same training set as residual physics frameworks.
    ├── beam_sys_all_longer_scaled.py # Run system identification for longer beam (scaled) (optimize Young's modulus and Poisson's ratio) based on the same training set as residual physics frameworks.
    ├── beam_sys_all_shorter.py # Run system identification for shorter beam (non-scaled) (optimize Young's modulus and Poisson's ratio) based on the same training set as residual physics frameworks.
    ├── beam_sys_all_shorter_scaled.py # Run system identification for shorter beam (scaled) (optimize Young's modulus and Poisson's ratio) based on the same training set as residual physics frameworks.
    ├── beam_sys_all_thicker.py # Run system identification for thicker beam (non-scaled) (optimize Young's modulus and Poisson's ratio) based on the same training set as residual physics frameworks.
    ├── beam_sys_all_thicker_scaled.py # Run system identification for thicker beam (scaled) (optimize Young's modulus and Poisson's ratio) based on the same training set as residual physics frameworks.
    ├── beam_sys_all_thinner.py # Run system identification for thinner beam (non-scaled) (optimize Young's modulus and Poisson's ratio) based on the same training set as residual physics frameworks.
    ├── beam_sys_all_thinner_scaled.py # Run system identification for thinner beam (scaled) (optimize Young's modulus and Poisson's ratio) based on the same training set as residual physics frameworks.
    ├── env_base.py # Base class of DiffPD model, modifying for damping optimization. 
    ├── env_cantilever.py # Canonical Beam DiffPD class.
    ├── env_cantilever_longer.py # Longer Beam (non-scaled) DiffPD class.
    ├── env_cantilever_longer_scaled.py # Longer Beam (scaled) DiffPD class.
    ├── env_cantilever_shorter.py # Shorter Beam (non-scaled) DiffPD class.
    ├── env_cantilever_shorter_scaled.py # Shorter Beam (scaled) DiffPD class.
    ├── env_cantilever_thicker.py # Thicker Beam (non-scaled) DiffPD class.
    ├── env_cantilever_thicker_scaled.py # Thicker Beam (scaled) DiffPD class.
    ├── env_cantilever_thinner.py # Thinner Beam (non-scaled) DiffPD class.
    ├── env_cantilever_thinner_scaled.py # Thinner Beam (scaled) DiffPD class.
    ├── model.py # Neural netowrk.
    ├── init_beam.py # Optimizing virtual forces to match the initial state. (used exclusively for canonical beam)
    ├── optimize_trajectory.py # Optimize the full state model step by step (used exclusively for canonical beam)
    ├── generate_augmented_data.py # Generate augmented data including target residual forces for canonical beam
    ├── generate_data_longer.py # Generate initial states for longer beam (non-scaled)
    ├── generate_data_longer_scaled.py # Generate initial states for longer beam (scaled)
    ├── generate_data_shorter.py # Generate initial states for shorter beam (non-scaled)
    ├── generate_data_shorter_scaled.py # Generate initial states for shorter beam (scaled)
    ├── generate_data_thicker.py # Generate initial states for thicker beam (non-scaled)
    ├── generate_data_thicker_scaled.py # Generate initial states for thicker beam (scaled)
    ├── generate_data_thinner.py # Generate initial states for thinner beam (non-scaled)
    ├── generate_data_thinner_scaled.py # Generate initial states for thinner beam (scaled)
    ├── beam_residual_physics.py # Supervised Learning Residual physics training framework for canonical beam
    ├── test_residual_physics.py # Test residual physics framework for canonical beam
    ├── test_residual_physics_longer.py # Test residual physics framework for longer beam (non-scaled)
    ├── test_residual_physics_longer_scaled.py # Test residual physics framework for longer beam (scaled)
    ├── test_residual_physics_shorter.py # Test residual physics framework for shorter beam (non-scaled)
    ├── test_residual_physics_shorter_scaled.py # Test residual physics framework for shorter beam (scaled)
    ├── test_residual_physics_thicker.py # Test residual physics framework for thicker beam (non-scaled)
    ├── test_residual_physics_thicker_scaled.py # Test residual physics framework for thicker beam (scaled)
    ├── test_residual_physics_thinner.py # Test residual physics framework for thinner beam (non-scaled)
    ├── test_residual_physics_thinner_scaled.py # Test residual physics framework for thinner beam (scaled)
    ├── training.py # Supervised Learning Residual Physics Training script for canonical beam
    ├── beam_residual_physics.py # Called by training.py for supervised learning
    ├── training_direct.py # RAFL Training script for canonical beam
    ├── training_direct_longer.py # RAFL Training script for longer beam (non-scaled)
    ├── training_direct_longer_scaled.py # RAFL Training script for longer beam (scaled)
    ├── training_direct_shorter.py # RAFL Training script for shorter beam (non-scaled)
    ├── training_direct_shorter_scaled.py # RAFL Training script for shorter beam (scaled)
    ├── training_direct_thicker.py # RAFL Training script for thicker beam (non-scaled)
    ├── training_direct_thicker_scaled.py # RAFL Training script for thicker beam (scaled)
    ├── training_direct_thinner.py # RAFL Training script for thinner beam (non-scaled)
    ├── training_direct_thinner_scaled.py # RAFL Training script for thinner beam (scaled)
    └── README.md


You can download Sim2real beam real data from [Google Drive](https://drive.google.com/drive/folders/16v8ItWT0kAm8PBaxGByvA-moDE78VFxq?usp=sharing)
, where `data_new`, `data_longer`, `data_shorter`, `data_thicker`, `data_thinner` store the collected raw data and `cantilver_data_new_straight`, `cantilver_data_longer_straight`, `cantilver_data_longer_scaled_straight`, `cantilver_data_shorter_straight`, `cantilver_data_shorter_scaled_straight`, `cantilver_data_thicker_straight`, `cantilver_data_thicker_scaled_straight`, `cantilver_data_thinner_straight`, `cantilver_data_thinner_scaled_straight` store the optimized trajectories with optimized initial states and target forces where applicable.

To run the complete residual physics framework, we first need to build augmented dataset by run `build_augmented_data.py`. Then we can run `training.py` to train the residual physics network. We save the best model performed on validation set as `residual_network.pth`. With the trained network, we can run `python test_residual_physics.py -model residual`.



