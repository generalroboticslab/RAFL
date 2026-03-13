    
### Code structure for Sim-to-sim beam model
    .
    ├── vibration # Sim2sim experiments for vibrating beams
    ├── twist # Sim2sim experiments for twisted beams
    ├── mesh # Contains tetrahedral mesh for Finer Fishtail (mesh is voxelized to allow compatibility with hex mesh simulation)
    ├── beam_residual_physics.py # Called by training.py for supervised learning
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
    ├── env_fishTail.py # Crude Fishtail DiffPD class.
    ├── env_fishTailMesh.py # Finer Fishtail DiffPD class.

For oscillating and twisting sim2sim experiments in `vibration` and `twist`:

    .
    ├── data_generation.py # Generate target data for the canonical beam.
    ├── data_generation_longer.py # Generate target data for the longer beam (non-scaled).
    ├── data_generation_longer_scaled.py # Generate target data for the longer beam (scaled).
    ├── data_generation_shorter.py # Generate target data for the shorter beam (non-scaled).
    ├── data_generation_shorter_scaled.py # Generate target data for the shorter beam (scaled).
    ├── data_generation_thicker.py # Generate target data for the thicker beam (non-scaled).
    ├── data_generation_thicker_scaled.py # Generate target data for the thicker beam (scaled).
    ├── data_generation_thinner.py # Generate target data for the thinner beam (non-scaled).
    ├── data_generation_thinner_scaled.py # Generate target data for the thinner beam (caled).
    ├── data_generation_fishTail.py # Generate target data for the crude fishtail.
    ├── data_generation_fishTailMesh.py # Generate target data for the finer fishtail.
    ├── optimize_trajectory.py # Optimize residual forces to match the real trajectory of canonical beam (exclusively used for supervised learning baseline).
    ├── optimize_trajectory_longer.py # Generate formatted trajectory data compatible with testing script for longer beam (non-scaled)
    ├── optimize_trajectory_longer_scaled.py # Generate formatted trajectory data compatible with testing script for longer beam (scaled)
    ├── optimize_trajectory_shorter.py # Generate formatted trajectory data compatible with testing script for shorter beam (non-scaled)
    ├── optimize_trajectory_shorter_scaled.py # Generate formatted trajectory data compatible with testing script for shorter beam (scaled)
    ├── optimize_trajectory_thicker.py # Generate formatted trajectory data compatible with testing script for thicker beam (non-scaled)
    ├── optimize_trajectory_thicker_scaled.py # Generate formatted trajectory data compatible with testing script for thicker beam (scaled)
    ├── optimize_trajectory_thinner.py # Generate formatted trajectory data compatible with testing script for thinner beam (non-scaled)
    ├── optimize_trajectory_thinner_scaled.py # Generate formatted trajectory data compatible with testing script for thinner beam (scaled)
    ├── optimize_trajectory_fishTail.py # Generate formatted trajectory data compatible with testing script for crude fishtail
    ├── optimize_trajectory_fishTailMesh.py # Generate formatted trajectory data compatible with testing script for finer fishtail
    ├── training.py # Supervised Learning Residual Physics Training script for canonical beam
    ├── beam_residual_physics.py # Called by training.py for supervised learning
    ├── training_direct.py # RAFL Training script for canonical beam
    ├── finetune_fishTail.py # RAFL finetuning script for for crude fishtail
    ├── finetune_fishTailMesh.py # RAFL finetuning script for for finer fintail
    ├── test_residual_physics.py # Test residual physics framework for canonical beam
    ├── test_residual_physics_longer.py # Test residual physics framework for longer beam (non-scaled)
    ├── test_residual_physics_longer_scaled.py # Test residual physics framework for longer beam (scaled)
    ├── test_residual_physics_shorter.py # Test residual physics framework for shorter beam (non-scaled)
    ├── test_residual_physics_shorter_scaled.py # Test residual physics framework for shorter beam (scaled)
    ├── test_residual_physics_thicker.py # Test residual physics framework for thicker beam (non-scaled)
    ├── test_residual_physics_thicker_scaled.py # Test residual physics framework for thicker beam (scaled)
    ├── test_residual_physics_thinner.py # Test residual physics framework for thinner beam (non-scaled)
    ├── test_residual_physics_thinner_scaled.py # Test residual physics framework for thinner beam (scaled)
    ├── test_residual_physics_fishTail.py # Test residual physics framework for crude fishtail
    ├── test_residual_physics_fishTailMesh.py # Test residual physics framework for finer fintail
    ├── test_residual_physics_fishTail_finetune.py # Test finetuned residual physics framework for crude fishtail
    ├── test_residual_physics_fishTailMesh_finetune.py # Test finetuned residual physics framework for finer fintail