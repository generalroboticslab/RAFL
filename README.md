# RAFL: Generalizable Sim-to-Real of Soft Robots with Residual Acceleration Field Learning

[Dong Heon Cho*](http://dc3042.github.io),
[Boyuan Chen](http://boyuanchen.com/)

<br>
Duke University
<br>


### [Project Website](http://generalroboticslab.com/RAFL) | [Video]() | [Paper]()

## Overview
This repo contains the PyTorch implementation for our paper "RAFL: Generalizable Sim-to-Real of Soft Robots with Residual Acceleration Field Learning".

![teaser](asset/banner.pdf)

Note: this is a modified version of https://github.com/srl-ethz/residual_physics_sim2real.git and https://github.com/mit-gfx/diff_pd, with the purpose of learning generalizable residual acceleration fields of soft deformable objects.

## Citation

If you find our paper or codebase helpful, please consider citing:

```
PLACEHOLDER

```

## Recommended systems
- Ubuntu 18.04
- (Mini)conda 4.7.12 or higher
- GCC 7.5 (Other versions might work but we tested the codebase with 7.5 only)

## Content

- [Installation](#installation)
- [Sim-to-Sim](#sim-to-sim)
- [Sim-to-Real](#sim-to-real)
- [License](#license)

## Installation

## Recommended systems
- Ubuntu 18.04
- (Mini)conda 4.7.12 or higher
- GCC 7.5 (Other versions might work but we tested the codebase with 7.5 only)

Clone the repository and install the dependencies.
```
git clone --recursive https://github.com/generalroboticslab/RAFL.git
cd RAFL
conda env create -f environment.yml
conda activate residual_physics
./install.sh
```

## Sim-to-Sim

Navigate to the corresponding sub-directory by 

```
cd python/beam_model/sim2sim_beam_model/[vibration/twist]
```

### Data Generation
To generate the target simulation trajectories and optimized residual forces for the canonical beam

```
python data_generation.py
python optimize_trajectories.py
```
This will additionally generate target residual forces for the supervised residual learning baseline. 
All raw trajectories will be stored under ```data_real``` subdirectory and all target datasets will be stored under ```data_sim2sim```.

Addtionally, to generate target simulation trajectories for various shapes run
```
python data_generation_[shape].py
python optimize_trajectories_[shape].py
```
where ```shape``` can be the following 
- longer: Longer beam (unscaled)
- longer_scaled: Longer beam (scaled)
- shorter: Shorter beam (unscaled)
- shorter_scaled: Shorter beam (scaled)
- thicker: Thicker beam (unscaled)
- thicker_scaled: Thicker beam (scaled)
- thinner: Thinner beam (unscaled)
- thinner_scaled: Thinner beam (scaled)
- fishTail: Crude Fishtail
- fishTailMesh: Finer Fishtail
Target residual forces for the supervised residual learning baseline are NOT generated for generalization datasets, but target datasets are reformatted to allow compatibility with test code. 
All raw trajectories will be stored under ```[shape]_data_real``` subdirectory and all target datasets will be stored under ```[shape]_data_sim2sim```.

### Training

To train our RAFL framework directly on target trajectories, run
```
python training_direct.py
```
The trained model will be stored under ```training/test_refactor_element_direct``` subdirectory 

To train the supervised residual baseline, run 
```
python training.py
```
The trained model will be stored under ```training/test_refactor``` subdirectory 

### Testing

To test the model, run
```
python test_residual_physics.py
```
Resulting trajectory plots are stored under ```sim2sim/test_refactor_element_direct``` subdirectory and raw trajectory/error data are stored under the ```training/test_refactor_element_direct``` subdirectory 

For generalization test, run
```
python test_residual_physics_[shape].py
```
Resulting trajectory plots are stored under ```[shape]_sim2sim/test_refactor_element_direct``` subdirectory and raw trajectory/error data are stored under the ```training/test_refactor_element_direct/[shape]``` subdirectory 

### Fine-tuning
For the two fish tail shapes, we additionally allow further finetuning via the function
```
python fine_tune_[fishTail/fishTailMesh].py
```
The finetuned model will be stored under ``training/test_refactor_element_direct/[fishTail/fishTailMesh]_finetune``` subdirectory 

To test the finetuned models run
```
python test_residual_physics_[fishTail/fishTailMesh]_finetune.py
```
Resulting trajectory plots are stored under ```[fishTail/fishTailMesh]_finetune_sim2sim/test_refactor_element_direct/[fishTail/fishTailMesh]_finetune``` subdirectory and raw trajectory/error data are stored under the ```training/test_refactor_element_direct/[fishTail/fishTailMesh]_finetune``` subdirectory 

## Sim-to-Real

Navigate to the corresponding sub-directory by 

```
cd python/beam_model/sim2real_beam_model
```

### Data Generation
You can download Sim2real beam real data from [Google Drive](https://drive.google.com/drive/folders/16v8ItWT0kAm8PBaxGByvA-moDE78VFxq?usp=sharing)
, where `data_new`, `data_longer`, `data_shorter`, `data_thicker`, `data_thinner` store the collected raw data and `cantilver_data_new_straight`, `cantilver_data_longer_straight`, `cantilver_data_longer_scaled_straight`, `cantilver_data_shorter_straight`, `cantilver_data_shorter_scaled_straight`, `cantilver_data_thicker_straight`, `cantilver_data_thicker_scaled_straight`, `cantilver_data_thinner_straight`, `cantilver_data_thinner_scaled_straight` store the target datasets with optimized initial states and target forces where applicable.

Alternatively, to generate the initial states and optimized residual forces for the canonical beam from raw data, run
```
python generate_augmented_data.py
```
This will additionally generate target residual forces for the supervised residual learning baseline. 
All target datasets will be stored under ```cantilever_data_new_straight```.

Addtionally, to generate target simulation trajectories for various shapes run
```
python generate_data_[shape].py
```
where ```shape``` can be the following 
- longer: Longer beam (unscaled)
- longer_scaled: Longer beam (scaled)
- shorter: Shorter beam (unscaled)
- shorter_scaled: Shorter beam (scaled)
- thicker: Thicker beam (unscaled)
- thicker_scaled: Thicker beam (scaled)
- thinner: Thinner beam (unscaled)
- thinner_scaled: Thinner beam (scaled)
Target residual forces for the supervised residual learning baseline are NOT generated for generalization datasets, and only initial states are generated. 
All target datasets will be stored under ```cantilver_data_[shape]_straight```.

### Training

To train our RAFL framework directly on target trajectories, run
```
python training_direct.py
```
The trained model will be stored under ```training/test_refactor_element_direct``` subdirectory 

To train the supervised residual baseline, run 
```
python training.py
```
The trained model will be stored under ```training/test_refactor``` subdirectory 

### Testing

To test the model, run
```
python test_residual_physics.py
```
Resulting trajectory plots are stored under ```2dplots_displacement/test_refactor_element_direct``` subdirectory and raw trajectory/error data are stored under the ```training/test_refactor_element_direct``` subdirectory 

For generalization test, run
```
python test_residual_physics_[shape].py
```
Resulting trajectory plots are stored under ```[shape]_2dplots_displacement/test_refactor_element_direct``` subdirectory and raw trajectory/error data are stored under the ```training/test_refactor_element_direct/[shape]``` subdirectory 


### Expanded Generalization

Additionally, RAFL can be trained on different shapes without requiring target residual forces. 

To train on each ```shape```, run
```
python training_direct_[shape].py
```
The trained model will be stored under ```training/test_refactor_element_[shape]_direct``` subdirectory 

### System Identification

To run system identification (fitting both Young's Modulus and Poissons's Ratio) on the canonical beam, run
```
python beam_sys_all.py
```
and for each ```shape```,
```
python beam_sys_all_[shape].py
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

