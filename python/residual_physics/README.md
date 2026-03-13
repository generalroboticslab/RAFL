### Training options for Supervised Learning Baseline (Gao et al.) framework

- seed [Int]: set random seed for training. Default: 42
- epochs [Int]: total training epochs. Default: 1000
- learning_rate [Float]: optimizer learning rate. Default: 1e-3
- optimizer: the optimizer to use when training neural network. Only support Adam for now. You can easily add other optimizers
- start_frame [Int] :The starting timestamp we select for each trajectory in the training set
- end_frame [Int]: The last timestamp we use for each trajectory in the traininig set.
- training_set [List[Int]]: Training set trajectory index we use for training
- validate_set [List[Int]]: Validation set trajectory index. No validation will be performed if not provided.
- cuda [Int | String]: The cuda device used for training. CPU will be used if not provided.
- normalize [Bool]: Perform normalization on inputs and outputs for neural network or not. Default: True
- Scale [Int]: Scaling the training loss after each epoch. Default: 1.
- data_type [String]: To select for which dataset to use
- weight_decay [float]: Weight decay for optimizer. Default: 0.0
- validate_physics [Bool]: Perform validation by running diffpd simulations. However, this is very time consuming.
- validate_epochs [Int]: This will be set only when `validate_physics` = True, so that we only perform validation each `validate_epochs`.
- fit [String] = "forces" | "SITL": [forces]: train the network by supervised learning on residual forces. [SITL]: train the network with "Solver-in-the-loop" formulation, such that we don't need residual forces.
- tolerance [Int]: After `tolerance` epochs, if the validation error is not improved, we stop training.

The skip_connection architecture used for supervised learning baseline requires the following parameters:
- num_mlp_blocks [Int]: How many MLP blocks in the network.
- hidden_size [Int]: Hidden_size in each MLP block.
- num_block_layer [Int]: The number of layers for each MLP.

### Training options for RAFL (ours) framework

- seed [Int]: set random seed for training. Default: 42
- epochs [Int]: total training epochs. Default: 100
- learning_rate [Float]: optimizer learning rate. Default: 1e-3
- optimizer: the optimizer to use when training neural network. Only support Adam for now. You can easily add other optimizers
- training_set [List[Int]]: Training set trajectory index we use for training
- validate_set [List[Int]]: Validation set trajectory index. No validation will be performed if not provided.
- normalize [Bool]: Perform normalization on inputs and outputs for neural network or not. Default: False
- Scale [Int]: Scaling the training loss. Default: 1e3.
- weight_decay [float]: Weight decay for optimizer. Default: 1e-5
- fit [String] = "forces" | "SITL": [forces]: train the network by supervised learning on residual forces. [SITL]: train the network with "Solver-in-the-loop" formulation, such that we don't need residual forces.

Residual Acceleration Field architecture requires the following parameters:
- hidden_size [Int]: Hidden_size in each MLP block. Default: 64
- num_hidden_layer [Int]: The number of layers. Default: 4