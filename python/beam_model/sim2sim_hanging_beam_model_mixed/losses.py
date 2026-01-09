import numpy as np


def stepwise_loss_and_grad(objective, q, v, i, frame_num):

    if "negative" in objective:
        return negative_stepwise_loss_and_grad(q, v, i)
    elif "positive" in objective:
        return positive_stepwise_loss_and_grad(q, v, i)
    elif "maxAmplitude" in objective:
        return maxAmplitude_stepwise_loss_and_grad(q, v, i, frame_num)
    elif "minAmplitude" in objective:
        return minAmplitude_stepwise_loss_and_grad(q, v, i, frame_num)
    else:
        exit()

def trajectory_loss_and_grad(objective, q_list, v_list, prepare=200):

    if "negative" in objective:
        return negative_trajectory_loss_and_grad(q_list, v_list, prepare)
    elif "positive" in objective:
        return positive_trajectory_loss_and_grad(q_list, v_list, prepare)
    elif "maxAmplitude" in objective:
        return maxAmplitude_trajectory_loss_and_grad(q_list, v_list, prepare)
    elif "minAmplitude" in objective:
        return minAmplitude_trajectory_loss_and_grad(q_list, v_list, prepare)
    else:
        exit()

x_mean = 0.00525
min_z_idx = [  0,  31,  62,  93, 124, 155, 186, 217, 248, 279, 310, 341, 372, 403, 434, 465]

# Most negative mean x position
def negative_stepwise_loss_and_grad(q, v, i):
    loss = - np.maximum(0, x_mean - q.reshape(-1, 3)[:, 0]).mean()

    q_flat = q.reshape(-1, 3)
    N = q_flat.shape[0]

    grad_q = np.zeros_like(q_flat)

    # d/dq_x of mean(max(0, q_x))
    grad_q[:, 0] = (q_flat[:, 0] < x_mean).astype(q.dtype) * (1.0 / N)

    grad_v = np.zeros_like(v)

    return loss, grad_q.reshape(q.shape), grad_v

def negative_trajectory_loss_and_grad(q_list, v_list, prepare=200):

    return 0.0, [np.zeros_like(q_list[0])] * len(q_list), [np.zeros_like(v_list[0])] * len(v_list)

# Most positive mean x position
def positive_stepwise_loss_and_grad(q, v, i):
    loss = - np.maximum(0, q.reshape(-1, 3)[:, 0] - x_mean).mean()

    q_flat = q.reshape(-1, 3)
    N = q_flat.shape[0]

    grad_q = np.zeros_like(q_flat)

    # d/dq_x of mean(max(0, q_x))
    grad_q[:, 0] = - (q_flat[:, 0] > x_mean).astype(q.dtype) * (1.0 / N)

    grad_v = np.zeros_like(v)

    return loss, grad_q.reshape(q.shape), grad_v

def positive_trajectory_loss_and_grad(q_list, v_list, prepare=200):

    return 0.0, [np.zeros_like(q_list[0])] * len(q_list), [np.zeros_like(v_list[0])] * len(v_list)


# Maximum oscillation x amplitude of bottom elements
def maxAmplitude_stepwise_loss_and_grad(q, v, i, frame_num):
    loss =  - np.abs(q.reshape(-1, 3)[min_z_idx, 0].mean() - x_mean) / frame_num
    q_flat = q.reshape(-1, 3)
    N = len(min_z_idx) 
    
    grad_q = np.zeros_like(q_flat) 
    grad_q_min_z_idx = np.zeros_like(q_flat[min_z_idx]) 

    if q_flat[min_z_idx, 0].mean() > x_mean:
        grad_q[min_z_idx] = - 1.0 / (N * frame_num)
    elif q_flat[min_z_idx, 0].mean() < x_mean:
        grad_q[min_z_idx] =  1.0 / (N * frame_num)
    
    grad_v = np.zeros_like(v) 
    
    return loss, grad_q.reshape(q.shape), grad_v
    # return 0.0, np.zeros_like(q), np.zeros_like(v)

def maxAmplitude_trajectory_loss_and_grad(q_list, v_list, prepare=200):
    """
    q_list: list of q[i], each shape (dofs,)
    v_list: unused here, but kept for interface symmetry
    prepare: number of warmup steps (frames before we start measuring spread)

    Returns:
        loss: scalar
        dq_full: np.ndarray of shape (num_steps, dofs)
        dv_full: np.ndarray of shape (num_steps, dofs)  (all zeros here)
    """
    num_steps = len(q_list)              # = prepare + frame_num + 1
    dofs = q_list[0].shape[0]

    dq_full = np.zeros((num_steps, dofs))
    dv_full = np.zeros((num_steps, dofs))  # loss does not depend on v

    # total internal integration steps
    internal_steps = num_steps - 1        # because q[0] is initial
    frame_num = internal_steps - prepare  # should match your frame_num arg

    # Collect mean bottom-row x over the measurement window
    # xs shape: (frame_num + 1, )
    xs = []
    for i in range(prepare, prepare + frame_num + 1):
        q_flat = q_list[i].reshape(-1, 3)
        x_bottom = q_flat[min_z_idx, 0].mean()
        xs.append(x_bottom)
    xs = np.array(xs)  # (frame_num, )

    # Find global max and min in this window
    max_idx = np.argmax(xs)  
    min_idx = np.argmin(xs)  

    t_max = prepare + max_idx
    t_min = prepare + min_idx

    x_max = xs[max_idx]
    x_min = xs[min_idx]

    # Loss: maximize spread
    loss = -(x_max - x_min)

    # Gradients: dL/dx_max = -1, dL/dx_min = +1
    grad_q_max = dq_full[t_max].reshape(-1, 3)
    grad_q_min = dq_full[t_min].reshape(-1, 3)

    grad_q_max[min_z_idx, 0] -= 1.0 / len(min_z_idx)
    grad_q_min[min_z_idx, 0] += 1.0 / len(min_z_idx)

    return loss, dq_full, dv_full

# Minimum oscillation x amplitude of bottom elements
def minAmplitude_stepwise_loss_and_grad(q, v, i, frame_num):
    loss = np.abs(q.reshape(-1, 3)[min_z_idx, 0].mean() - x_mean) / frame_num
    q_flat = q.reshape(-1, 3)
    N = len(min_z_idx) 
    
    grad_q = np.zeros_like(q_flat) 
    grad_q_min_z_idx = np.zeros_like(q_flat[min_z_idx]) 

    if q_flat[min_z_idx, 0].mean() > x_mean:
        grad_q[min_z_idx] = 1.0 / (N * frame_num)
    elif q_flat[min_z_idx, 0].mean() < x_mean:
        grad_q[min_z_idx] =  - 1.0 / (N * frame_num)
    
    grad_v = np.zeros_like(v) 
    
    return loss, grad_q.reshape(q.shape), grad_v
    # return 0.0, np.zeros_like(q), np.zeros_like(v)

def minAmplitude_trajectory_loss_and_grad(q_list, v_list, prepare=200):
    """
    q_list: list of q[i], each shape (dofs,)
    v_list: unused here, but kept for interface symmetry
    prepare: number of warmup steps (frames before we start measuring spread)

    Returns:
        loss: scalar
        dq_full: np.ndarray of shape (num_steps, dofs)
        dv_full: np.ndarray of shape (num_steps, dofs)  (all zeros here)
    """
    num_steps = len(q_list)              # = prepare + frame_num + 1
    dofs = q_list[0].shape[0]

    dq_full = np.zeros((num_steps, dofs))
    dv_full = np.zeros((num_steps, dofs))  # loss does not depend on v

    # total internal integration steps
    internal_steps = num_steps - 1        # because q[0] is initial
    frame_num = internal_steps - prepare  # should match your frame_num arg

    # Collect mean bottom-row x over the measurement window
    # xs shape: (frame_num + 1, )
    xs = []
    for i in range(prepare, prepare + frame_num + 1):
        q_flat = q_list[i].reshape(-1, 3)
        x_bottom = q_flat[min_z_idx, 0].mean()
        xs.append(x_bottom)
    xs = np.array(xs)  # (frame_num, )

    # Find global max and min in this window
    max_idx = np.argmax(xs)  
    min_idx = np.argmin(xs)  

    t_max = prepare + max_idx
    t_min = prepare + min_idx

    x_max = xs[max_idx]
    x_min = xs[min_idx]

    # Loss: minimize spread
    loss = (x_max - x_min)

    # Gradients: dL/dx_max = -1, dL/dx_min = +1
    grad_q_max = dq_full[t_max].reshape(-1, 3)
    grad_q_min = dq_full[t_min].reshape(-1, 3)

    grad_q_max[min_z_idx, 0] += 1.0 / len(min_z_idx)
    grad_q_min[min_z_idx, 0] -= 1.0 / len(min_z_idx)

    return loss, dq_full, dv_full
