import logging
import torch
from torch.autograd import Variable
import scipy
import numpy as np
log = logging.getLogger(__name__)
dtype = torch.FloatTensor


def R_axis(k, a, b, angle):
    # n_angles = int(scipy.misc.comb(k, 2))
    M = Variable(torch.eye(k).type(dtype), requires_grad=False)
    M[a, a] = torch.cos(angle)
    M[b, b] = torch.cos(angle)
    M[a, b] = -torch.sin(angle)
    M[b, a] = torch.sin(angle)
    return M


def R(k, angles):
    rotation_matrix = Variable(torch.eye(k).type(dtype), requires_grad=False)
    i = 0
    for a in range(k):
        for b in range(k):
            if a <= b:
                continue
            next_rot = R_axis(k, a, b, angles[i])
            rotation_matrix = torch.mm(rotation_matrix, next_rot)
            i += 1

    return rotation_matrix


def Cayley(k, angles):
    C_left = Variable(torch.eye(k).type(dtype), requires_grad=False)
    C_right = Variable(torch.eye(k).type(dtype), requires_grad=False)
    angles_c = 0
    for i in range(k):
        for j in range(i, k):
            if i == j:
                continue
            C_left[i, j] = angles[angles_c]
            C_left[j, i] = -angles[angles_c]
            C_right[i, j] = -angles[angles_c]
            C_right[j, i] = angles[angles_c]
            angles_c += 1
    Q = torch.inverse(C_left).mm(C_right)
    return Q


def get_random_rotation_matrix(k):
    n_angles = int(scipy.misc.comb(k, 2))
    angles = torch.rand(n_angles) * 2 * 3.14159
    angles = Variable(angles.type(dtype), requires_grad=False)
    return R(k, angles).data.numpy()


def optimize(basis, loss_fun="trace"):
    # Schuhlöffel
    cayley = True
    if loss_fun == "trace":
        optim = torch.optim.Adagrad
        lr = 0.5
        early_stop_after = 5
        max_steps = 30
    elif loss_fun == "L1":
        optim = torch.optim.Adagrad
        lr = 0.1
        early_stop_after = 5
        max_steps = 40
    elif loss_fun == "nearest":
        optim = torch.optim.Adagrad
        lr = 0.2
        early_stop_after = 5
        max_steps = 40
    elif loss_fun == "max":
        optim = torch.optim.Adagrad
        lr = 0.2
        early_stop_after = 5
        max_steps = 40
    else:
        optim = torch.optim.Adagrad
        lr = 0.35
        early_stop_after = 5
        max_steps = 40

    k = basis.shape[0]
    n_angles = int(scipy.misc.comb(k, 2))
    if n_angles == 0:
        return basis

    if loss_fun == "nearest":
        best_axis_idxs = np.argsort(-(basis ** 2).sum(0))[:k]
        best_cols = basis[:, best_axis_idxs]
        _, col_ind = scipy.optimize.linear_sum_assignment(-np.abs(best_cols))  # hungarian algo

        best_axis_ids_order = list(map(int, best_axis_idxs[col_ind]))
        sign_mask = Variable(torch.from_numpy(np.sign(best_cols[range(k), col_ind])))

    angles = Variable(torch.zeros(n_angles,).type(dtype), requires_grad=True)
    basis = Variable(torch.from_numpy(basis).type(dtype), requires_grad=False)

    optimizer = optim([angles], lr=lr)

    best_loss = 999999999
    best = None
    stop_in = early_stop_after

    for i in range(max_steps):
        optimizer.zero_grad()
        if cayley:
            rotation_matrix = Cayley(k, angles)
        else:
            rotation_matrix = R(k, angles)
        new_basis = rotation_matrix.mm(basis)

        if loss_fun == "trace":
            loss = -torch.trace(new_basis)  # equivalent to euclidean dist to I in Stiefel manifold
        elif loss_fun == "L1":
            loss = torch.abs(new_basis).sum()  # l1
        elif loss_fun == "L1+trace":
            loss = torch.abs(new_basis).sum() - torch.trace(new_basis)  # l1 and trace
        elif loss_fun == "max":
            # might approximate euclidean distance to (Stiefel) closest basic plane
            loss = -torch.max(torch.abs(new_basis), 1)[0].sum()
        elif loss_fun == "nearest":
            # euclidean distance to (Grassmannian) closest basic plane
            axis_vals = new_basis[range(k), best_axis_ids_order]
            axis_vals_signs = torch.mul(axis_vals, sign_mask)
            loss = -torch.sum(axis_vals_signs)

        loss_val = np.round(loss.data[0], 3)

        if loss_val < best_loss:
            best_loss = loss_val
            best = new_basis.data
            stop_in = early_stop_after
        else:
            stop_in -= 1
            if stop_in == 0:
                break

        # loss.backward(retain_graph=True)
        loss.backward()

        optimizer.step()

    return best.numpy()