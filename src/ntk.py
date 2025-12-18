import numpy as np
import torch
import torch.nn as nn
from typing import Optional


def compute_ntk_gram(model, X, device='cpu'):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()

    X = X.to(device)
    model = model.to(device)
    model.eval()

    N = X.shape[0]
    K = torch.zeros((N, N), device=device)

    for i in range(N):
        model.zero_grad()

        x_i = X[i:i+1]
        out_i = model(x_i)

        grads_i = torch.autograd.grad(
            outputs=out_i,
            inputs=model.parameters(),
            create_graph=False,
            retain_graph=False
        )

        grad_i_flat = torch.cat([g.reshape(-1) for g in grads_i])

        for j in range(i, N):
            model.zero_grad()

            x_j = X[j:j+1]
            out_j = model(x_j)

            grads_j = torch.autograd.grad(
                outputs=out_j,
                inputs=model.parameters(),
                create_graph=False,
                retain_graph=False
            )

            grad_j_flat = torch.cat([g.reshape(-1) for g in grads_j])

            K[i, j] = torch.dot(grad_i_flat, grad_j_flat)
            K[j, i] = K[i, j]

    return K.cpu().numpy()


def compute_ntk_gram_batched(model, X, device='cpu', batch_size=32):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()

    X = X.to(device)
    model = model.to(device)
    model.eval()

    N = X.shape[0]

    jacobians = []

    for i in range(0, N, batch_size):
        batch = X[i:min(i+batch_size, N)]
        batch_size_actual = batch.shape[0]

        batch_jac = []
        for j in range(batch_size_actual):
            model.zero_grad()
            out = model(batch[j:j+1])
            grads = torch.autograd.grad(
                outputs=out,
                inputs=model.parameters(),
                create_graph=False,
                retain_graph=False
            )
            grad_flat = torch.cat([g.reshape(-1) for g in grads])
            batch_jac.append(grad_flat)

        jacobians.extend(batch_jac)

    J = torch.stack(jacobians)

    K = J @ J.T

    return K.cpu().numpy()


def effective_dimension_ntk(K, lam, N):
    eigs = np.linalg.eigvalsh(K)
    eigs = eigs[eigs > 0]

    deff = np.sum(eigs / (eigs + lam * N))

    return deff, eigs


def analytical_ntk_relu_2layer(X, width, init_scale=1.0):
    N = X.shape[0]
    K = np.zeros((N, N))

    for i in range(N):
        for j in range(i, N):
            x_i = X[i]
            x_j = X[j]

            dot = np.dot(x_i, x_j)
            norm_i = np.linalg.norm(x_i)
            norm_j = np.linalg.norm(x_j)

            if norm_i > 0 and norm_j > 0:
                cos_theta = np.clip(dot / (norm_i * norm_j), -1, 1)
                theta = np.arccos(cos_theta)

                K[i, j] = (norm_i * norm_j / (2 * np.pi)) * (np.sin(theta) + (np.pi - theta) * cos_theta)
            else:
                K[i, j] = 0

            K[j, i] = K[i, j]

    return K * init_scale**2
