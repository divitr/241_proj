import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple


class TwoLayerMLP(nn.Module):

    def __init__(self, d_in, width, d_out=1, activation='relu', init_scale=1.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, width)
        self.fc2 = nn.Linear(width, d_out)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'erf':
            self.activation = lambda x: torch.erf(x)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self._init_weights(init_scale)

    def _init_weights(self, scale):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=scale / np.sqrt(self.fc1.in_features))
        nn.init.normal_(self.fc2.weight, mean=0.0, std=scale / np.sqrt(self.fc2.in_features))
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.activation(self.fc1(x))
        return self.fc2(h).squeeze(-1)


def train_mlp(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    lr: float = 0.001,
    max_epochs: int = 10000,
    target_train_mse: Optional[float] = None,
    batch_size: Optional[int] = None,
    device: str = 'cpu',
    verbose: bool = False
) -> Tuple[list, list]:

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)

    if X_test is not None and y_test is not None:
        X_test_t = torch.from_numpy(X_test).float().to(device)
        y_test_t = torch.from_numpy(y_test).float().to(device)
        has_test = True
    else:
        has_test = False

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    N = X_train.shape[0]
    if batch_size is None:
        batch_size = N

    for epoch in range(max_epochs):
        model.train()

        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, N, batch_size):
            batch_idx = perm[i:min(i + batch_size, N)]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)

        if has_test:
            model.eval()
            with torch.no_grad():
                y_test_pred = model(X_test_t)
                test_loss = criterion(y_test_pred, y_test_t).item()
                test_losses.append(test_loss)

        if verbose and (epoch % 100 == 0 or epoch == max_epochs - 1):
            if has_test:
                print(f"Epoch {epoch}: Train MSE = {avg_train_loss:.6f}, Test MSE = {test_loss:.6f}")
            else:
                print(f"Epoch {epoch}: Train MSE = {avg_train_loss:.6f}")

        if target_train_mse is not None and avg_train_loss <= target_train_mse:
            if verbose:
                print(f"Reached target train MSE at epoch {epoch}")
            break

    return train_losses, test_losses


def compute_mlp_test_mse(model, X_test, y_test, device='cpu'):
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        y_pred = model(X_test_t)
        mse = torch.mean((y_pred - y_test_t) ** 2).item()

    return mse
