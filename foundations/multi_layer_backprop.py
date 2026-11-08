import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        
        # Convert lists to numpy arrays
        x_arr = np.array(x, dtype=float)
        W1_arr = np.array(W1, dtype=float)
        b1_arr = np.array(b1, dtype=float)
        W2_arr = np.array(W2, dtype=float)
        b2_arr = np.array(b2, dtype=float)
        y_arr = np.array(y_true, dtype=float)

        # ------------------ Forward Pass ------------------
        # z1 shape: (hidden_size,)
        z1 = np.dot(W1_arr, x_arr) + b1_arr
        a1 = np.maximum(0, z1)

        # z2 shape: (output_size,)
        z2 = np.dot(W2_arr, a1) + b2_arr

        # Loss (scalar float)
        loss = np.mean((z2 - y_arr) ** 2)

        # ------------------ Backward Pass -----------------
        # dz2: d(Loss)/d(z2) = 2 * (z2 - y_true) / output_size (output_size = len(y_true))
        dz2 = 2.0 * (z2 - y_arr) / len(y_arr)

        # dW2: shape (output_size, hidden_size)
        dW2 = np.outer(dz2, a1)
        db2 = dz2.copy()

        # da1: flow error back through W2
        da1 = np.dot(dz2, W2_arr)

        # dz1: gate through ReLU derivative
        dz1 = da1 * (z1 > 0)

        # dW1: shape (hidden_size, input_size)
        dW1 = np.outer(dz1, x_arr)
        db1 = dz1.copy()

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }