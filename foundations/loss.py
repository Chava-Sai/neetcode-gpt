import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        sum = 0
        y_pred_mod = np.clip(y_pred,1e-7,1 - 1e-7)
        for i in range(len(y_true)):
            if y_true[i] == 1:
                sum += math.log(y_pred_mod[i])
            else:
                sum += math.log(1-y_pred_mod[i])
        return round(-1 * sum / len(y_pred_mod),4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        sum = 0
        y_pred_mod = np.clip(y_pred,1e-7,1 - 1e-7)
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                if y_true[i][j] == 1:
                    sum += math.log(y_pred_mod[i][j])
        return  round(-1 * sum / len(y_pred),4)


