import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        y_pred = np.dot(X,weights)
        return np.round(y_pred,5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        sum = 0
        for i in range(len(ground_truth)):
            sum += ((model_prediction[i] - ground_truth[i])**2)
        final = np.round(sum/len(ground_truth),5)
        return final.item()
