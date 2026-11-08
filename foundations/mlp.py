import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        prev  = x
        for i in range(len(weights)):
            h_prev = np.dot(prev,weights[i]) + biases[i]
            a_prev = np.maximum(h_prev,0)
            prev = a_prev
        return h_prev

