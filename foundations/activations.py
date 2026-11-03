import numpy as np
from numpy.typing import NDArray
import math


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        final = []
        for num in z:
            x = round(float(1 / (1 + math.e ** (-num))),5)
            final.append(x)
        return final
        

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        final = []
        for num in z:
            x = round(float(max(0, num)),5)
            final.append(x)
        return final
