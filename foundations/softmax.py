import numpy as np
from numpy.typing import NDArray
import math


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        mod_z = z - max(z)
        den = 0
        final = []
        for i in range(len(mod_z)):
            den += math.e ** mod_z[i]
        for i in range(len(mod_z)):
            final.append(round((math.e ** mod_z[i])/den,4))
        return final
