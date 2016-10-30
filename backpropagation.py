import numpy as np

A = [1,2,3]
list(zip(A))

class BackPropagationNetwork:
    ##Class members
    layerCount = 0
    shape = None
    weights = [];

    def __init__(self, layerSize):
        """Initialise the network"""

        #the input layer is just a buffer which holds information
        self.layerCount = len(layerSize) - 1
        self.shape = layerSize

        #input/output data from last run
        self._layerInput = []
        self._layerOutput = []

        #create the weight arrays(l1+1 because of the bias term)
        for (l1,l2) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(np.random.normal(scale=0.01, size=(l2, l1+1)))

if __name__ == "__main__":
    bpn = BackPropagationNetwork((2,2,1))
    print(bpn.shape)
    print(bpn.weights)
