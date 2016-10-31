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

    #Run method
    def Run(self, input):
        #Run network based on input data

        '''shape returns the no. of input data such that for:
            test = np.array([[1,2],[3,4],[5,6]])
            test.shape[0] returns 3, i.e. the no. of training cases
        '''
        lnCases = input.shape[0]

        #Clear out previous intermediate lists
        self._layerInput = []
        self._layerOutput = []

        #Run network
        #if its an input layer we need to get data from input
        #if its an output, otherwise
        for index in range(self.layerCount):
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, lnCases])]))





    #Transfer function
    def sigmoid(self, x, Derivative=False):
        if not Derivative:
            return 1 / (1+np.exp(-x))
        else:
            out = self.sigmoid(x)
            return out*(1-out)

if __name__ == "__main__":
    bpn = BackPropagationNetwork((2,2,1))
    print(bpn.shape)
    print(bpn.weights)
