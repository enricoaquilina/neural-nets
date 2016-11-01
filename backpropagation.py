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
            self.weights.append(np.random.normal(scale=0.1, size=(l2, l1+1)))

    #Run method
    def run(self, input):
        #Run network based on input data

        '''shape returns the no. of input data such that for:
            test = np.array([[1,2],[3,4],[5,6]])
            test.shape[0] returns 3, i.e. the no. of training cases
        '''
        InputCases = input.shape[0]

        #Clear out previous intermediate lists
        self._layerInput = []
        self._layerOutput = []

        #Run network
        #if its an input layer we need to get data from input
        #if its an output, otherwise
        for index in range(self.layerCount):
            #Determine layer input
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, InputCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, InputCases])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.sigmoid(layerInput))

        return self._layerOutput[-1].T

    def train_epoch(self, input, target, trainingRate=0.2):
        #this method trains for one epoch

        deltas = [];
        InputCases = input.shape[0]

        #First run the network
        self.run(input)

        for layer in reversed(range(self.layerCount)):
            if layer == self.layerCount - 1:
                output_delta = self._layerOutput[layer] - target.T
                error = np.sum(output_delta ** 2)
                deltas.append(output_delta * self.sigmoid(self._layerInput[layer], True))
            else:
                #Sum(k∈K) = δk . Wjk
                delta_pullback = self.weights[layer + 1].T * deltas[-1]
                # Oj(1 − Oj)
                derivative = self.sigmoid(self._layerInputs[layer], True)
                #this is to remove the bias from the deltas which is in the last row
                deltas.append(delta_pullback[:-1,:] * derivative)

        #Compute the weight deltas
        for index in range(self.layerCount):
            delta_index = self.layerCount - 1 - index

            if index == 0:
                layerOutput = np.vstack([input.T, np.ones([1, InputCases])])
            else:
                layerOutput = np.vstack([self._layerOutput[index-1], np.ones([1, self._layerOutput[index - 1].shape[1]])])

            weightDelta = np.sum(
                             layerOutput[None, :, :].transpose(2, 0, 1) * deltas[delta_index][None, :, :].transpose(2, 1, 0)
                             , axis = 0)

            self.weights[index] -= trainingRate * weightDelta

        return error

    #Transfer function
    def sigmoid(self, x, Derivative=False):
        if not Derivative:
            return 1 / (1+np.exp(-x))
        else:
            out = self.sigmoid(x)
            return out*(1-out)

if __name__ == "__main__":
    bpn = BackPropagationNetwork((2,2,1))
    # print(bpn.layerCount)
    # print(bpn.weights[0])
    print(bpn.weights[0].T)
    print(bpn.shape)
    print(bpn.weights)

