#allow for relative importing for pytest, and absolute importing when running directly
try:
    from .Layer import Layer
except ImportError:
    from Layer import Layer
from typing import List
import numpy as np

class Network:
    """
    A neural network consisting of multiple layers.

    Attributes:
        _network (List[Layer]): List of layers in the neural network.
    """

    def __init__(self, architecture : List[int]):
        """
        Initialize the neural network based on the provided architecture.

        Parameters:
            architecture (List[int]): A list where each element represents the number
                                      of neurons in that layer of the network.

        Raises:
            TypeError: If 'architecture' is not an iterable or contains non-integers.
            ValueError: If 'architecture' contains negative integers or is empty.
        """

        #https://stackoverflow.com/a/1952481
        try:
            iter(architecture)
        except TypeError:
            raise TypeError(
                "Argument 'architecture' must be an iterable of integers."
                )
        
        if len(architecture) == 0:
            raise TypeError(
                "Argument 'architecture' must be an iterable with at least one element."
                )

        na_architecture = np.array(architecture)

        if not np.issubdtype(na_architecture.dtype, np.int_):
            raise ValueError(
                "Argument 'architecture' must be an iterable of integers."
                )
        
        #https://stackoverflow.com/a/55523245
        if np.any((na_architecture < 0)):
            raise ValueError(
                "Argument 'architecture' must be an iterable of non-negative integers."
                )
        
        self._network = list()
        for i, num_neurons in enumerate(architecture):
            if i == 0:
                #In the input layer each Neuron only has a single input.
                self._network.append( Layer(1, num_neurons) )
            else:
                #each subsequent layer has inputs equal to the number of Neurons in \
                # the previous layer.
                self._network.append( Layer(architecture[i-1], num_neurons) )
    
    def forward_pass(self, inputs : List[float]) -> List[float]:
        """
        Perform a forward pass through the neural network.

        Parameters:
            inputs (List[float]): Input values for the input layer of the network.

        Returns:
            List[float]: Output values from the final layer of the network.

        Raises:
            TypeError: If 'inputs' is not a list of numeric elements.
            ValueError: If 'inputs' is not of the expected length or contains non-finite numeric elements.
        """

        #https://stackoverflow.com/a/1952481
        try:
            iter(inputs)
        except TypeError:
            raise TypeError(
                "Argument 'inputs' must be an iterable of numeric elements."
                )
        
        input_layer_size = self._network[0].get_size()
        
        if len(inputs) != input_layer_size:
            raise ValueError(
                f"Argument 'inputs' must be an iterable of size {input_layer_size}."
                )
        
        na_inputs = np.array(inputs)

        if not np.issubdtype(na_inputs.dtype, np.number):
            raise TypeError(
                "Argument 'inputs' must be an iterable of numeric elements."
                )
        if not np.isfinite(na_inputs).all():
            raise ValueError(
                "Argument 'inputs' must be an iterable of finite numeric elements."
                )
        
        outputs = inputs
        for layer in self._network:
            outputs = layer.forward_pass(outputs)

        return outputs
    
    def _backward_pass(self):
        #PLACEHOLDER
        pass
    
    def training(self, data, learning_rate=None, epochs=None):
        '''
        #doing a simple training algorithm
        #no batching, using the entire training set for training
        #no splitting data into "Training" and "Validation" sets
        #no early stopping criteria

        self._check_data(data)

        for _ in range(epochs):
            data = random.shuffle(data)

            outputs = [self.forward_pass(x) for x in data[x]]

            _log_accuracy(results, data[y])

            losses = loss_calc(results, data[y])

            #logging loss
            print(losses)

            #how to do the backward pass? What parameters does it need?
            new_weights_and_biases = self._backward_pass(losses) #PLACEHOLDER

            for i, update in enumerate(new_weights_and_biases):
                #TODO, how to update the new weights and biases given the current 
                # learning rate during the i_th epoch?
                new_weight = update[weight]*learning_rate #PLACEHOLDER
                new_bias = update[bias]*learning_rate #PLACEHOLDER

                self.update_weight(i, new_weight)
                self.update_bias(i, new_bias)
        '''
        pass

    def _check_data(self, data):
        pass