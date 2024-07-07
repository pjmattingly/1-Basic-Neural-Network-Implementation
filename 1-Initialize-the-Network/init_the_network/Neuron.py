import random
from typing import List
import numbers
import numpy as np

class Neuron:
    """
    A single neuron in a neural network.

    Attributes:
        weights (list of float): Weights associated with the neuron's inputs.
        bias (float): Bias term added to the weighted sum of inputs.
    """

    def __init__(self, inputs : int):
        """
        Initialize a Neuron with a specified number of inputs.

        Parameters:
            inputs (int): The number of input connections to the neuron.

        Raises:
            TypeError: If 'inputs' is not an integer.
            ValueError: If 'inputs' is a negative integer.
        """

        if not isinstance(inputs, int):
            raise TypeError("Argument 'inputs' must be an integer.")
        if inputs < 0:
            raise ValueError("Argument 'inputs' must be a non-negative integer.")
        
        self._weights = np.random.rand(inputs)
        self._bias = random.random()
        self._activation_function = self._relu #TODO, more activation functions

    def forward_pass(self, inputs : List[float]) -> float:
        if isinstance(inputs, str):
            raise TypeError("Argument 'inputs' must be a list of numbers.")
        
        #https://stackoverflow.com/a/1952481
        try:
            iter(inputs)
        except TypeError:
            raise TypeError("Argument 'inputs' must be an iterable.")
        else:
            #the weighted sum of the inputs through the activation function
            return self._activation_function( 
                np.sum(np.dot(np.array(inputs), self._weights)) 
                )

    def _relu(self, _in : float):
        if not isinstance(_in, numbers.Real):
            raise TypeError("Argument '_in' must be a real number.")

        return max([0, _in])