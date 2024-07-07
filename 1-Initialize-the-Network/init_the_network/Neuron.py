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
        if len(inputs) != len(self._weights):
            raise ValueError(
                "Argument 'inputs' must be an iterable of the same size as the Neuron."
                )
        
        if isinstance(inputs, str):
            raise TypeError(
                "Argument 'inputs' must be an iterable of numeric elements."
                )
        
        #https://stackoverflow.com/a/1952481
        try:
            iter(inputs)
        except TypeError:
            raise TypeError("Argument 'inputs' must be an iterable.")
        
        na_inputs = np.array(inputs)
        
        #https://stackoverflow.com/a/33043793
        try:
            are_all_numeric = np.isfinite(na_inputs).all()
        except TypeError:
            raise ValueError(
                "Argument 'inputs' must be an iterable of numeric elements."
                )
        
        if not are_all_numeric:
            raise ValueError(
                "Argument 'inputs' must be an iterable of finite numeric elements."
                )
        
        #the weighted sum of the inputs through the activation function
        return self._activation_function( 
            np.sum(np.dot(na_inputs, self._weights)) 
            )

    def _relu(self, input : float):
        if not isinstance(input, numbers.Real):
            raise TypeError("Argument 'input' must be a real number.")

        return max([0, input])
    
n = Neuron(3)
#n.forward_pass([1, 2, 3, 4])
print("--")
#n.forward_pass("test")
#n.forward_pass(["a", "b", "c"])
#n.forward_pass([1, 2, "c"])
#n.forward_pass([1, 2, np.inf])
#n.forward_pass([1, 2, np.nan])
#n.forward_pass([1, 2, None])
#n.forward_pass([1, 2, np.newaxis])