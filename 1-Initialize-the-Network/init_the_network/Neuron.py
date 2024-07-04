import random
from typing import List
import numbers

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
        
        self._weights = [random.random() for _ in range(inputs)]
        self._bias = random.random()
        self._activation_function = self._relu

    def forward_pass(inputs : List[float]) -> float:
        pass

    def _relu(_in : float):
        if not isinstance(_in, numbers.Real):
            raise TypeError("Argument '_in' must be a real number.")
        
        return max([0, _in])