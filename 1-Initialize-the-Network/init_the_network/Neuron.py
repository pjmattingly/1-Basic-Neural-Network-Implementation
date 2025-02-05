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

    def __init__(self, inputs : int, 
                 seed : None | int | float | str | bytes | bytearray = None):
        """
        Initialize a Neuron with a specified number of inputs.

        Parameters:
            inputs (int): The number of input connections to the neuron.
            seed (None | int | float | str | bytes | bytearray): \
                Seed for random number generation.

        Raises:
            TypeError: If 'inputs' is not an integer or \
                if 'seed' is of an incorrect type.
            ValueError: If 'inputs' is a negative integer.
        """

        if not isinstance(inputs, int):
            raise TypeError("Argument 'inputs' must be an integer.")
        if inputs < 0:
            raise ValueError("Argument 'inputs' must be a non-negative integer.")
        
        if seed is not None:
            if not isinstance(seed, (int, float, str, bytes, bytearray)):
                raise TypeError("Argument 'seed' must be an integer, real number, \
                                string, bytes, None, or type `bytearray`.")

            np.random.seed(seed)
        
        self._weights = np.random.rand(inputs)
        self._bias = 0
        self._activation_function = self._relu #TODO, more activation functions

    def forward_pass(self, inputs : List[float]) -> float:
        """
        Compute the forward pass for the neuron.

        Parameters:
            inputs (List[float]): Input values for the neuron.

        Returns:
            float: Output of the neuron after applying the activation function.

        Raises:
            TypeError: If 'inputs' is not an iterable of numeric elements.
            ValueError: If 'inputs' is not of the expected length or contains \
                non-finite numeric elements.
        """

        #https://stackoverflow.com/a/1952481
        try:
            iter(inputs)
        except TypeError:
            raise TypeError(
                "Argument 'inputs' must be an iterable of numeric elements."
                )

        if len(inputs) != len(self._weights):
            raise ValueError(
                "Argument 'inputs' must be an iterable of the same size as the \
                Neuron's weights.")
        
        if isinstance(inputs, str):
            raise TypeError(
                "Argument 'inputs' must be an iterable of numeric elements."
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
        
        #the weighted sum of the inputs with the bias through the activation function
        return self._activation_function(np.dot(na_inputs, self._weights) + self._bias)

    def _relu(self, x : float) -> float:
        """
        Apply the ReLU activation function.

        Parameters:
            x (float): Input value to the activation function.

        Returns:
            float: Output of the ReLU activation function.

        Raises:
            TypeError: If 'x' is not a real number.
        """

        if not isinstance(x, numbers.Real):
            raise TypeError("Argument 'x' must be a real number.")

        return max(0, x)
    
    def derivative_of_activation_function(self, x: float) -> float:
        """
        Compute the derivative of the ReLU activation function.

        Parameters:
            x (float): Input value to the activation function.

        Returns:
            float: Derivative of the ReLU function. Returns 1 if x > 0, otherwise returns 0.
        """

        if not isinstance(x, numbers.Real):
            raise TypeError("Argument 'x' must be a real number.")

        if x > 0:
            return 1
        else:
            # The derivative of ReLU is not defined at x = 0; conventionally, \
            # it is set to 0.
            # Reference: https://stackoverflow.com/a/76396054
            return 0