#allow for relative importing for pytest, and absolute importing when running directly
try:
    from .Neuron import Neuron
except ImportError:
    from Neuron import Neuron
from typing import List
import numpy as np

class Layer:
    """
    A layer in a neural network consisting of multiple neurons.

    Attributes:
        _num_inputs (int): Number of inputs to each neuron in the layer.
        _neurons (List[Neuron]): List of neurons in the layer.
    """

    def __init__(self, inputs : int, size : int, 
                 seed : None | int | float | str | bytes | bytearray = None):
        """
        Initialize a layer of a neural network.

        Parameters:
            inputs (int): Number of input connections to each neuron.
            size (int): Number of neurons in the layer.
            seed (None | int | float | str | bytes | bytearray): \
                Seed for random number generation.

        Raises:
            TypeError: If 'inputs' or 'size' is not an integer, \
                or if 'seed' is of an incorrect type.
            ValueError: If 'inputs' or 'size' is a negative integer.
        """

        if not isinstance(inputs, int):
            raise TypeError("Argument 'inputs' must be an integer.")
        if inputs < 0:
            raise ValueError("Argument 'inputs' must be a non-negative integer.")
        
        if not isinstance(size, int):
            raise TypeError("Argument 'size' must be an integer.")
        if size < 0:
            raise ValueError("Argument 'size' must be a non-negative integer.")
        
        if seed is not None:
            if not isinstance(seed, (int, float, str, bytes, bytearray)):
                raise TypeError("Argument 'seed' must be an integer, real number, \
                                string, bytes, None, or type `bytearray`.")
        
        self._num_inputs = inputs
        self._neurons = [Neuron(self._num_inputs, seed) for _ in range(size)]

    def forward_pass(self, inputs : List[float]) -> List[float]:
        """
        Perform a forward pass through the layer.

        Parameters:
            inputs (List[float]): Input values for the neurons in the layer.

        Returns:
            List[float]: Output values from each neuron in the layer.

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
        
        if len(inputs) != self._num_inputs:
            raise ValueError(
                f"Argument 'inputs' must be an iterable of size {self._num_inputs}."
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

        return [neuron.forward_pass(inputs) for neuron in self._neurons]
    
    def backward_pass(self):
        #PLACEHOLDER
        pass

    def get_size(self):
        return len(self._neurons)