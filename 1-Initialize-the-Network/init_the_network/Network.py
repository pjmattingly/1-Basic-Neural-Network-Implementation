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