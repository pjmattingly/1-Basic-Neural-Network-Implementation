from Neuron import Neuron

class Layer:
    """
    A layer in a neural network consisting of multiple neurons.

    Attributes:
        _num_inputs (int): Number of inputs to each neuron in the layer.
        _neurons (List[Neuron]): List of neurons in the layer.
    """

    def __init__(self, inputs : int, size : int):
        """
        Initialize a neural layer.

        Parameters:
            inputs (int): Number of input connections to each neuron.
            size (int): Number of neurons in the layer.

        Raises:
            TypeError: If 'inputs' or 'size' is not an integer.
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
        
        self._num_inputs = inputs
        self._neurons = [Neuron(self._num_inputs) for _ in range(size)]