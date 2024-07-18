#allow for relative importing for pytest, and absolute importing when running directly
try:
    from .Layer import Layer
except ImportError:
    from Layer import Layer
from typing import List, Dict, Any
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

    def _check_data(self, data: Dict[str, Any]) -> None:
        """
        Perform sanity checks on the training data.

        Parameters:
            data (Dict[str, Any]): Data dictionary containing 'x' and 'y'.

        Raises:
            TypeError: If 'data' does not have the required structure or contains 
                non-numeric elements.
            ValueError: If 'data' contains inconsistent lengths, empty datasets, 
                non-finite elements, or duplicate entries.
        """

        if not callable(getattr(data, "get", None)):
            raise TypeError( "Data object should contain a method 'get' to fetch a set \
                            of data." )
        
        if data.get("x") is None:
            raise TypeError( "Data should contain a key of 'x'." )

        if data.get("y") is None:
            raise TypeError( "Data should contain a key of 'y'." )
        
        x_data = data["x"]
        y_data = data["y"]
        
        if not len(x_data) == len(y_data):
            raise ValueError("Data-sets 'x' and 'y' should be the same size.")
        
        if len(x_data) == 0 or len(y_data) == 0:
            raise ValueError("Data-sets 'x' or 'y' should not be empty.")
        
        def is_iter(x):
            try:
                iter(x)
            except TypeError:
                return False
            return True

        #the data-sets should be either all iterables, or all finite numeric elements
        #If all the items in the data-sets are iterables, then their lengths should all 
        # be the same.
        #And, no iterable in any data-set should be empty.
        #Otherwise, if they are all finite numeric elements, then there's no further 
        # checking needed
        #But, if there's a mix of iterables and finite numeric elements
        # then raise to the caller
        x_is_iterable = [is_iter(_) for _ in x_data]
        if all( x_is_iterable ):
            #if they are all iterables, then their lengths should all be the same
            x_lengths = {len(x_sample) for x_sample in x_data}
            if len(x_lengths) != 1:
                raise ValueError(
                    "All samples in data-set 'x' should have the same dimension."
                    )
            
            #check for empty iterables
            if 0 in x_lengths:
                raise ValueError(
                    "No samples in data-set 'x' should be empty."
                    )
        else:
            #check for a mix of iterables and finite numeric elements
            if len(set(x_is_iterable)) != 1:
                raise TypeError("Dataset 'x' should either be entirely numeric or \
                                entirely made up of iterables.")
            
            
        y_is_iterable = [is_iter(_) for _ in y_data]
        if all( y_is_iterable ):
            #if they are all iterables, then their lengths should all be the same
            y_lengths = {len(y_sample) for y_sample in y_data}
            if len(y_lengths) != 1:
                raise ValueError(
                    "All samples in data-set 'y' should have the same dimension."
                    )
            
            #check for empty iterables
            if 0 in y_lengths:
                raise ValueError(
                    "No samples in data-set 'y' should be empty."
                    )
        else:
            #check for a mix of iterables and finite numeric elements
            if len(set(y_is_iterable)) != 1:
                raise TypeError("Dataset 'y' should either be entirely numeric or \
                                entirely made up of iterables.")

        na_x = np.array(x_data)
        na_y = np.array(y_data)

        if not np.issubdtype(na_x.dtype, np.number):
            raise TypeError("Data-set 'x' should contain only numeric elements.")
        if not np.isfinite(na_x).all():
            raise ValueError(
                "Data-set 'x' should contain only finite numeric elements."
                )
        
        if not np.issubdtype(na_y.dtype, np.number):
            raise TypeError("Data-set 'y' should contain only numeric elements.")
        if not np.isfinite(na_y).all():
            raise ValueError(
                "Data-set 'y' should contain only finite numeric elements."
                )

        #checking for duplicates; see: https://stackoverflow.com/q/11528078
        sorted_x = np.sort(x_data, axis=None)
        if any(sorted_x[1:] == sorted_x[:-1]):
            raise ValueError(
                "Found duplicate entries in data-set 'x', please clean the data and \
                    try again."
                )
        
        sorted_y = np.sort(y_data, axis=None)
        if any(sorted_y[1:] == sorted_y[:-1]):
            raise ValueError(
                "Found duplicate entries in data-set 'y', please clean the data and \
                    try again."
                )
        
Network._check_data(None, {"x": [1, 2, 3, 4], "y": [1, 2, 3, 4]})
#Network._check_data(None, {"x": [[1], [2], [3], [4]], "y": [[1], [2], [3], [4]]})
#Network._check_data(None, {"x": [[np.nan], [2], [3], [4]], "y": [[1], [2], [3], [4]]})
#Network._check_data(None, {"x": [["bad"], [2], [3], [4]], "y": [[1], [2], [3], [4]]})
#Network._check_data(None, {"x": [[1], [2], [3], [4]], "y": [["bad"], [2], [3], [4]]})
Network._check_data(None, {"x": [[1], [2], [3], [4]], "y": [[1], [2], [3], []]})
#Network._check_data(None, {"x": [[], [], [], []], "y": [[], [], [], []]})