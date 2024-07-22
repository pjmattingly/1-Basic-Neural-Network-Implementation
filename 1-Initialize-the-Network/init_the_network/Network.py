#allow for relative importing for pytest, and absolute importing when running directly
try:
    from .Layer import Layer
except ImportError:
    from Layer import Layer
from typing import List, Dict, Any
from collections.abc import Iterable
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
        learning_rate = self._check_and_set_learning_rate(learning_rate)
        epochs = self._check_and_set_epochs(epochs)

        for _ in range(epochs):
            _x = random.shuffle(data[x])
            _y = random.shuffle(data[y])

            outputs = [self.forward_pass(x) for x in _x]

            self._log_accuracy(results, _y)

            losses = self._loss_calc(results, _y)

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

        self._is_dict_like(data)

        self._has_x_and_y_keys(data)
        
        x_data = data.get("x")
        y_data = data.get("y")

        self._is_iterable(x_data, "x")
        self._is_iterable(y_data, "y")

        self._x_and_y_not_empty(x_data, y_data)
        
        self._x_and_y_have_equal_size(x_data, y_data)

        self._all_elements_have_the_same_shape(x_data, "x")
        self._all_elements_have_the_same_shape(y_data, "y")

        self._has_only_numeric_elements(x_data, "x")
        self._has_only_numeric_elements(y_data, "y")

        self._has_only_finite_elements(x_data, "x")
        self._has_only_finite_elements(y_data, "y")

        self._does_not_have_duplicate_elements(x_data, "x")
        self._does_not_have_duplicate_elements(y_data, "y")
    
    def _is_dict_like(self, data: Dict[str, Any]) -> None:
        """Check if the data is dictionary-like."""

        if not callable(getattr(data, "get", None)):
            raise TypeError( "Data object should contain a method 'get' to fetch a set \
                            of data." )
        
    def _has_x_and_y_keys(self, data: Dict[str, Any]) -> None:
        """Check if the data contains 'x' and 'y' keys."""

        if data.get("x") is None:
            raise TypeError( "Data should contain a key of 'x'." )

        if data.get("y") is None:
            raise TypeError( "Data should contain a key of 'y'." )
        
    def _is_iterable(self, x: Any, label: str) -> None:
        """Check if the object is iterable."""

        try:
            iter(x)
        except TypeError:
            raise TypeError(f"Dataset '{label}' should be an iterable.")
        
    #TODO, modify error message to be more generic, as this called in Network._loss_calc
    def _x_and_y_not_empty(self, x: Iterable[Any], y: Iterable[Any]) -> None:
        """Check if datasets 'x' and 'y' are non-empty."""

        if len(x) == 0 or len(y) == 0:
            raise ValueError("Data-sets 'x' or 'y' should not be empty.")

    #TODO, modify error message to be more generic, as this called in Network._loss_calc
    def _x_and_y_have_equal_size(self, x: Iterable[Any], y: Iterable[Any]) -> None:
        """Check if datasets 'x' and 'y' have equal size."""
        
        if not len(x) == len(y):
            raise ValueError("Data-sets 'x' and 'y' should be the same size.")

    def _all_elements_have_the_same_shape(self, x: Iterable[Any], label: str) -> None:
        """Check if all elements in the dataset have the same shape."""

        try:
            np.array(x)
        except ValueError:
            raise ValueError(
                    f"All elements in data-set '{label}' should have the same \
                        dimension."
                    )
    
    def _has_only_numeric_elements(self, x: Iterable[Any], label: str) -> None:
        """Check if all elements in the dataset are numeric."""

        if not np.issubdtype(np.array(x).dtype, np.number):
            raise TypeError(f"Data-set '{label}' should contain only numeric elements.")
        
    def _has_only_finite_elements(self, x: Iterable[Any], label: str) -> None:
        """Check if all elements in the dataset are finite."""

        if not np.isfinite(np.array(x)).all():
            raise ValueError(
                f"Data-set '{label}' should contain only finite numeric elements."
                )

    def _does_not_have_duplicate_elements(self, x: Iterable[Any], label: str) -> None:
        """Check if there are no duplicate elements in the dataset."""

        #checking for duplicates; see: https://stackoverflow.com/q/11528078
        sorted_x = np.sort(x, axis=None)
        if any(sorted_x[1:] == sorted_x[:-1]):
            raise ValueError(
                f"Found duplicate elements in data-set '{label}', please clean the data\
                    and try again."
                )
        
    def _check_and_set_learning_rate(self, learning_rate: float) -> float:
        """
        Validate and set the learning rate.

        Parameters:
            learning_rate (float): The learning rate to be validated and set.

        Returns:
            float: The validated learning rate.

        Raises:
            TypeError: If the learning rate is not numeric.
            ValueError: If the learning rate is not between 0 and 1 inclusive.
        """
        
        if not isinstance(learning_rate, (int, float)):
            raise TypeError("The learning rate should be numeric.")
        
        if not 0 <= learning_rate <= 1:
            raise ValueError("The learning rate should be great than or equal to 0 and \
                            less than or equal to 1.")
        
        return float(learning_rate)
    
    def _check_and_set_epochs(self, epochs: int) -> int:
        """
        Validate and set the number of epochs.

        Parameters:
            epochs (int): The number of epochs to be validated and set.

        Returns:
            int: The validated number of epochs.

        Raises:
            TypeError: If the 'epochs' parameter is not an integer.
            ValueError: If the 'epochs' parameter is less than zero.
        """
        
        if not isinstance(epochs, int):
            raise TypeError("The 'epochs' parameter should be an integer.")
        
        if not 0 <= epochs:
            raise ValueError("The 'epochs' parameter should be greater than or equal \
                             to zero.")
        
        return epochs
    
    def _loss_calc(self, observed: Iterable[Any], target: Iterable[Any]) -> float:
        """
        Validate observed values and calculate the Mean Squared Error (MSE).

        Parameters:
            observed (Iterable[Any]): Observed values from the network.
            target (Iterable[Any]): Target values for comparison.

        Returns:
            float: The calculated Mean Squared Error.

        Raises:
            TypeError, ValueError: If validation checks fail on the observed values.
        """

        #Variable `target` was checked with Network._check_data, but the output values 
        # from the Network (observed values) need to be checked before processing.
        self._is_iterable(observed, "observed")
        self._is_iterable(target, "target")
        self._x_and_y_not_empty(observed, target)
        self._x_and_y_have_equal_size(observed, target)
        self._all_elements_have_the_same_shape(observed, "observed")
        self._all_elements_have_the_same_shape(target, "target")
        self._has_only_numeric_elements(observed, "observed")
        self._has_only_numeric_elements(target, "target")
        self._has_only_finite_elements(observed, "observed")
        self._has_only_finite_elements(target, "target")

        return self._calculate_MSE(observed, target)

    def _calculate_MSE(self, y: Iterable[Any], y_bar: Iterable[Any]) -> float:
        """
        Calculate the Mean Squared Error (MSE) between two sets of values.

        Parameters:
            y (Iterable[Any]): The predicted values.
            y_bar (Iterable[Any]): The true values.

        Returns:
            float: The Mean Squared Error.
        """

        #see: https://www.datacamp.com/tutorial/loss-function-in-machine-learning
        #and: https://stackoverflow.com/a/47374870
        return np.square(np.subtract(np.array(y), np.array(y_bar))).mean()