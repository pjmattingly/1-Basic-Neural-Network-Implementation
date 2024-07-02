import random

class Neuron:
    def __init__(self, inputs : int):
        if (type(inputs) is not int) or (inputs < 0):
            raise TypeError("Argument 'inputs' should be positive integer.")
        
        self._weights = [random.random() for x in range(inputs)]
        self._bias = random.random()