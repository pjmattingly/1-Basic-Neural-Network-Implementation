import random

class Neuron:
    def __init__(self, inputs):
        self._weights = [random.random() for x in range(inputs)]
        self._bias = random.random()