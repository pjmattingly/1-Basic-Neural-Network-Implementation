import pytest
from init_the_network.Neuron import Neuron

class Test__init__:
    def test_bad_input_not_int(self):
        with pytest.raises(TypeError):
            Neuron("some bad input")

    def test_bad_input_not_positive_int(self):
        with pytest.raises(ValueError):
            Neuron(-1)