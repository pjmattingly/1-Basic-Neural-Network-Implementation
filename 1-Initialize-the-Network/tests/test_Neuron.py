import pytest
from init_the_network.Neuron import Neuron

class Test__init__:
    def test_bad_input_not_int(self):
        with pytest.raises(TypeError, match=r".*positive integer.*"):
            Neuron("some bad input")

    def test_bad_input_not_positive_int(self):
        with pytest.raises(TypeError, match=r".*positive integer.*"):
            Neuron(-1)