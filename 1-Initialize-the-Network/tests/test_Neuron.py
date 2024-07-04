import pytest
from init_the_network.Neuron import Neuron

@pytest.fixture
def inst():
    return Neuron(3)

class Test__init__:
    def test_bad_input_not_int(self):
        with pytest.raises(TypeError):
            Neuron("some bad input")

    def test_bad_input_not_positive_int(self):
        with pytest.raises(ValueError):
            Neuron(-1)

class Test_relu:
    def test_bad_input(self, inst):
        with pytest.raises(TypeError):
            #Neuron._relu("some bad input")
            inst._relu("some bad input")