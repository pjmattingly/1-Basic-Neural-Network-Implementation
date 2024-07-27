import pytest
from init_the_network.Neuron import Neuron
import numpy as np
import math

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

    def test_correct(self):
        Neuron(1)

    def test_seed_correct(self):
        Neuron(1, 0)

    def test_seed_bsd_input(self):
        with pytest.raises(TypeError):
            Neuron(1, list())
    
class Test_relu:
    def test_bad_input(self, inst):
        with pytest.raises(TypeError):
            inst._relu("some bad input")

class Test_forward_pass:
    def test_string_input(self, inst):
        with pytest.raises(TypeError):
            inst.forward_pass("str")

    def test_noniterable_input(self, inst):
        with pytest.raises(TypeError):
            inst.forward_pass(0)

    def test_nonnumeric_input_1(self, inst):
        with pytest.raises(TypeError):
            inst.forward_pass(["a", "b", "c"])

    def test_nonnumeric_input_2(self, inst):
        with pytest.raises(TypeError):
            inst.forward_pass([1, 2, "c"])

    def test_special_nonnumeric_input_1(self, inst):
        with pytest.raises(ValueError):
            inst.forward_pass([1, 2, np.inf])

    def test_special_nonnumeric_input_2(self, inst):
        with pytest.raises(ValueError):
            inst.forward_pass([1, 2, np.nan])

    def test_special_nonnumeric_input_3(self, inst):
        with pytest.raises(TypeError):
            inst.forward_pass([1, 2, None])

    def test_special_nonnumeric_input_4(self, inst):
        with pytest.raises(TypeError):
            inst.forward_pass([1, 2, np.newaxis])

            #see: https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis

    def test_wrong_size_inputs(self, inst):
        with pytest.raises(ValueError):
            inst.forward_pass([1, 2, 3, 4])

    def test_forward_pass_math(self, inst):
        #sanity check in case there a change in the way the forward pass is done \
        #in the future
        inputs = [1, 1, 1]
        test_out = inst._activation_function(np.dot(np.array(inputs), inst._weights) + inst._bias)
        obv_out = inst.forward_pass(inputs)

        #see: https://stackoverflow.com/a/68763927
        assert math.isclose(test_out, obv_out, rel_tol=.01)

class Test_derivative_of_activation_function:
    def test_bad_input(self, inst):
        with pytest.raises(TypeError):
            inst.derivative_of_activation_function("some bad input")

    def test_postive_input(self, inst):
        assert inst.derivative_of_activation_function(1) == 1

    def test_not_postive_input(self, inst):
        assert inst.derivative_of_activation_function(-1) == 0