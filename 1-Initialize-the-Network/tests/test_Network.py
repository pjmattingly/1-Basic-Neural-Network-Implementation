import pytest
from init_the_network.Network import Network
import numpy as np

@pytest.fixture
def inst():
    return Network([3, 3, 1])

class Test__init__:
    def test_inputs_bad_input_not_iter(self):
        with pytest.raises(TypeError):
            Network(3)

    def test_inputs_bad_input_empty(self):
        with pytest.raises(TypeError):
            Network([])

    def test_inputs_bad_input_not_int(self):
        with pytest.raises(ValueError):
            Network([.5])

    def test_inputs_bad_input_neg_int(self):
        with pytest.raises(ValueError):
            Network([-1])

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