import pytest
from init_the_network.Layer import Layer
import numpy as np

@pytest.fixture
def inst():
    return Layer(3, 3)

class Test__init__:
    def test_inputs_bad_input_not_int(self):
        with pytest.raises(TypeError):
            Layer("some bad input", 3)

    def test_inputs_bad_input_not_positive(self):
        with pytest.raises(ValueError):
            Layer(-1, 3)

    def test_size_bad_input_not_int(self):
        with pytest.raises(TypeError):
            Layer(3, "some bad input")

    def test_size_bad_input_not_positive(self):
        with pytest.raises(ValueError):
            Layer(3, -1)

    def test_correct(self):
        Layer(3, 3, seed=0)
        assert True

    def test_correct(self):
        with pytest.raises(TypeError):
            Layer(3, 3, seed=list())
    
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