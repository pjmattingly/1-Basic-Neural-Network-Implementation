import pytest
from init_the_network.Layer import Layer

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