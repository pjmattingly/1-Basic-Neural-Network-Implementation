import pytest
from init_the_network.Network import Network

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