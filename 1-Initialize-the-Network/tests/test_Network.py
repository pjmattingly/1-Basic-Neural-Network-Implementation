import pytest
from init_the_network.Network import Network
import numpy as np
from types import SimpleNamespace

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

class Test_is_dict_like:
    def test_correct(self, inst):
        inst._is_dict_like(dict())
        assert True

    def test_property_not_callable(self, inst):
        test_var = SimpleNamespace()
        test_var.get = "test"
        with pytest.raises(TypeError):
            inst._is_dict_like(test_var)

class Test_has_x_and_y_keys:
    def test_correct(self, inst):
        test_var = dict()
        test_var["x"] = True
        test_var["y"] = True
        inst._has_x_and_y_keys(test_var)
        assert True

    def test_no_x(self, inst):
        test_var = dict()
        #test_var["x"] = True
        test_var["y"] = True
        with pytest.raises(TypeError):
            inst._has_x_and_y_keys(test_var)

    def test_no_y(self, inst):
        test_var = dict()
        test_var["x"] = True
        #test_var["y"] = True
        with pytest.raises(TypeError):
            inst._has_x_and_y_keys(test_var)

class Test_is_iterable:
    def test_correct(self, inst):
        inst._is_iterable(list(), "test")
        assert True

    def test_correct_with_string(self, inst):
        inst._is_iterable("test", "test")
        assert True

    def test_not_iterable(self, inst):
        with pytest.raises(TypeError):
            inst._is_iterable(0, "test")

class Test_x_and_y_have_equal_size:
    def test_correct(self, inst):
        inst._x_and_y_have_equal_size(list(), list())
        assert True

    def test_different_sizes(self, inst):
        with pytest.raises(ValueError):
            inst._x_and_y_have_equal_size([0], list())

class Test_x_and_y_not_empty:
    def test_correct(self, inst):
        inst._x_and_y_not_empty([0], [0])
        assert True

    def test_first_empty(self, inst):
        with pytest.raises(ValueError):
            inst._x_and_y_not_empty([], [0])

    def test_second_empty(self, inst):
        with pytest.raises(ValueError):
            inst._x_and_y_not_empty([0], [])

    def test_both_empty(self, inst):
        with pytest.raises(ValueError):
            inst._x_and_y_not_empty([], [])

class Test_all_elements_have_the_same_shape:
    def test_correct(self, inst):
        inst._all_elements_have_the_same_shape([0, 0], "test")
        assert True

    def test_one_empty(self, inst):
        with pytest.raises(ValueError):
            inst._all_elements_have_the_same_shape([[], [0]], "test")

    def test_one_is_a_number(self, inst):
        with pytest.raises(ValueError):
            inst._all_elements_have_the_same_shape([0, [0]], "test")

    def test_different_sized_iterables(self, inst): 
        with pytest.raises(ValueError):
            inst._all_elements_have_the_same_shape([[0], [0, 0], [[0], 0]], "test")

class Test_has_only_numeric_elements:
    def test_correct(self, inst):
        inst._has_only_numeric_elements([0], "test")
        assert True

    def test_nonnumeric_input_1(self, inst):
        with pytest.raises(TypeError):
            inst._has_only_numeric_elements(["a", "b", "c"])

    def test_nonnumeric_input_2(self, inst):
        with pytest.raises(TypeError):
            inst._has_only_numeric_elements([1, 2, "c"])

    def test_nonnumeric_input_3(self, inst):
        with pytest.raises(TypeError):
            inst._has_only_numeric_elements([1, 2, None])

    def test_nonnumeric_input_4(self, inst):
        with pytest.raises(TypeError):
            inst._has_only_numeric_elements([1, 2, list()])

class Test_has_only_finite_elements:
    def test_correct(self, inst):
        inst._has_only_finite_elements([0], "test")
        assert True

    def test_special_nonnumeric_input_1(self, inst):
        with pytest.raises(TypeError):
            inst._has_only_finite_elements([1, 2, np.inf])

    def test_special_nonnumeric_input_2(self, inst):
        with pytest.raises(TypeError):
            inst._has_only_finite_elements([1, 2, np.nan])

    def test_special_nonnumeric_input_3(self, inst):
        with pytest.raises(TypeError):
            inst._has_only_finite_elements([1, 2, None])

    def test_special_nonnumeric_input_4(self, inst):
        with pytest.raises(TypeError):
            inst._has_only_finite_elements([1, 2, np.newaxis])

            #see: https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis

class Test_does_not_have_duplicate_elements:
    def test_correct(self, inst):
        inst._does_not_have_duplicate_elements([0, 1], "test")
        assert True

    def test_correct2(self, inst):
        inst._does_not_have_duplicate_elements([[0], [1]], "test")
        assert True

    def test_duplicates1(self, inst):
        with pytest.raises(ValueError):
            inst._does_not_have_duplicate_elements([0, 0], "test")

    def test_duplicates2(self, inst):
        with pytest.raises(ValueError):
            inst._does_not_have_duplicate_elements([[1], [1]], "test")

class Test_check_data:
    def test_correct(self, inst):
        test_var = dict()
        test_var["x"] = [0, 1]
        test_var["y"] = [[0], [1]]
        inst._check_data(test_var)
        assert True

    def test_property_not_callable(self, inst):
        test_var = SimpleNamespace()
        test_var.get = "test"
        with pytest.raises(TypeError):
            inst._check_data(test_var)

    def test_no_x(self, inst):
        test_var = dict()
        #test_var["x"] = True
        test_var["y"] = True
        with pytest.raises(TypeError):
            inst._check_data(test_var)

    def test_no_y(self, inst):
        test_var = dict()
        test_var["x"] = True
        #test_var["y"] = True
        with pytest.raises(TypeError):
            inst._check_data(test_var)

    def test_not_iterable_x(self, inst):
        test_var = dict()
        test_var["x"] = True
        test_var["y"] = list()

        with pytest.raises(TypeError):
            inst._check_data(test_var)

    def test_not_iterable_y(self, inst):
        test_var = dict()
        test_var["x"] = list()
        test_var["y"] = True

        with pytest.raises(TypeError):
            inst._check_data(test_var)

    def test_not_iterable_both(self, inst):
        test_var = dict()
        test_var["x"] = True
        test_var["y"] = True

        with pytest.raises(TypeError):
            inst._check_data(test_var)

    def test_first_empty(self, inst):
        test_var = dict()
        test_var["x"] = []
        test_var["y"] = [0]

        with pytest.raises(ValueError):
            inst._check_data(test_var)

    def test_second_empty(self, inst):
        test_var = dict()
        test_var["x"] = [0]
        test_var["y"] = []

        with pytest.raises(ValueError):
            inst._check_data(test_var)

    def test_both_empty(self, inst):
        test_var = dict()
        test_var["x"] = []
        test_var["y"] = []

        with pytest.raises(ValueError):
            inst._check_data(test_var)

    def test_different_sizes(self, inst):
        test_var = dict()
        test_var["x"] = [0]
        test_var["y"] = [0, 1]

        with pytest.raises(ValueError):
            inst._check_data(test_var)

    def test_one_is_a_number(self, inst):
        test_var = dict()
        test_var["x"] = [0, [0]]
        test_var["y"] = [0, 1]

        with pytest.raises(ValueError):
            inst._check_data(test_var)

    def test_different_sized_iterables(self, inst):
        test_var = dict()
        test_var["x"] = [[0], [0, 0], [[0], 0]]
        test_var["y"] = [0, 1, 2]

        with pytest.raises(ValueError):
            inst._check_data(test_var)

    def test_single_empty_element(self, inst):
        test_var = dict()
        test_var["x"] = [[0], [1], [2], [3], []]
        test_var["y"] = [0, 1, 2, 3, 4]

        with pytest.raises(ValueError):
            inst._check_data(test_var)

    def test_nonnumeric_input_1(self, inst):
        test_var = dict()
        test_var["x"] = ["a", "b", "c"]
        test_var["y"] = [0, 1, 2]

        with pytest.raises(TypeError):
            inst._check_data(test_var)

    def test_nonnumeric_input_2(self, inst):
        test_var = dict()
        test_var["x"] = [1, 2, "c"]
        test_var["y"] = [0, 1, 2]

        with pytest.raises(TypeError):
            inst._check_data(test_var)

    def test_nonnumeric_input_3(self, inst):
        test_var = dict()
        test_var["x"] = [1, 2, None]
        test_var["y"] = [0, 1, 2]

        with pytest.raises(TypeError):
            inst._check_data(test_var)

    def test_special_nonnumeric_input_1(self, inst):
        test_var = dict()
        test_var["x"] = [1, 2, np.inf]
        test_var["y"] = [0, 1, 2]

        with pytest.raises(ValueError):
            inst._check_data(test_var)

    def test_special_nonnumeric_input_2(self, inst):
        test_var = dict()
        test_var["x"] = [1, 2, np.nan]
        test_var["y"] = [0, 1, 2]

        with pytest.raises(ValueError):
            inst._check_data(test_var)

    def test_duplicates1(self, inst):
        test_var = dict()
        test_var["x"] = [0, 0]
        test_var["y"] = [0, 1]

        with pytest.raises(ValueError):
            inst._check_data(test_var)

    def test_duplicates2(self, inst):
        test_var = dict()
        test_var["x"] = [[1], [1]]
        test_var["y"] = [0, 1]

        with pytest.raises(ValueError):
            inst._check_data(test_var)

class Test_check_and_set_learning_rate:
    def test_correct(self, inst):
        inst._check_and_set_learning_rate(1)
        assert True

    def test_not_numeric_input(self, inst):
        with pytest.raises(TypeError):
            inst._check_and_set_learning_rate("q")

    def test_out_of_range_input(self, inst):
        with pytest.raises(ValueError):
            inst._check_and_set_learning_rate(2)

class Test_check_and_set_epochs:
    def test_correct(self, inst):
        inst._check_and_set_epochs(0)
        assert True

    def test_not_numeric_input(self, inst):
        with pytest.raises(TypeError):
            inst._check_and_set_epochs("q")

    def test_out_of_range_input(self, inst):
        with pytest.raises(ValueError):
            inst._check_and_set_epochs(-1)

class Test_loss_calc:
    def test_correct(self, inst):
        x = [0, 1]
        y = [[0], [1]]
        inst._loss_calc(x, y)
        assert True

    def test_not_iterable_x(self, inst):
        x = True
        y = list()

        with pytest.raises(TypeError):
            inst._loss_calc(x, y)

    def test_not_iterable_y(self, inst):
        x = list()
        y = True

        with pytest.raises(TypeError):
            inst._loss_calc(x, y)

    def test_not_iterable_both(self, inst):
        x = True
        y = True

        with pytest.raises(TypeError):
            inst._loss_calc(x, y)

    def test_first_empty(self, inst):
        x = []
        y = [0]

        with pytest.raises(ValueError):
            inst._loss_calc(x, y)

    def test_second_empty(self, inst):
        x = [0]
        y = []

        with pytest.raises(ValueError):
            inst._loss_calc(x, y)

    def test_both_empty(self, inst):
        x = []
        y = []

        with pytest.raises(ValueError):
            inst._loss_calc(x, y)

    def test_one_is_a_number(self, inst):
        x = [0, [0]]
        y = [0, 1]

        with pytest.raises(ValueError):
            inst._loss_calc(x, y)

    def test_different_sized_iterables(self, inst):
        x = [[0], [0, 0], [[0], 0]]
        y = [0, 1, 2]

        with pytest.raises(ValueError):
            inst._loss_calc(x, y)

    def test_single_empty_element(self, inst):
        x = [[0], [1], [2], [3], []]
        y = [0, 1, 2, 3, 4]

        with pytest.raises(ValueError):
            inst._loss_calc(x, y)

    def test_nonnumeric_input_1(self, inst):
        x = ["a", "b", "c"]
        y = [0, 1, 2]

        with pytest.raises(TypeError):
            inst._loss_calc(x, y)

    def test_nonnumeric_input_2(self, inst):
        x = [1, 2, "c"]
        y = [0, 1, 2]

        with pytest.raises(TypeError):
            inst._loss_calc(x, y)

    def test_nonnumeric_input_3(self, inst):
        x = [1, 2, None]
        y = [0, 1, 2]

        with pytest.raises(TypeError):
            inst._loss_calc(x, y)

    def test_special_nonnumeric_input_1(self, inst):
        x = [1, 2, np.inf]
        y = [0, 1, 2]

        with pytest.raises(ValueError):
            inst._loss_calc(x, y)

    def test_special_nonnumeric_input_2(self, inst):
        x = [1, 2, np.nan]
        y = [0, 1, 2]

        with pytest.raises(ValueError):
            inst._loss_calc(x, y)