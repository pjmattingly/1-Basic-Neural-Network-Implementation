import pytest
from init_the_network.Network import Network
import numpy as np

import random
from typing import List, Tuple

@pytest.fixture
def inst():
    return Network([2, 3, 2])

@pytest.fixture
def mock_data(seed: int = 0, num_samples: int = 1000, sigma: int=10) -> List[Tuple[float, float]]:
    """
    Generate mock data based on a linear equation with added noise.

    Parameters:
        seed (int): Seed for random number generator to ensure reproducibility.
        num_samples (int): Number of data samples to generate.
        sigma (float): Standard deviation of the Gaussian noise added to the data.

    Returns:
        List[Tuple[float, float]]: Generated noisy data samples.
    """

    random.seed(seed)

    #make a linear equation
    def linear_equation(x: float) -> float:
        slope = random.randint(1, 10)
        intercept = random.randint(1, 10)
        
        return slope * x + intercept

    # Generate data from the linear equation, sampled over the range [-100, 100]
    xs = [random.uniform(-100, 100) for _ in range(num_samples)]
    ys = [linear_equation(x) for x in xs]
    data = list(zip(xs, ys))

    # Add noise to the data to simulate real data
    noisy_data = [
        (random.normalvariate(x, sigma), random.normalvariate(y, sigma))
        for x, y in data
    ]
    
    return noisy_data