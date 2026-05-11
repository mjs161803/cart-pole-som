import numpy as np


class SensorSOM:
    def __init__(self, input_dimension: int, output_dimension: int):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._weights = np.random.rand(output_dimension, input_dimension)
        self._input = np.zeros(input_dimension)
        self._output = np.zeros(output_dimension)

    @property
    def input_dimension(self):
        return self._input_dimension

    @property
    def output_dimension(self):
        return self._output_dimension

    @property
    def weights(self):
        return self._weights

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value: np.ndarray):
        self._input = np.asarray(value)

    @property
    def output(self):
        return self._output

    def step(self) -> np.ndarray:
        """Compute the output activations for the current input."""
        distances = np.linalg.norm(self._weights - self._input, axis=1)
        self._output = distances
        return self._output
