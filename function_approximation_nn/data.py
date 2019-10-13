"""Data module."""
import torch
import numpy as np
from torch.utils.data.dataset import Dataset

# NOTE: functions sampling point in [0, 1) x [0, 1)
SAMPLING_FUNCTIONS = {
    'uniform':
        lambda number_of_points, dimension: np.random.
        rand(number_of_points, dimension),
    'grid':
        lambda number_of_points, dimension: np.hstack(
            [
                np.expand_dims(
                    np.linspace(0., 1., num=number_of_points, endpoint=False),
                    axis=1
                )
                for _ in range(dimension)
            ]
        )
}


class BoundedPointsDataset(Dataset):
    """Bounded points dataset."""

    def __init__(
        self,
        number_of_points,
        input_dimension,
        output_dimension,
        function,
        sampling='uniform',
        lower_bound=.0,
        upper_bound=1.,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """
        Initialize the bounded points dataset.
        Args:
            number_of_points (int): number of points.
            input_dimension (int): input dimension.
            output_dimension (int): output dimension.
            function (function): a function handling np.arrays.
                This function should accept 1-D arrays.
                It is applied to 1-D slices of arr along the specified axis.
            sampling (str): sampling schme.
                It can be chosen among: uniform, grid.
                Defaults to uniform.
            lower_bound (float): lower bound per dimension.
            upper_bound (float): upper bound per dimension.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
        """
        self.number_of_points = number_of_points
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.function = function
        self.sampling = sampling
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.range = self.upper_bound - self.lower_bound
        self.device = device
        self.sampling_fn = SAMPLING_FUNCTIONS.get(
            self.sampling,
            SAMPLING_FUNCTIONS['uniform']
        )
        self.x = self.sampling_fn(
            self.number_of_points, self.input_dimension
        ) * self.range
        self.fx = np.apply_along_axis(
            func1d=self.function, axis=1, arr=self.x
        )

    def __len__(self):
        """
        Get number of pairs.

        Returns:
            the number of pairs.
        """
        return self.number_of_points

    def __getitem__(self, index):
        """
        Get a point and the evaluation.
        Args:
            index (int): the index of a point.
        Returns:
            a tuple with two torch.Tenors:
                - the first containing x.
                - the second containing f(x).
        """
        return (
            torch.tensor(self.x[index], device=self.device).float(),
            torch.tensor(self.fx[index], device=self.device).float()
        )
