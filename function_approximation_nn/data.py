"""Data module."""
import torch
from torch.utils.data.dataset import Dataset


class BoundedPointsDataset(Dataset):
    """Bounded points dataset."""

    def __init__(
        self,
        number_of_points,
        input_dimension,
        output_dimension,
        function,
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
            function (function): a function handling torch.tensor:
                input_dimension -> output_dimension.
            lower_bound (float): lower bound per dimension.
            upper_bound (float): upper bound per dimension.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
        """
        self.number_of_points = number_of_points
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.function = function
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.range = self.upper_bound - self.lower_bound
        self.device = device
        self.x = torch.rand(
            self.number_of_points, self.input_dimension
        )*self.range
        self.fx = self.function(self.x)

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
        return self.x, self.fx
