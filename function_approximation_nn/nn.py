"""Neural netowrk modeule."""
import torch.nn as nn
from collections import OrderedDict


def create_dense_layer(
    input_size, output_size, activation_fn=nn.ReLU(), dropout=.5
):
    """
    Create a dense layer.
    Args:
        - input_size (int): size of the input.
        - output_size (int): size of the output.
        - activation_fn (an activation): activation function.
            Defaults to ReLU.
        - dropout: dropout rate. Defaults to 0.5.
    Returns:
        a nn.Sequential.
    """
    return nn.Sequential(
        OrderedDict(
            [
                ('linear', nn.Linear(input_size, output_size)),
                ('activation_fn', activation_fn),
                ('dropout', nn.Dropout(p=dropout)),
            ]
        )
    )


class Approximator(nn.Module):
    """Approximation function network."""

    def __init__(
        self,
        input_dimension,
        output_dimension,
        hidden_units=[32, 32],
        activation_fn=nn.ReLU(),
        dropout=.5
    ):
        """
        Initialize the approximation function network.

        Args:
            input_dimension (int): input dimension.
            output_dimension (int): output dimension.
            hidden_units (list): hidden units. Defaults to [32, 32].
            activation_fn (nn.Module): activation function.
                Defaults to nn.Relu.
            dropout (float): dropout rate. Defaults to 0.5.
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.dimensions = [self.input_dimension] + hidden_units
        self.output_dimension = output_dimension
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.stacked_layers = nn.Sequential(
            *[
                create_dense_layer(
                    input_size, output_size, dropout=self.dropout
                )
                for input_size, output_size in zip(
                    self.dimensions, self.dimensions[1:]
                )
            ]
        )
        self.output = nn.Linear(self.dimensions[-1], self.output_dimension)

    def forward(self, x):
        """
        Apply the network.

        Args:
            x (torch.tensor): a batch.

        Returns:
            a torch.tensor
        """
        return self.output(self.stacked_layers(x))
