# imports
import logging
import sys
import argparse
import tempfile
import torch
import numpy as np
import pytorch_lightning as pl
from brontes import Brontes
from function_approximation_nn.data import BoundedPointsDataset
from function_approximation_nn.nn import Approximator

# logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('sin-example')

# # configure argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-n', '--model_name', type=str,
    help='model name.', default='model',
    required=False
)
parser.add_argument(
    '--training_points',
    type=int,
    help='number of training points.',
    default=100,
    required=False
)
parser.add_argument(
    '--validation_points',
    type=int,
    help='number of validation points.',
    default=100,
    required=False
)
parser.add_argument(
    '-s', '--seed', type=int,
    help='seed for reproducible results.', default=42,
    required=False
)
parser.add_argument(
    '-b', '--batch_size', type=int,
    help='batch size.', default=10,
    required=False
)
parser.add_argument(
    '--epochs', type=int,
    help='epochs.', default=50,
    required=False
)
parser.add_argument(
    '-l', '--learning_rate', type=float,
    help='learning rate.', default=1e-5,
    required=False
)


def main(arguments):
    """
    Train approximator with brontes.
    Args:
        arguments (Namespace): parsed arguments.
    """
    # create aliases
    MODEL_NAME = arguments.model_name
    TRAINING_POINTS = arguments.training_points
    VALIDATION_POINTS = arguments.validation_points
    SEED = arguments.seed
    BATCH_SIZE = arguments.batch_size
    EPOCHS = arguments.epochs
    LEARNING_RATE = arguments.learning_rate

    # set the seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # data loaders for the MNIST dataset
    dataset_loaders = {
        'train':
            torch.utils.data.DataLoader(
                BoundedPointsDataset(
                    number_of_points=TRAINING_POINTS,
                    input_dimension=1,
                    output_dimension=1,
                    function=torch.sin
                ),
                batch_size=BATCH_SIZE,
                shuffle=True
            ),
        'val':
            torch.utils.data.DataLoader(
                BoundedPointsDataset(
                    number_of_points=VALIDATION_POINTS,
                    input_dimension=1,
                    output_dimension=1,
                    function=torch.sin
                ),
                batch_size=BATCH_SIZE,
                shuffle=True
            )
    }

    # definition of base model
    model = Approximator(1, 1)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5  # standard value
    )

    # brontes model is initialized with base_model, optimizer, loss,
    # data_loaders. Optionally a dict of metrics functions and a
    # batch_fn applied to every batch can be provided.
    brontes_model = Brontes(
        model=model,
        loss=torch.nn.MSELoss(),
        data_loaders=dataset_loaders,
        optimizers=optimizer,
        training_log_interval=10
    )

    # finally, train the model
    trainer = pl.Trainer(max_nb_epochs=EPOCHS)
    trainer.fit(brontes_model)

    # save the model
    saved_model = f'{tempfile.mkdtemp()}/{MODEL_NAME}.pt'
    logger.info(f'storing model in: {saved_model}')
    torch.save(brontes_model.model, saved_model)


if __name__ == "__main__":
    main(arguments=parser.parse_args())
