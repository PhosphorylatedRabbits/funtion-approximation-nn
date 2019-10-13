# function-approximation-nn

## development setup

Create and activate the environment:

```sh
conda env create -f conda.yml
conda activate function-approximation-nn
```

Install in editable mode:

```sh
pip install -e .
```

## run the approximation

After installation the commmand `approximate` is made available:

```console
usage: approximate [-h] [--input_dimension INPUT_DIMENSION]
                   [--output_dimension OUTPUT_DIMENSION] [-n MODEL_NAME]
                   [--training_points TRAINING_POINTS]
                   [--training_sampling TRAINING_SAMPLING]
                   [--validation_points VALIDATION_POINTS]
                   [--validation_sampling VALIDATION_SAMPLING] [-s SEED]
                   [-b BATCH_SIZE] [--epochs EPOCHS] [-l LEARNING_RATE]
                   [-o OUTPUT_PATH]
                   function

positional arguments:
  function              string representing a function to be evaluated with eval.

optional arguments:
  -h, --help            show this help message and exit
  --input_dimension INPUT_DIMENSION
                        input dimension.
  --output_dimension OUTPUT_DIMENSION
                        output dimension.
  -n MODEL_NAME, --model_name MODEL_NAME
                        model name.
  --training_points TRAINING_POINTS
                        number of training points.
  --training_sampling TRAINING_SAMPLING
                        training sampling strategy.
  --validation_points VALIDATION_POINTS
                        number of validation points.
  --validation_sampling VALIDATION_SAMPLING
                        validation sampling strategy.
  -s SEED, --seed SEED  seed for reproducible results.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size.
  --epochs EPOCHS       epochs.
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        output path.
```

For example, to approximate a sin using 2 hidden layers with 32 units each, just run:

```sh
approximate np.sin -l 2 -u 32
```
