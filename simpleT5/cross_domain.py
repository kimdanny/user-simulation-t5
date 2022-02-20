import os
from pathlib import Path
import numpy as np
import pandas as pd
from simpletransformers.t5 import T5Model
from transform_df import transform_df
import torch
import argparse
import shutil
import pickle
torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(42)  # pytorch random seed
np.random.seed(42)  # numpy random seed


TASK = 'act-sat_no-alt'
DATASET = 'CCPE'

output_dir_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))), TASK, DATASET, 'output')
best_model_dir_path = os.path.join(output_dir_path, 'best_model')


generation_args = {
    "num_beams": 5,
    "do_sample": True,  # Whether or not to use sampling ; use greedy decoding otherwise.
    "length_penalty": 1.0,  # Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences.
    "repetition_penalty": 2.0,  # The parameter for repetition penalty. 1.0 means no penalty.
    "top_k": 50,    # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    "top_p": 0.95,  # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    "num_return_sequences": 1  # The number of independently computed returned sequences for each element in the batch.
}

# Load the best trained model
model = T5Model("t5", best_model_dir_path, args=generation_args)






