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


TASK_load = 'act-sat-utt_no-alt'
DATASET_load = 'MWOZ'

output_dir_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))), TASK_load, DATASET_load, 'output')
best_model_load_path = os.path.join(output_dir_path, 'best_model')


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
model = T5Model("t5", best_model_load_path, args=generation_args)

# Prepare the three tasks prediction
test_dict = {
    'prefix': [
        'satisfaction score', 'action prediction', 'utterance generation'
    ],
    'input_text': [
        # Sample conversation history from MultiWOZ 2.1
        "hi, i'm looking for an attraction in the center of town to visit. we have quite a few interesting attractions in the center of town. is there anything in particular you would like to see? i have no preference, i just need the address, postcode, and entrance fee. there are 44 different attractions here in cambridge. would you like me to pick one of them for you?"
    ] * 3
}
test_df = pd.DataFrame.from_dict(test_dict)


# Get the model predictions
to_predict = [
    prefix + ": " + str(input_text)
    for prefix, input_text in zip(test_df["prefix"].tolist(), test_df["input_text"].tolist())
]

preds = model.predict(to_predict)
print(preds)




