# Modified from: https://github.com/ThilinaRajapakse/simpletransformers/tree/master/examples/t5/mixed_tasks

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


torch.manual_seed(42)  # pytorch random seed
np.random.seed(42)  # numpy random seed

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1

# parser = argparse.ArgumentParser(description='pass the TASK name and DATASET name to train a T5 MTL model')
# parser.add_argument('-t','--task', help='e.g. act-sat', required=True)
# parser.add_argument('-d','--dataset', help='e.g. MWOZ', required=True)
# args = vars(parser.parse_args())

# TASK = args['task']
# DATASET = args['dataset']
# assert TASK in {'act-sat', 'act-sat-utt', 'act-sat_no-alt', 'act-sat-utt_no-alt'}
# assert DATASET in {'CCPE', 'MWOZ', 'SGD'}
# del parser, args
# print(f"{DATASET} will be trained for {TASK} task")



TASK = 'act-sat_no-alt'
DATASET = 'SGD'
dataset_dir_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, 'dataset', TASK)
dataset_path = os.path.join(dataset_dir_path, f'{DATASET}_df.csv')

# clear the task-dataset output directory
if os.path.exists(os.path.join(Path(os.path.dirname(os.path.realpath(__file__))), TASK, DATASET)):
    shutil.rmtree(os.path.join(Path(os.path.dirname(os.path.realpath(__file__))), TASK, DATASET))

output_dir_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))), TASK, DATASET, 'output')
best_model_dir_path = os.path.join(output_dir_path, 'best_model')

predictions_dir_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))), TASK, DATASET, 'predictions')
results_dir_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))), TASK, DATASET, 'results')

df = pd.read_csv(dataset_path, index_col=False).astype(str)
df = transform_df(df)
df = df.sample(frac=1).reset_index(drop=True)

# Train-valid-test split
perm = np.random.permutation(df.index)
m = len(df.index)
train_end = int(TRAIN_RATIO * m)
validate_end = int(VALID_RATIO * m) + train_end
train_df = df.iloc[perm[:train_end]]
valid_df = df.iloc[perm[train_end:validate_end]]
test_df = df.iloc[perm[validate_end:]]

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
del train_end, validate_end, m, perm

print(f"FULL Dataset: {df.shape}")
print(f"TRAIN Dataset: {train_df.shape}")
print(f"VALIDATION Dataset: {valid_df.shape}")
print(f"TEST Dataset: {test_df.shape}")


#########
# TRAIN
########
model_args = {
    'manual_seed': 42,
    "max_seq_length": 512,
    "max_length": 10,

    "train_batch_size": 4,
    "eval_batch_size": 4,

    "num_train_epochs": 7,
    # 'learning_rate': 4e-5,
    'optimizer': 'AdamW',
    'scheduler': 'linear_schedule_with_warmup',

    'use_early_stopping': True,
    'early_stopping_consider_epochs': True,
    'early_stopping_metric': 'eval_loss',
    'early_stopping_patience': 3,

    "evaluate_during_training": True,
    "evaluate_during_training_steps": 5000,
    "evaluate_during_training_verbose": True,

    "overwrite_output_dir": True,
    'output_dir': output_dir_path,
    "best_model_dir": best_model_dir_path,
    "logging_steps": 500,

    "use_multiprocessing": False,
    "fp16": False,
    "save_steps": -1,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "reprocess_input_data": True
}

model = T5Model("t5", "t5-base", args=model_args, use_cuda=True)

model.train_model(train_df, eval_data=valid_df)


##########
# TEST
##########

# args for generation
# outputs = self.model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 num_beams=self.args.num_beams,
#                 max_length=self.args.max_length,
#                 length_penalty=self.args.length_penalty,
#                 early_stopping=self.args.early_stopping,
#                 repetition_penalty=self.args.repetition_penalty,
#                 do_sample=self.args.do_sample,
#                 top_k=self.args.top_k,
#                 top_p=self.args.top_p,
#                 num_return_sequences=self.args.num_return_sequences,
#             )
generation_args = {
    "num_beams": 5,
    "do_sample": True,  # Whether or not to use sampling ; use greedy decoding otherwise.
    "length_penalty": 0.8,  # Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences.
    "repetition_penalty": 2.0,  # The parameter for repetition penalty. 1.0 means no penalty.
    "top_k": 50,    # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    "top_p": 0.95,  # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    "num_return_sequences": 1  # The number of independently computed returned sequences for each element in the batch.
}

# Load the best trained model
model = T5Model("t5", best_model_dir_path, args=generation_args)

# Prepare the data for testing
to_predict = [
    prefix + ": " + str(input_text)
    for prefix, input_text in zip(test_df["prefix"].tolist(), test_df["input_text"].tolist())
]
truth = test_df["target_text"].tolist()
tasks = test_df["prefix"].tolist()

# Get the model predictions
preds = model.predict(to_predict)

# Saving the predictions
# 1. csv file
Path(predictions_dir_path).mkdir(parents=True, exist_ok=True)
final_df = pd.DataFrame({
            'Generated Text': preds,
            'Actual Text': truth
            })
final_df.to_csv(os.path.join(predictions_dir_path, f"predictions.csv"), index=False)

# 2. txt file for better readability
with open(os.path.join(predictions_dir_path, f"predictions.txt"), "w") as f:
    for i, text in enumerate(test_df["input_text"].tolist()):
        f.write(f"Test row: {str(i)}\n")
        f.write(str(text) + "\n\n")

        f.write("Truth:\n")
        f.write(truth[i] + "\n\n")

        f.write("Prediction:\n")
        f.write(str(preds[i]) + "\n")
        f.write("________________________________________________________________________________\n")

# putting preds in test_df
test_df["predicted"] = preds

# Evaluating the tasks separately
predictions_dict = {
    "satisfaction score": {"truth": [], "preds": [],},
    "action prediction": {"truth": [], "preds": [],},
    "utterance generation": {"truth": [], "preds": [],}
}

for task, truth_value, pred in zip(tasks, truth, preds):
    predictions_dict[task]["truth"].append(truth_value)
    predictions_dict[task]["preds"].append(pred)

# save predictions_dict as pickle that is to be used in test.py
with open(os.path.join(predictions_dir_path, f"predictions_dict.pickle"), 'wb') as f:
    pickle.dump(predictions_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
