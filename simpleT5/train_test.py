# Modified from: https://github.com/ThilinaRajapakse/simpletransformers/tree/master/examples/t5/mixed_tasks

# for training
import os
from pathlib import Path
import numpy as np
import pandas as pd
from simpletransformers.t5 import T5Model
from transform_df import transform_df
import torch

# for testing
import json
from datetime import datetime
from statistics import mean
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1

# from weiwei's testing script
from sklearn.metrics import cohen_kappa_score
from metrics import spearman
from sklearn.metrics import f1_score, precision_score, recall_score

torch.manual_seed(42)  # pytorch random seed
np.random.seed(42)  # numpy random seed

# df = pd.read_csv("../sample_dataset/eval.tsv", sep="\t").astype(str)
# df = df.sample(frac=1).reset_index(drop=True)
# train_df = df[:1000]
# eval_df = df[1000:1200]
# test_df = df[1200: 1400]
# del df

TASK = 'act-sat_no-alt'
DATASET = 'MWOZ'
dataset_dir_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, 'dataset', TASK)
dataset_path = os.path.join(dataset_dir_path, f'{DATASET}_df.csv')

df = pd.read_csv(dataset_path, index_col=False).astype(str)
df = transform_df(df)
df = df.sample(frac=1).reset_index(drop=True)
train_df = df[:50]
eval_df = df[50:60]
test_df = df[60: 70]


model_args = {
    "max_seq_length": 512,
    "train_batch_size": 1,
    "eval_batch_size": 1,
    "num_train_epochs": 1,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 500,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "fp16": False,
    "save_steps": -1,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "reprocess_input_data": True,
    "overwrite_output_dir": True
}

model = T5Model("t5", "t5-base", args=model_args)

model.train_model(train_df, eval_data=eval_df)


##########
# TESTING
##########
def f1(truths, preds):
    return mean([compute_f1(truth, pred) for truth, pred in zip(truths, preds)])


def exact(truths, preds):
    return mean([compute_exact(truth, pred) for truth, pred in zip(truths, preds)])


model_args = {
    "overwrite_output_dir": True,
    "max_seq_length": 512,
    "eval_batch_size": 1,
    "num_train_epochs": 1,
    "use_multiprocessing": False,
    "num_beams": 5,
    "do_sample": True,
    "max_length": 10,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 3,
}

# Load the trained model
model = T5Model("t5", "outputs", args=model_args)

# Prepare the data for testing
to_predict = [
    prefix + ": " + str(input_text)
    for prefix, input_text in zip(test_df["prefix"].tolist(), test_df["input_text"].tolist())
]
truth = test_df["target_text"].tolist()
tasks = test_df["prefix"].tolist()

# Get the model predictions
preds = model.predict(to_predict)

# Saving the predictions if needed
with open(f"predictions/predictions_{datetime.now()}.txt", "w") as f:
    for i, text in enumerate(test_df["input_text"].tolist()):
        f.write(str(text) + "\n\n")

        f.write("Truth:\n")
        f.write(truth[i] + "\n\n")

        f.write("Prediction:\n")
        for pred in preds[i]:
            f.write(str(pred) + "\n")
        f.write(
            "________________________________________________________________________________\n"
        )

# Taking only the first prediction
preds = [pred[0] for pred in preds]
test_df["predicted"] = preds

# Evaluating the tasks separately
output_dict = {
    "satisfaction score": {"truth": [], "preds": [],},
    "action prediction": {"truth": [], "preds": [],},
    "utterance generation": {"truth": [], "preds": [],}
}

results_dict = {}

for task, truth_value, pred in zip(tasks, truth, preds):
    output_dict[task]["truth"].append(truth_value)
    output_dict[task]["preds"].append(pred)

print("-----------------------------------")
print("Results: ")
for task, outputs in output_dict.items():
    if task == "satisfaction score":
        try:
            task_truth = output_dict[task]["truth"]
            task_preds = output_dict[task]["preds"]
            results_dict[task] = {
                "F1 Score": f1(task_truth, task_preds),
                "Accuracy Score": accuracy_score(task_truth, task_preds),
                "Exact matches": exact(task_truth, task_preds),
            }
            print(f"Scores for {task}:")
            print(f"F1 score: {f1(task_truth, task_preds)}")
            print(f"Accuracy Score: {results_dict[task]['Accuracy Score']}")
            print(f"Exact matches: {exact(task_truth, task_preds)}")
            print()
        except:
            pass
    elif task == "action prediction":
        try:
            task_truth = output_dict[task]["truth"]
            task_preds = output_dict[task]["preds"]
            results_dict[task] = {
                "F1 Score": f1_score(task_truth, task_preds),
                "Accuracy Score": accuracy_score(task_truth, task_preds),
                "Exact matches": exact(task_truth, task_preds),
            }
            print(f"Scores for {task}:")
            print(f"F1 score: {results_dict[task]['F1 Score']}")
            print(f"Accuracy Score: {results_dict[task]['Accuracy Score']}")
            print(f"Exact matches: {results_dict[task]['Exact matches']}")
            print()
        except Exception as e:
            print(e)

with open(f"results/result_{datetime.now()}.json", "w") as f:
    json.dump(results_dict, f)