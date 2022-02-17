# for training
import pandas as pd
from simpletransformers.t5 import T5Model

# for testing
import json
from datetime import datetime
from pprint import pprint
from statistics import mean
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1

df = pd.read_csv("../sample_dataset/eval.tsv", sep="\t").astype(str)
df = df.sample(frac=1).reset_index(drop=True)
train_df = df[:1000]
eval_df = df[1000:1200]
test_df = df[1200: 1400]
del df

model_args = {
    "max_seq_length": 196,
    "train_batch_size": 4,
    "eval_batch_size": 4,
    "num_train_epochs": 1,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 15000,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "fp16": False,
    "save_steps": -1,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    # "wandb_project": "T5 mixed tasks - Binary, Multi-Label, Regression",
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
    "max_seq_length": 196,
    "eval_batch_size": 4,
    "num_train_epochs": 1,
    "use_multiprocessing": False,
    "num_beams": None,
    "do_sample": True,
    "max_length": 50,
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
    "binary classification": {"truth": [], "preds": [],},
    "multilabel classification": {"truth": [], "preds": [],}
}

results_dict = {}

for task, truth_value, pred in zip(tasks, truth, preds):
    output_dict[task]["truth"].append(truth_value)
    output_dict[task]["preds"].append(pred)

print("-----------------------------------")
print("Results: ")
for task, outputs in output_dict.items():
    if task == "multilabel classification":
        try:
            task_truth = output_dict[task]["truth"]
            task_preds = output_dict[task]["preds"]
            results_dict[task] = {
                "F1 Score": f1(task_truth, task_preds),
                "Exact matches": exact(task_truth, task_preds),
            }
            print(f"Scores for {task}:")
            print(f"F1 score: {f1(task_truth, task_preds)}")
            print(f"Exact matches: {exact(task_truth, task_preds)}")
            print()
        except:
            pass
    elif task == "binary classification":
        try:
            task_truth = [int(t) for t in output_dict[task]["truth"]]
            task_preds = [int(p) for p in output_dict[task]["preds"]]
            results_dict[task] = {
                "F1 Score": f1_score(task_truth, task_preds),
                "Accuracy Score": accuracy_score(task_truth, task_preds),
            }
            print(f"Scores for {task}:")
            print(f"F1 score: {results_dict[task]['F1 Score']}")
            print(f"Accuracy Score: {results_dict[task]['Accuracy Score']}")
            print()
        except:
            pass

with open(f"results/result_{datetime.now()}.json", "w") as f:
    json.dump(results_dict, f)