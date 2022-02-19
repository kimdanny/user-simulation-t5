# Modified from https://github.com/sunnweiwei/user-satisfaction-simulation/blob/master/baselines/test.py

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score, precision_score, recall_score
from metrics import spearman, get_bleu_1_4, get_rouge_1_2_L, get_sts
import pickle
import json
import os
from pathlib import Path
import argparse


def evaluate_satisfaction(output_dict):
    truths = [int(score) - 1 for score in output_dict['truth']]
    preds = [int(score) - 1 for score in output_dict['preds']]
    
    recall = [[0, 0] for _ in range(5)]
    for p, l in zip(preds, truths):
        recall[l][1] += 1
        recall[l][0] += int(p == l)
    recall_value = [item[0] / max(item[1], 1) for item in recall]
    
    UAR = sum(recall_value) / len(recall_value)
    kappa = cohen_kappa_score(preds, truths)
    rho = spearman(preds, truths)

    bi_preds = [int(item < 2) for item in preds]
    bi_truths = [int(item < 2) for item in truths]
    bi_recall = sum([int(p == l) for p, l in zip(bi_preds, bi_truths) if l == 1]) / max(bi_truths.count(1), 1)
    bi_precision = sum([int(p == l) for p, l in zip(bi_preds, bi_truths) if p == 1]) / max(bi_preds.count(1), 1)
    bi_f1 = 2 * bi_recall * bi_precision / max((bi_recall + bi_precision), 1)

    return UAR, kappa, rho, bi_f1


def evaluate_action(output_dict):
    prediction = [pred for pred in output_dict['preds']]
    label = [truth for truth in output_dict['truth']]
    acc = sum([p == l for p, l in zip(prediction, label)]) / len(label)
    precision = precision_score(label, prediction, average='macro', zero_division=0)
    recall = recall_score(label, prediction, average='macro', zero_division=0)
    f1 = f1_score(label, prediction, average='macro', zero_division=0)

    return acc, precision, recall, f1


def evaluate_utterance(output_dict):
    references = [truth for truth in output_dict['truth']]
    candidates = [pred for pred in output_dict['preds']]
    # BLEU
    bleu_1, bleu_4 = get_bleu_1_4(references, candidates)
    # ROUGE
    rouge_1_f1, rouge_2_f1, rouge_L_f1 = get_rouge_1_2_L(references, candidates)
    # STS
    sts = get_sts(references, candidates)
    
    return bleu_1, bleu_4, rouge_1_f1, rouge_2_f1, rouge_L_f1, sts


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='pass the TASK name and DATASET name to train a T5 MTL model')
    # parser.add_argument('-t','--task', help='e.g. act-sat', required=True)
    # parser.add_argument('-d','--dataset', help='e.g. MWOZ', required=True)
    # args = vars(parser.parse_args())

    # TASK = args['task']
    # DATASET = args['dataset']
    # assert TASK in {'act-sat', 'act-sat-utt', 'act-sat_no-alt', 'act-sat-utt_no-alt'}
    # assert DATASET in {'CCPE', 'MWOZ', 'SGD'}
    # del parser, args
    # print(f"{TASK}/{DATASET} will be evaluated")

    TASK = 'act-sat-utt_no-alt'
    DATASET = 'CCPE'

    predictions_dir_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))), TASK, DATASET, 'predictions')
    results_dir_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))), TASK, DATASET, 'results')

    output_dict = dict()
    with open(os.path.join(predictions_dir_path, f"predictions_dict.pickle"), 'rb') as f:
        output_dict = pickle.load(f)


    ###############
    # Save the results
    ###############
    results_dict = {}

    print("-----------------------------------")
    print("Results: ")
    for task, outputs in output_dict.items():
        if task == "satisfaction score":
            try:
                UAR, kappa, rho, bi_f1 = evaluate_satisfaction(output_dict[task])
                results_dict[task] = {
                    "UAR": UAR,
                    "Kappa": kappa,
                    "Rho": rho,
                    "bi-F1": bi_f1
                }
            except Exception as e:
                print(f"Error from {task} evaluation:")
                print(e)
        elif task == "action prediction":
            try:
                acc, precision, recall, f1 = evaluate_action(output_dict[task])
                results_dict[task] = {
                    "Acc": acc,
                    "Prec": precision,
                    "Recall": recall,
                    "F1": f1
                }
            except Exception as e:
                print(f"Error from {task} evaluation:")
                print(e)
        elif task == 'utterance generation':
            try:
                bleu_1, bleu_4, rouge_1_f1, rouge_2_f1, rouge_L_f1, sts = evaluate_utterance(output_dict[task])
                results_dict[task] = {
                    "BLEU-1": bleu_1,
                    "BLEU_4": bleu_4,
                    "ROUGE-1_f1": rouge_1_f1,
                    "ROUGE-2_f1": rouge_2_f1,
                    "ROUGE-L_f1": rouge_L_f1,
                    "STS": sts
                }
            except Exception as e:
                print(f"Error from {task} evaluation:")
                print(e)

    # save to path
    Path(results_dir_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(results_dir_path, f"results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)