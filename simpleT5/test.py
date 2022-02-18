from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score, precision_score, recall_score
import pickle
import json
import os
from pathlib import Path
import argparse


# sample output_dict
output_dict = {'satisfaction score': {'truth': ['3', '3', '4', '3', '4', '2', '3', '4', '3', '2', '2', '3', '3', '2', '3', '3', '3', '2', '4', '3', '4', '3', '4', '2', '3', '2', '2', '4', '4', '3', '3', '4', '4', '3', '4', '4', '3', '3', '3', '3', '2', '2', '3', '2', '4', '3', '4', '2', '3', '3', '4', '3', '3', '3', '4', '3', '4', '3', '3', '4', '3', '2', '4', '3', '3', '3', '3', '3', '2', '3', '2', '2', '3', '4', '3', '2', '3', '2', '2', '3', '2', '3', '2', '2', '3', '2', '2', '2', '3', '2', '3', '2', '4', '4', '3', '2', '3', '4', '4', '2', '3', '4', '4', '2', '4', '2', '4', '3', '4', '3', '2', '2', '3', '3', '4', '2', '4', '3', '3', '3', '2', '2', '4', '3', '3', '4', '2', '3', '3', '3', '4', '3', '3', '2', '3', '4', '2', '2', '2', '3', '3', '3', '2', '3', '3', '2', '4', '2', '4', '2', '2', '2', '3', '3', '3', '3', '3', '2', '2'], 'preds': ['2', '3', '3', '4', '3', '3', '2', '3', '4', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '4', '3', '3', '3', '3', '2', '3', '3', '3', '3', '3', '2', '4', '3', '3', '3', '3', '3', '3', '3', '4', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '2', '2', '3', '3', '3', '2', '2', '3', '3', '3', '2', '3', '3', '3', '3', '2', '3', '3', '4', '3', '2', '3', '3', '3', '3', '2', '3', '3', '3', '3', '3', '3', '3', '3', '2', '3', '3', '3', '2', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '2', '3', '3', '2', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '2', '3', '3', '3', '2', '3', '3', '3', '3', '3', '3', '2', '2', '3', '3', '3', '3', '3']}, 'action prediction': {'truth': ['Train-Inform', 'Restaurant-Inform', 'Hotel-Request', 'Hotel-Inform', 'Hotel-Inform', 'Attraction-Inform', 'Train-Inform', 'Hotel-Inform', 'Restaurant-Inform', 'Train-Inform', 'Hotel-Inform', 'Train-Inform', 'Restaurant-Inform', 'Taxi-Inform', 'Taxi-Request', 'Hotel-Inform', 'Train-Request', 'Attraction-Inform', 'Train-Inform', 'Attraction-Request', 'Restaurant-Request', 'Restaurant-Inform', 'Train-Inform', 'general-thank', 'Hotel-Inform', 'Hotel-Inform', 'general-thank', 'Restaurant-Inform', 'Hotel-Request', 'general-thank', 'Attraction-Inform', 'Attraction-Request', 'Restaurant-Inform', 'Restaurant-Request', 'Train-Inform', 'general-thank', 'general-thank', 'Train-Inform', 'Restaurant-Inform', 'general-welcome', 'general-thank'], 'preds': ['Train-Inform', 'Restaurant-Inform', 'Hotel-Inform', 'Hotel-Inform', 'general-thank', 'Train-Inform', 'Train-Inform', 'general-thank', 'Restaurant-Inform', 'Train-Inform', 'Hotel-Inform', 'Train-Inform', 'Restaurant-Inform', 'general-thank', 'general-thank', 'Attraction-Inform', 'general-thank', 'Restaurant-Inform', 'Attraction-Inform', 'general-thank', 'Restaurant-Inform', 'general-thank', 'general-thank', 'Hotel-Inform', 'general-thank', 'general-thank', 'general-thank', 'Train-Inform', 'Hotel-Inform', 'general-thank', 'general-thank', 'Restaurant-Inform', 'Restaurant-Inform', 'Restaurant-Inform', 'Train-Inform', 'general-thank', 'general-thank', 'Attraction-Request', 'Train-Inform', 'general-thank', 'general-thank']}, 'utterance generation': {'truth': [], 'preds': []}}


# Spearman algorithm
def spearman(x, y):
    assert len(x) == len(y) > 0
    q = lambda n: map(lambda val: sorted(n).index(val) + 1, n)
    d = sum(map(lambda x, y: (x - y) ** 2, q(x), q(y)))
    return 1.0 - 6.0 * d / float(len(x) * (len(y) ** 2 - 1.0))


def test_satisfaction(output_dict):
    truths = [int(score) for score in output_dict['truth']]
    preds = [int(score) for score in output_dict['preds']]
    
    recall = [[0, 0] for _ in range(5)]
    for p, l in zip(preds, truths):
        recall[l][1] += 1
        recall[l][0] += int(p == l)
    recall_value = [item[0] / max(item[1], 1) for item in recall]
    # print('Recall value:', recall_value)
    # print('Recall:', recall)
    UAR = sum(recall_value) / len(recall_value)
    kappa = cohen_kappa_score(preds, truths)
    rho = spearman(preds, truths)

    bi_preds = [int(item < 3) for item in preds]
    bi_truths = [int(item < 3) for item in truths]
    bi_recall = sum([int(p == l) for p, l in zip(bi_preds, bi_truths) if l == 1]) / max(bi_truths.count(1), 1)
    bi_precision = sum([int(p == l) for p, l in zip(bi_preds, bi_truths) if p == 1]) / max(bi_preds.count(1), 1)
    bi_f1 = 2 * bi_recall * bi_precision / max((bi_recall + bi_precision), 1)

    return UAR, kappa, rho, bi_f1


# TODO: figure out multiple truths -> e.g
def test_action(output_dict):
    # prediction = [int(line.split(',')[0]) for line in data]
    # label = [int(line.split(',')[1]) for line in data]
    # acc = sum([int(p == l) for p, l in zip(prediction, label)]) / len(label)
    # precision = precision_score(label, prediction, average='macro', zero_division=0)
    # recall = recall_score(label, prediction, average='macro', zero_division=0)
    # f1 = f1_score(label, prediction, average='macro', zero_division=0)

    # return acc, precision, recall, f1
    pass


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
    # print(f"{DATASET} will be trained for {TASK} task")

    TASK = 'act-sat_no-alt'
    DATASET = 'MWOZ'

    predictions_dir_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))), TASK, DATASET, 'predictions')
    results_dir_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))), TASK, DATASET, 'results')

    # output_dict = dict()
    # with open(os.path.join(predictions_dir_path, f"predictions_dict.pickle"), 'rb') as f:
    #     output_dict = pickle.load(f)

    UAR, kappa, rho, bi_f1 = test_satisfaction(output_dict['satisfaction score'])
    print(UAR, kappa, rho, bi_f1)

    acc, precision, recall, f1 = test_action(output_dict['action prediction'])
    print(acc, precision, recall, f1)


    results_dict = {}

    print("-----------------------------------")
    print("Results: ")
    for task, outputs in output_dict.items():
        if task == "satisfaction score":
            try:
                UAR, kappa, rho, bi_f1 = test_satisfaction(output_dict[task])
                results_dict[task] = {
                    "UAR": UAR,
                    "Kappa": kappa,
                    "Rho": rho,
                    "bi-F1": bi_f1
                }
                print(f"Scores for {task}:")
                print(f"UAR: {UAR}")
                print(f"Kappa: {kappa}")
                print(f"Rho: {rho}")
                print(f"bi-F1: {bi_f1}")
                print()
            except Exception as e:
                print(f"Error from {task} testing:")
                print(e)
        elif task == "action prediction":
            try:
                acc, precision, recall, f1 = test_action(output_dict[task])
                results_dict[task] = {
                    "Acc": acc,
                    "Prec": precision,
                    "Recall": recall,
                    "F1": f1
                }
                print(f"Scores for {task}:")
                print(f"Acc: {acc}")
                print(f"Prec: {precision}")
                print(f"Recall: {recall}")
                print(f"F1: {f1}")
                print()
            except Exception as e:
                print(f"Error from {task} testing:")
                print(e)
        elif task == 'utterance generation':
            # TODO: BLEU, STS
            pass


    Path(results_dir_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(results_dir_path, f"results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)