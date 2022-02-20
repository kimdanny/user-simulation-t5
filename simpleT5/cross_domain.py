import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from simpletransformers.t5 import T5Model
from transform_df import transform_df
import torch
from evaluate import evaluate_satisfaction, evaluate_utterance
torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(42)  # pytorch random seed
np.random.seed(42)  # numpy random seed


TASK = 'act-sat-utt_no-alt'
generation_args = {
                "num_beams": 5,
                "do_sample": True,  # Whether or not to use sampling ; use greedy decoding otherwise.
                "length_penalty": 1.0,  # Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences.
                "repetition_penalty": 2.0,  # The parameter for repetition penalty. 1.0 means no penalty.
                "top_k": 50,    # The number of highest probability vocabulary tokens to keep for top-k-filtering.
                "top_p": 0.95,  # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
                "num_return_sequences": 1  # The number of independently computed returned sequences for each element in the batch.
}

with open('cross_domain_results.txt', 'w') as f:
    for DATASET_on_model in tqdm(['MWOZ', 'SGD', 'CCPE']):
        for other_DATASET in tqdm(['MWOZ', 'SGD', 'CCPE']):
            if DATASET_on_model == other_DATASET:
                continue
            else:
                best_model_load_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))), 
                                                    TASK, DATASET_on_model, 'output', 'best_model')


                # Load the best trained model
                model = T5Model("t5", best_model_load_path, args=generation_args)

                # load other dataset
                other_dataset_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, 
                                                    'dataset', TASK, f'{other_DATASET}_df.csv')

                df = pd.read_csv(other_dataset_path, index_col=False).astype(str)
                df = transform_df(df)
                df = df.sample(frac=1).reset_index(drop=True)

                # Train-valid-test split as the same as when training and testing.
                # ensure that we have same test_df with same seed and same split ratio
                perm = np.random.permutation(df.index)
                m = len(df.index)
                train_end = int(0.8 * m)
                validate_end = int(0.1 * m) + train_end
                test_df = df.iloc[perm[validate_end:]]
                test_df = test_df.reset_index(drop=True)
                del train_end, validate_end, m, perm, df

                # Prepare the data for testing
                to_predict = [
                    prefix + ": " + str(input_text)
                    for prefix, input_text in zip(test_df["prefix"].tolist(), test_df["input_text"].tolist())
                ]
                truth = test_df["target_text"].tolist()
                tasks = test_df["prefix"].tolist()

                # Get the model predictions
                preds = model.predict(to_predict)
                del to_predict

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


                UAR, kappa, rho, bi_f1 = evaluate_satisfaction(predictions_dict['satisfaction score'])
                bleu_1, bleu_4, rouge_1_f1, rouge_2_f1, rouge_L_f1, sts = evaluate_utterance(predictions_dict['utterance generation'])

                f.write(f"{DATASET_on_model} Model does generation on {other_DATASET}\n")
                f.write("Satisfaction Result: \n")
                f.write('UAR,    kappa,    rho,    bi_f1\n')
                f.write(f"{UAR}, {kappa}, {rho}, {bi_f1}\n")
                f.write('\n')
                f.write("Utterance Generation Result: \n")
                f.write('bleu_1,   bleu_4,   rouge_1_f1,   rouge_2_f1,   rouge_L_f1,   sts\n')
                f.write(f"{bleu_1}, {bleu_4}, {rouge_1_f1}, {rouge_2_f1}, {rouge_L_f1}, {sts}\n")
                f.write("____________________\n")