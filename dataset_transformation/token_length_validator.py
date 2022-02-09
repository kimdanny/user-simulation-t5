from tqdm import tqdm
import os
from transformers import T5Tokenizer


tokenizer = T5Tokenizer.from_pretrained('t5-base')


def check_token_length(text, max_length=512):
    """
    Returns two elements:
        (True/False, int) means (valid or not, overflowing amount if invalid)
    """
    tokens = tokenizer.batch_encode_plus([text], truncation=False,
                                         return_tensors='pt')
    token_length = tokens['input_ids'].shape[1]
    if token_length < max_length + 1:
        return True, 0
    else:
        overflow_amount = int(token_length - max_length)
        return False, overflow_amount


def experiment_token_length(dataset: str, config: dict, max_length: int = 512) -> None:
    from dataset_transformer import DatasetTransformer
    
    if not os.path.exists('experiment_logs'):
        os.makedirs('experiment_logs')

    loader = DatasetTransformer(config=config)
    histories, _, _, _, utterances = loader.transform(dataset)

    # histories check
    with open(f'experiment_logs/histories_{dataset}.txt', 'w') as f:
        cnt = 0
        for i, text in enumerate(histories):
            tokens = tokenizer.batch_encode_plus([text], truncation=False,
                                                 return_tensors='pt')

            if tokens['input_ids'].shape[1] > max_length:
                cnt += 1
                f.write(
                    f"Index {i} -> length: {tokens['input_ids'].shape[1]}\n")
                f.write(text)
                f.write('\n\n')
        f.write(f"# cases: {cnt}/{len(histories)}")
        f.close()

    # utterances check
    with open(f'experiment_logs/utterances_{dataset}.txt', 'w') as f:
        cnt = 0
        for i, text in enumerate(utterances):
            tokens = tokenizer.batch_encode_plus([text], truncation=False,
                                                 return_tensors='pt')

            if tokens['input_ids'].shape[1] > max_length:
                cnt += 1
                f.write(
                    f"Index {i} -> length: {tokens['input_ids'].shape[1]}\n")
                f.write(text)
                f.write('\n\n')
        f.write(f"# cases: {cnt}/{len(utterances)}")
        f.close()


if __name__ == "__main__":

    dataset_config = {
        'LOOK_N_TURNS': 10,
        'ENSURE_ALTERNATING_ROLES': True
    }

    for dataset in tqdm(['CCPE', 'MWOZ', 'ReDial', 'SGD']):
        experiment_token_length(dataset, dataset_config)

    """
    Experiment: different LOOK_N_TURNS and number of cases where token length exceeds 512

    dataset_config = {
        'LOOK_N_TURNS': -1,
        'ENSURE_ALTERNATING_ROLES': True
    }
    CCPE: 2854/5188
    MWOZ: 10540/10553
    ReDial: 3302/6976
    SGD: 12807/12832


    dataset_config = {
        'LOOK_N_TURNS': 5,
        'ENSURE_ALTERNATING_ROLES': True
    }
    CCPE: 1/5188


    dataset_config = {
        'LOOK_N_TURNS': 6,
        'ENSURE_ALTERNATING_ROLES': True
    }
    CCPE: 6/5188
    

    dataset_config = {
        'LOOK_N_TURNS': 7,
        'ENSURE_ALTERNATING_ROLES': True
    }
    CCPE: 7/5188
    

    dataset_config = {
        'LOOK_N_TURNS': 8,
        'ENSURE_ALTERNATING_ROLES': True
    }
    CCPE: 19/5188


    dataset_config = {
        'LOOK_N_TURNS': 9,
        'ENSURE_ALTERNATING_ROLES': True
    }
    CCPE: 23/5188


    dataset_config = {
        'LOOK_N_TURNS': 10,
        'ENSURE_ALTERNATING_ROLES': True
    }
    CCPE: 56/5188


    dataset_config = {
        'LOOK_N_TURNS': 15,
        'ENSURE_ALTERNATING_ROLES': True
    }
    CCPE: 241/5188


    dataset_config = {
        'LOOK_N_TURNS': 17,
        'ENSURE_ALTERNATING_ROLES': True
    }
    CCPE: 356/5188


    dataset_config = {
        'LOOK_N_TURNS': 20,
        'ENSURE_ALTERNATING_ROLES': True
    }
    CCPE: 618/5188
    MWOZ: 16/10553
    ReDial: 19/6976
    SGD: 4/12832

    """
