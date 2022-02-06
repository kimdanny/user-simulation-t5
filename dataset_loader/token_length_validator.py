# import sys
# sys.path.insert(0, '/home/ec2-user/user-simulation/dataset_loader')
from dataset_loader import DatasetLoader
from tqdm import tqdm
import os
from transformers import T5Tokenizer


tokenizer = T5Tokenizer.from_pretrained('t5-base')


def validate_token_length(dataset: str, config: dict, max_length: int = 512) -> None:
    if not os.path.exists('validation_logs'):
        os.makedirs('validation_logs')

    loader = DatasetLoader(config=config)
    histories, _, _, _, utterances = loader.load(dataset)

    # histories check
    with open(f'validation_logs/histories_{dataset}.txt', 'w') as f:
        for i, text in enumerate(tqdm(histories)):
            tokens = tokenizer.batch_encode_plus([text], truncation=False,
                                                 return_tensors='pt')

            if tokens['input_ids'].shape[1] > max_length:
                f.write(f"Index {i}\n")
                f.write(text)
                f.write('\n\n')

        f.close()

    # utterances check
    with open(f'validation_logs/utterances_{dataset}.txt', 'w') as f:
        for i, text in enumerate(tqdm(utterances)):
            tokens = tokenizer.batch_encode_plus([text], truncation=False,
                                                 return_tensors='pt')

            if tokens['input_ids'].shape[1] > max_length:
                f.write(f"Index {i}\n")
                f.write(text)
                f.write('\n\n')

        f.close()


if __name__ == "__main__":

    dataset_config = {
        'LOOK_N_TURNS': 5,
        'ENSURE_ALTERNATING_ROLES': True
    }

    for dataset in ['CCPE', 'MWOZ', 'ReDial', 'SGD']:
        validate_token_length(dataset, dataset_config)
