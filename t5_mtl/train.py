import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from tqdm import tqdm

from transformers import T5Tokenizer, T5ForConditionalGeneration

from rich.table import Column, Table
from rich import box
from rich.console import Console


class MTLDataSet(Dataset):
    """
    Creating a custom dataset for reading the dataset and 
    loading it into the dataloader to pass it to the neural network for finetuning the model

    """

    def __init__(self, dataframe, tokenizer, source_len, target_len, source_text, target_text):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = ' '.join(source_text.split())
        target_text = ' '.join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text], max_length=self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        target = self.tokenizer.batch_encode_plus(
            [target_text], max_length=self.summ_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


class T5Trainer:
    def __init__(self, model_params: dict,
                 source_column: str = 'input_text', target_column: str = 'target_text',
                 output_dir=None) -> None:
        self.model_params = model_params
        self.source_column = source_column
        self.target_column = target_column
        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(
                Path(os.path.dirname(os.path.realpath(__file__))), 'outputs')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.device = 'cuda' if cuda.is_available() else 'cpu'
        print(f"device is set to {self.device}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_params["MODEL"]).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=model_params["LEARNING_RATE"])

        # For logging
        self._console = Console(record=True)
        self._training_logger = Table(Column("Epoch", justify="center"),
                                      Column("Steps", justify="center"),
                                      Column("Loss", justify="center"),
                                      title="Training Status", pad_edge=False, box=box.ASCII)

    def run(self, dataframe):
        # Set random seeds and deterministic pytorch for reproducibility
        torch.manual_seed(self.model_params["SEED"])  # pytorch random seed
        np.random.seed(self.model_params["SEED"])  # numpy random seed
        torch.backends.cudnn.deterministic = True

        # logging
        self._console.log(f"[Data]: Reading data...\n")

        # Importing the df dataset
        dataframe = dataframe[[self.source_column, self.target_column]]

        # Creation of Dataset and Dataloader
        # Defining the train size. So 80% of the data will be used for training and the rest for validation.
        train_size = 0.8
        train_dataset = dataframe.sample(
            frac=train_size, random_state=self.model_params["SEED"])
        val_dataset = dataframe.drop(
            train_dataset.index).reset_index(drop=True)
        train_dataset = train_dataset.reset_index(drop=True)

        self._console.print(f"FULL Dataset: {dataframe.shape}")
        self._console.print(f"TRAIN Dataset: {train_dataset.shape}")
        self._console.print(f"TEST Dataset: {val_dataset.shape}\n")

        # Creating the Training and Validation dataset for further creation of Dataloader
        training_set = MTLDataSet(train_dataset, self.tokenizer,
                                  self.model_params["MAX_SOURCE_TEXT_LENGTH"], self.model_params["MAX_TARGET_TEXT_LENGTH"],
                                  self.source_column, self.target_column)
        val_set = MTLDataSet(val_dataset, self.tokenizer,
                             self.model_params["MAX_SOURCE_TEXT_LENGTH"], self.model_params["MAX_TARGET_TEXT_LENGTH"],
                             self.source_column, self.target_column)

        # Defining the parameters for creation of dataloaders
        train_params = {
            'batch_size': self.model_params["TRAIN_BATCH_SIZE"],
            'shuffle': True,
            # 'num_workers': 0
        }

        val_params = {
            'batch_size': self.model_params["VALID_BATCH_SIZE"],
            'shuffle': False,
            # 'num_workers': 0
        }

        # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
        training_loader = DataLoader(training_set, **train_params)
        val_loader = DataLoader(val_set, **val_params)

        # Train
        self._console.log(f'[Initiating Fine Tuning]...\n')
        self.train(training_loader)

        # Saving the model after training
        save_path = os.path.join(self.output_dir, "model_files")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # evaluating the test dataset
        self._console.log(f"[Initiating Validation]...\n")
        predictions, actuals = self.validate(val_loader)
        final_df = pd.DataFrame({
            'Generated Text': predictions,
            'Actual Text': actuals
        })
        final_df.to_csv(os.path.join(self.output_dir, 'predictions.csv'))

        self._console.save_text(os.path.join(self.output_dir, 'logs.txt'))

        self._console.log(f"[Validation Completed.]\n")
        self._console.print(
            f"""[Model] Model saved @ {os.path.join(self.output_dir, "model_files")}\n""")
        self._console.print(
            f"""[Validation] Generation on Validation data saved @ {os.path.join(self.output_dir,'predictions.csv')}\n""")
        self._console.print(
            f"""[Logs] Logs saved @ {os.path.join(self.output_dir,'logs.txt')}\n""")

    def train(self, loader):
        self.model.train()
        for epoch in range(self.model_params["TRAIN_EPOCHS"]):
            for i, data in enumerate(tqdm(loader)):
                y = data['target_ids'].to(self.device, dtype=torch.long)
                y_ids = y[:, :-1].contiguous()
                lm_labels = y[:, 1:].clone().detach()
                lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100
                ids = data['source_ids'].to(self.device, dtype=torch.long)
                mask = data['source_mask'].to(self.device, dtype=torch.long)

                outputs = self.model(input_ids=ids, attention_mask=mask,
                                     decoder_input_ids=y_ids, labels=lm_labels)
                loss = outputs[0]

                if i % 500 == 0:
                    self._training_logger.add_row(
                        str(epoch), str(i), str(loss))
                    self._console.print(self._training_logger)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def validate(self, loader):
        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for i, data in enumerate(loader, 0):
                y = data['target_ids'].to(self.device, dtype=torch.long)
                ids = data['source_ids'].to(self.device, dtype=torch.long)
                mask = data['source_mask'].to(self.device, dtype=torch.long)

                generated_ids = self.model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_length=512,
                    num_beams=2,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True
                )
                preds = [self.tokenizer.decode(g, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=True) for g in generated_ids]
                target = [self.tokenizer.decode(t, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True) for t in y]
                if i % 50 == 0:
                    self._console.print(f'Completed {i}')

                predictions.extend(preds)
                actuals.extend(target)
        return predictions, actuals


if __name__ == "__main__":
    sample_dataset_dir_path = os.path.join(
        Path(os.path.dirname(os.path.realpath(__file__))).parent, 'sample_dataset')
    sample_data_path = os.path.join(sample_dataset_dir_path, 'eval.tsv')
    df = pd.read_csv(sample_data_path, sep="\t",
                     index_col=False).astype(str)
    df = df[['prefix', 'input_text', 'target_text']]
    df['colon'] = ': '

    df['input_text'] = df['prefix'] + df['colon'] + df['input_text']
    df = df[['input_text', 'target_text']]
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[:10_000]
    print(df.head(10))
    print(len(df))

    model_params = {
        "MODEL": "t5-base",             # model_type: t5-base/t5-large
        "TRAIN_BATCH_SIZE": 4,          # training batch size
        "VALID_BATCH_SIZE": 4,          # validation batch size
        "TRAIN_EPOCHS": 1,              # number of training epochs
        "LEARNING_RATE": 1e-4,          # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 256,  # max length of target text
        "SEED": 42                      # set seed for reproducibility

    }

    t5_trainer = T5Trainer(model_params=model_params)
    t5_trainer.run(dataframe=df)
