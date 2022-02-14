import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import cuda
from tqdm import tqdm

from transformers import T5Tokenizer, T5ForConditionalGeneration
from loader import MTLDataSet

from rich.table import Column, Table
from rich import box
from rich.console import Console
import sys


class T5Trainer:
    def __init__(self, model_params: dict,
                 source_column: str = 'input_text', target_column: str = 'target_text',
                 output_dir=None) -> None:
        self.model_params = model_params
        self.source_column = source_column
        self.target_column = target_column
        if output_dir is not None:
            self.output_dir = os.path.join(
                Path(os.path.dirname(os.path.realpath(__file__))), output_dir)
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
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
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
        # Splitting dataset into train-val-test
        perm = np.random.permutation(dataframe.index)
        m = len(dataframe.index)
        train_end = int(self.model_params["TRAIN_RATIO"] * m)
        validate_end = int(self.model_params["VALID_RATIO"] * m) + train_end
        train_dataset = dataframe.iloc[perm[:train_end]]
        validation_dataset = dataframe.iloc[perm[train_end:validate_end]]
        test_dataset = dataframe.iloc[perm[validate_end:]]

        train_dataset = train_dataset.reset_index(drop=True)
        validation_dataset = validation_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)
        del train_end, validate_end, m, perm

        self._console.print(f"FULL Dataset: {dataframe.shape}")
        self._console.print(f"TRAIN Dataset: {train_dataset.shape}")
        self._console.print(f"VALIDATION Dataset: {validation_dataset.shape}")
        self._console.print(f"TEST Dataset: {test_dataset.shape}\n")

        # Creating the Training and Validation dataset for further creation of Dataloader
        training_set = MTLDataSet(train_dataset, self.tokenizer,
                                  self.model_params["MAX_SOURCE_TEXT_LENGTH"], self.model_params["MAX_TARGET_TEXT_LENGTH"],
                                  self.source_column, self.target_column)
        validation_set = MTLDataSet(validation_dataset, self.tokenizer,
                                    self.model_params["MAX_SOURCE_TEXT_LENGTH"], self.model_params["MAX_TARGET_TEXT_LENGTH"],
                                    self.source_column, self.target_column)
        test_set = MTLDataSet(test_dataset, self.tokenizer,
                             self.model_params["MAX_SOURCE_TEXT_LENGTH"], self.model_params["MAX_TARGET_TEXT_LENGTH"],
                             self.source_column, self.target_column)

        # Defining the parameters for creation of dataloaders
        train_params = {
            'batch_size': self.model_params["TRAIN_BATCH_SIZE"],
            'shuffle': True,
            'num_workers': 0
        }

        validation_params = {
            'batch_size': self.model_params["VALID_BATCH_SIZE"],
            'shuffle': True,
            'num_workers': 0
        }

        test_params = {
            'batch_size': self.model_params["TEST_BATCH_SIZE"],
            'shuffle': False,
            'num_workers': 0
        }

        # Creation of Dataloaders for training and validation.
        training_loader = DataLoader(training_set, **train_params)
        validation_loader = DataLoader(validation_set, **validation_params)
        test_loader = DataLoader(test_set, **test_params)

        # Train and validate
        self._console.log(f'[Initiating Fine Tuning]...\n')
        self.train(training_loader, validation_loader)

        # Saving the model after training
        save_path = os.path.join(self.output_dir, "model_files")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # evaluating the validation dataset
        self._console.log(f"[Initiating Test]...\n")
        predictions, actuals = self.test(test_loader)
        final_df = pd.DataFrame({
            'Generated Text': predictions,
            'Actual Text': actuals
        })
        final_df.to_csv(os.path.join(self.output_dir, 'predictions.csv'), index=False)

        self._console.save_text(os.path.join(self.output_dir, 'logs.txt'))

        self._console.log(f"[TEST Completed.]\n")
        self._console.print(
            f"""[Model] Model saved @ {os.path.join(self.output_dir, "model_files")}\n""")
        self._console.print(
            f"""[TEST] Generation on Test data saved @ {os.path.join(self.output_dir,'predictions.csv')}\n""")
        self._console.print(
            f"""[Logs] Logs saved @ {os.path.join(self.output_dir,'logs.txt')}\n""")

    def train(self, train_loader, validation_loader):
        """
        After each epoch of training, validate with validation set and save the best model
        """
        # TODO
        # TODO
        # TODO
        minimum_loss = np.inf
        # TRAIN
        self.model.train()
        for epoch in range(self.model_params["TRAIN_EPOCHS"]):
            for i, batch in enumerate(tqdm(train_loader)):
                ids = batch['source_ids'].to(self.device, dtype=torch.long)
                mask = batch['source_mask'].to(self.device, dtype=torch.long)
                
                y = batch['target_ids'].to(self.device, dtype=torch.long)
                y_ids = y[:, :-1].contiguous()
                
                #  Labels for computing the sequence classification/regression loss. 
                # Indices should be in [-100, 0, ..., config.vocab_size - 1]. 
                # All labels set to -100 are ignored (masked), the loss is only computed for labels in [0, ..., config.vocab_size]
                lm_labels = y[:, 1:].clone().detach()
                lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100
                
                target_mask = batch['target_ids_y'].to(self.device, dtype=torch.long)
                target_mask = target_mask[:, :-1].contiguous()

                # outputs = self.model(input_ids=input_ids, attention_mask=input_mask,
                #                      labels=target_ids, decoder_attention_mask=target_mask)

                outputs = self.model(input_ids=ids, attention_mask=mask,
                                     decoder_input_ids=y_ids, labels=lm_labels,
                                     decoder_attention_mask=target_mask)
                loss = outputs[0]

                if i % 500 == 0:
                    self._training_logger.add_row(
                        str(epoch), str(i), str(loss))
                    self._console.print(self._training_logger)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Validation and save best model
            self._console.log(f"[Initiating Validation on epoch {epoch}]...\n")
            # TODO
            # TODO
            # TODO
            # TODO
            validation_loss = self.validate(validation_loader)
        
    
    def validate(self, loader):
        """
        sum the validation loss over all batches and return the average validation loss
        """
        self.model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for batch in tqdm(loader):
                ids = batch['source_ids'].to(self.device, dtype=torch.long)
                mask = batch['source_mask'].to(self.device, dtype=torch.long)
                
                y = batch['target_ids'].to(self.device, dtype=torch.long)
                y_ids = y[:, :-1].contiguous()
                
                #  Labels for computing the sequence classification/regression loss. 
                # Indices should be in [-100, 0, ..., config.vocab_size - 1]. 
                # All labels set to -100 are ignored (masked), the loss is only computed for labels in [0, ..., config.vocab_size]
                lm_labels = y[:, 1:].clone().detach()
                lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100
                
                target_mask = batch['target_ids_y'].to(self.device, dtype=torch.long)
                target_mask = target_mask[:, :-1].contiguous()

                outputs = self.model(input_ids=ids, attention_mask=mask,
                                        decoder_input_ids=y_ids, labels=lm_labels,
                                        decoder_attention_mask=target_mask)
                loss = outputs[0]
                
                total_valid_loss += loss
        
        avg_valid_loss = total_valid_loss / len(loader)
        return avg_valid_loss


    def test(self, loader):
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
                    max_length=self.model_params['MAX_TARGET_TEXT_LENGTH'],
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
    # TODO: ARGPARSE
    """
    act-sat
        MWOZ: training
        CCPE: training

    act-sat-utt
    """

    DATASET = 'CCPE'
    TASK = 'act-sat'

    dataset_dir_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, 'dataset')
    dataset_act_sat_path = os.path.join(dataset_dir_path, TASK)
    # dataset_act_sat_utt_path = os.path.join(dataset_dir_path, 'act-sat-utt')

    mwoz_path = os.path.join(dataset_act_sat_path, f'{DATASET}_df.csv')
    df = pd.read_csv(mwoz_path, index_col=False).astype(str)
    df = df.drop(df[df['target_text'] == 'None'].index)
    
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.head(10))
    print(len(df))

    model_params = {
        "MODEL": "t5-base",             # model_type: t5-base/t5-large
        "TRAIN_BATCH_SIZE": 4,          # training batch size
        "VALID_BATCH_SIZE": 4,          # validation batch size
        "TEST_BATCH_SIZE": 4,
        "TRAIN_RATIO": 0.8,
        "VALID_RATIO": 0.1,
        "TRAIN_EPOCHS": 7,              # number of training epochs
        "LEARNING_RATE": 1e-4,          # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 150,  # max length of target text
        "SEED": 42                      # set seed for reproducibility

    }

    t5_trainer = T5Trainer(model_params=model_params, output_dir=f'{TASK}_{DATASET}')
    t5_trainer.run(dataframe=df)
