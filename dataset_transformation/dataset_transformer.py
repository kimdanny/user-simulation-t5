"""
Adapted from https://github.com/sunnweiwei/user-satisfaction-simulation

Difference from original:
    1. Avoid empty string prediction.
        For example, in SGD, USER starts the conversation, so the first history is empty string.
        We omitted these cases.

    2. Do not predict OVERALL.
        Generating a user utterance "OVERALL" does not make sense, 
        so we omiited the last turn (OVERALL) of each session.
        
    3. (Optional) Can ensure that two roles alternate with each other.
        i.e. Transforms cases like SYSTEM-USER-USER-USER-SYSTEM to SYSTEM-USER-SYSTEM
"""
from urllib.error import HTTPError
from token_length_validator import check_token_length
import os
from pathlib import Path
import numpy as np
import pandas as pd
from random import randint, sample
from copy import deepcopy
from functools import lru_cache
import nltk
from textaugment import EDA, Translate, Wordnet


class DatasetTransformer:
    def __init__(self, config: dict) -> None:
        """
        Config can have following fields:
            1. LOOK_N_TURNS:                int (-1 to look the whole previous turns)
            2. ENSURE_ALTERNATING_ROLES:    bool
            3. AUGMENT_WHEN_UPSAMPLE:       bool
        """
        self.look_n_turns: int = config['LOOK_N_TURNS']
        self.ensure_alternating_roles: bool = config['ENSURE_ALTERNATING_ROLES']
        self.augment_when_upsample: bool = config['AUGMENT_WHEN_UPSAMPLE']
        if self.augment_when_upsample:
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('punkt')
            self.eda = EDA()
            # Translate class is not initalised with languages yet
            self.translate = Translate

        self.dataset_dir_path = os.path.join(
            Path(os.path.dirname(os.path.realpath(__file__))).parent, 'dataset')

    @staticmethod
    def get_main_score(scores: list) -> int:
        """
        Returns most frequent score.
        In Weiwei's paper, they just return score, without adding 1 again,
            making the score range 0-4.
        However, we keep the range as 1-5.
        """
        number = [0, 0, 0, 0, 0]
        for item in scores:
            number[item] += 1
        score = np.argmax(number)
        return score + 1

    @staticmethod
    def alternate_roles(df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a pandas DataFrame that has no same roles in a row.
        """
        # create an empty dataframe
        new_df = pd.DataFrame(columns=['Role', 'Text', 'Action', 'Score'],
                              index=[i for i in range(len(df))])

        last_role = ''
        last_index = 0
        for i, row in df.iterrows():
            role, text, action, scores = row.Role, row.Text, row.Action, row.Score
            if last_role == role and role == 'USER':
                if text != 'OVERALL':
                    # append current text to the previous text
                    new_df.at[last_index, 'Text'] += f" {text}"
                    # TODO: consider the set. can overlap
                    new_df.at[last_index, 'Action'] += f" , {action}"
                    new_df.at[last_index, 'Score'] = scores

            else:
                new_df.at[i] = row
                last_index = i

            last_role = role

        new_df = new_df.dropna()

        return new_df

    def _transform_v1(self, dataset: str) -> tuple:
        dataset_path = os.path.join(self.dataset_dir_path, f'{dataset}.txt')
        raw = [line[:-1]
               for line in open(dataset_path, encoding='utf-8')]
        data = []
        for line in raw:
            if line == '':
                data.append([])
            else:
                data[-1].append(line)

        # Now, 'data' looks like:
        # [[line, line, line...], [line, line...], [line, line...]... ]
        # list of sessions; turns in a session

        histories = []
        satisfactions = []
        actions = []
        actions_set = set()
        utterances = []
        for session in data:
            his_input_ids = []
            for turn in session:
                # action can be an empty string
                role, text, action, score = turn.split('\t')

                text = text.strip()
                if text != 'OVERALL':
                    scores = score.split(',')
                    action = action.split(',')
                    action = action[0]
                    if role.upper() == 'USER' and len(his_input_ids) != 0:
                        # histories input
                        if self.look_n_turns == -1:
                            histories.append(deepcopy(' '.join(his_input_ids)))
                        else:
                            # Only look at N previous turns
                            histories.append(
                                deepcopy(' '.join(his_input_ids[-self.look_n_turns:])))

                        # utterance generation labels
                        utterances.append(text)

                        # satisfaction labels
                        satisfactions.append(self.get_main_score(
                            [int(item) - 1 for item in scores]))

                        # action labels
                        action = action.strip()
                        action = "None" if action == '' else action
                        if action not in actions_set:
                            actions_set.add(action)
                        actions.append(action)

                    his_input_ids.append(text)

        return_data = (histories, satisfactions,
                       actions, actions_set, utterances)

        return return_data

    def _transform_v2(self, dataset: str) -> tuple:
        dataset_path = os.path.join(self.dataset_dir_path, f'{dataset}.txt')
        dataset_df = pd.read_csv(dataset_path, sep='\t', header=None)
        dataset_df.columns = ['Role', 'Text', 'Action', 'Score']
        dataset_df = dataset_df.fillna('')

        if self.ensure_alternating_roles:
            dataset_df = self.alternate_roles(dataset_df)

        histories = []
        satisfactions = []
        actions = []
        actions_set = set()
        utterances = []

        history = []
        for i, row in dataset_df.iterrows():
            role, text, action, scores = row.Role, row.Text, row.Action, row.Score
            text = text.strip()

            if text != 'OVERALL':
                if role == 'USER' and len(history) != 0:
                    # preprocess
                    action = action.strip()
                    action = "None" if action == '' else action
                    scores = scores.split(',')
                    score = self.get_main_score(
                        [int(item) - 1 for item in scores])

                    # histories input
                    if self.look_n_turns == -1:
                        histories.append(deepcopy(' '.join(history)))
                    else:
                        # (Only look at N previous turns)
                        histories.append(
                            deepcopy(' '.join(history[-self.look_n_turns:])))

                    # utterance generation labels
                    utterances.append(text)

                    # satisfaction labels
                    satisfactions.append(score)

                    # action labels
                    if action not in actions_set:
                        actions_set.add(action)
                    actions.append(action)

                history.append(text)

            else:
                # clearing for the next session
                history = []

        return_data = (histories, satisfactions,
                       actions, actions_set, utterances)

        return return_data

    def transform(self, dataset: str):
        """
        Returns a tuple of (histories, satisfactions, actions, actions_set, utterances)
        """
        if self.ensure_alternating_roles:
            return self._transform_v2(dataset=dataset)
        else:
            return self._transform_v1(dataset=dataset)

    @lru_cache(maxsize=None)
    def adjust_token_length(self, text, max_length=512):
        """
        Recursion function that truncates the former part of the text to ensure the maximum token length.
        Returns a truncated text that is less than max token length.
        """
        final_text = deepcopy(text)
        is_valid, overflow_amount = check_token_length(text, max_length)
        if is_valid:
            return final_text
        else:
            # heuristic to cut the first 10 chars multiplied by overflowing token amount
            truncated_text = text[overflow_amount * 10:]
            truncated_text = truncated_text.strip().split(' ')[1:]
            final_text = deepcopy(truncated_text)
            self.adjust_token_length(truncated_text, max_length)

    def to_mtl_df(self, dataset: str) -> pd.DataFrame:
        histories, satisfactions, actions, _, utterances = self.transform(
            dataset)
        assert len(histories) == len(satisfactions) == \
            len(actions) == len(utterances)

        _prefix = ['sat: ' for _ in range(len(histories))] + \
            ['act: ' for _ in range(len(histories))] + \
            ['utt: ' for _ in range(len(histories))]
        _input = histories * 3
        _target = satisfactions + actions + utterances

        assert len(_prefix) == len(_input) == len(_target)

        df_data = {
            'prefix': _prefix,
            'input_text': _input,
            'target_text': _target
        }

        return pd.DataFrame(df_data)

    def upsample_non_3(self):
        """
        Upsampling non-3 rating data while augmenting the texts
        """
        pass

    def augment_text(self, text) -> str:
        """
        Available methods:
            1. random_deltion
            2. random_swap
            3. random_insertion
            4. synonym_replacement (wordnet based)
            5. back translation
        """
        rand_num = randint(1, 5)
        rand_num = 5
        if rand_num == 1:
            augmented_text = self.eda.random_deletion(text, p=0.2)
        elif rand_num == 2:
            augmented_text = self.eda.random_swap(text,
                                                  n=1 if int(len(text.split())*0.05) == 0 else int(len(text.split())*0.05))
        elif rand_num == 3:
            augmented_text = self.eda.random_insertion(text,
                                                       n=1 if int(len(text.split())*0.05) == 0 else int(len(text.split())*0.05))
        elif rand_num == 4:
            augmented_text = self.eda.synonym_replacement(text,
                                                          n=1 if int(len(text.split())*0.1) == 0 else int(len(text.split())*0.1))
        else:
            target_lang = sample(['ko', 'it', 'fa', 'es', 'el', 'la'], k=1)[0]
            try:
                augmented_text = self.translate(src='en',
                                                to=target_lang).augment(text)
            except HTTPError:
                augmented_text = self.eda.synonym_replacement(text,
                                                              n=1 if int(len(text.split())*0.1) == 0 else int(len(text.split())*0.1))

        return augmented_text


if __name__ == "__main__":
    dataset_config = {
        'LOOK_N_TURNS': 10,
        'ENSURE_ALTERNATING_ROLES': True,
        'AUGMENT_WHEN_UPSAMPLE': True
    }
    dataset_transformer = DatasetTransformer(config=dataset_config)

    # dataset_dir_path = dataset_transformer.dataset_dir_path
    # for dataset in ['CCPE', 'MWOZ', 'ReDial', 'SGD']:
    #     df = dataset_transformer.to_mtl_df(dataset)
    #     df.to_csv(os.path.join(dataset_dir_path,
    #               f'./{dataset}_df.csv'), index=False)

    sample_text = '''I'm looking for a cheap restaurant in the east part of town. 
    the missing sock is a nice restaurant in the east part of town in the cheap price range What is the address and phone number? 
    The address of The Missing Sock is Finders Corner Newmarket Road and the phone number is 01223 812660. 
    May I help you with anything else today?'''
    augmented_text = dataset_transformer.augment_text(sample_text)
    print(augmented_text)
