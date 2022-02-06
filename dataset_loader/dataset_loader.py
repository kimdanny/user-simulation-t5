"""
Adapted from https://github.com/sunnweiwei/user-satisfaction-simulation

Difference from original:
    1. Avoid empty string prediction.
        For example, in SGD, USER starts the conversation, so the first history is empty string.
        We omitted these cases.
    2. Do not predict OVERALL.
        Generating a user utterance "OVERALL" does not make sense, 
        so we omiited the last turn (OVERALL) of each session.
    
    3. (Optional) Can specify the N turns of conversation histories to look at.
        set N to -1 to look at the whole previous turns
    4. (Optional) Can ensure that two roles alternate with each other.
        i.e. Transforms cases like SYSTEM-USER-USER-USER-SYSTEM to SYSTEM-USER-SYSTEM
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from copy import deepcopy


class DatasetLoader:
    def __init__(self, config: dict) -> None:
        """
        Config can have following fields:
            1. LOOK_N_TURNS:                int (-1 to look the whole previous turns)
            2. ENSURE_ALTERNATING_ROLES:    bool
        """
        self.look_n_turns: int = config['LOOK_N_TURNS']
        self.ensure_alternating_roles: bool = config['ENSURE_ALTERNATING_ROLES']

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
                    # append to the previous row
                    new_df.at[last_index, 'Text'] += f" {text}"
                    new_df.at[last_index, 'Action'] += f",{action}"
                    new_df.at[last_index, 'Score'] = scores

            else:
                new_df.at[i] = row
                last_index = i

            last_role = role

        new_df = new_df.dropna()

        return new_df

    def _load_v1(self, dataset: str) -> tuple:
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

    def _load_v2(self, dataset: str) -> tuple:
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

    def load(self, dataset: str):
        """
        Returns a tuple of (histories, satisfactions, actions, actions_set, utterances)
        """
        if self.ensure_alternating_roles:
            return self._load_v2(dataset=dataset)
        else:
            return self._load_v1(dataset=dataset)


if __name__ == "__main__":
    dataset_config = {
        'LOOK_N_TURNS': 5,
        'ENSURE_ALTERNATING_ROLES': True
    }
    loader = DatasetLoader(config=dataset_config)

    histories, satisfactions, actions, actions_set, utterances = loader.load(
        'SGD')

    print(len(histories), len(satisfactions),
          len(actions), len(actions_set), len(utterances))

    print(histories[:3])
    print()
    print(utterances[:3])
