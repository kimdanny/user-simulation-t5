import os
from pathlib import Path
import numpy as np
import pandas as pd


def transform_df(df) -> pd.DataFrame:
    """
    1. split input_text into prefix column and input_text column
    2. remove </s> from the end
    """
    # split
    df[['prefix', 'input_text']] = df['input_text'].str.split(': ', 1, expand=True)
    
    # replace
    df['input_text'] = df['input_text'].map(lambda x: x.replace('</s>', ''))
    df['target_text'] = df['target_text'].map(lambda x: x.replace('</s>', ''))

    # remove excess white spaces
    df['prefix'] = df['prefix'].apply(lambda x: " ".join(x.split()))
    df['input_text'] = df['input_text'].apply(lambda x: " ".join(x.split()))
    df['target_text'] = df['target_text'].apply(lambda x: " ".join(x.split()))

    df = df[['prefix', 'input_text', 'target_text']]
    
    return df


if __name__ == "__main__":
    TASK = 'act-sat_no-alt'
    DATASET = 'MWOZ'
    dataset_dir_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, 'dataset', TASK)
    dataset_path = os.path.join(dataset_dir_path, f'{DATASET}_df.csv')

    df = pd.read_csv(dataset_path, index_col=False).astype(str)
    df = transform_df(df)
