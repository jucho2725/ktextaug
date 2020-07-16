from ktextaug.aug_util import tokenize
from tqdm import tqdm, trange
import pandas as pd


def main(file_path):
    df = pd.read_table(file_path, encoding='utf-8')

    row_list_org = []
    for i in trange(len(df), desc="Tokenization"):
        try:
            tmp = {'review': '|'.join(tokenize(df.iloc[i, 1])),
                    'label': df.iloc[i, 2]}
            row_list_org.append(tmp)
        except Exception as e:
            print(f"Index : {i}")
            print(e)
            print(df.iloc[i, :])
            continue
    df_org = pd.DataFrame(row_list_org)
    df_org.to_csv("../src/data/tok_train_all.csv")

if __name__ == '__main__':
    path = "../src/data/ratings_train.txt"
    main(file_path=path)