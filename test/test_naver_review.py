from ktextaug.backtranslate import BackTranslate
from ktextaug.random_swap import tokenize, random_swap
from tqdm import tqdm, trange
import os
import pandas as pd

def main(file_path='review_total.csv', save_path=None):
    translator = BackTranslate()

    df = pd.read_csv(file_path, index_col=False)
    # print(df.head())
    #
    # total = 0
    #
    # for i in range(len(df)):
    #     try:
    #         if '|' in df.iloc[i, 1]:
    #             print(df.iloc[i, 1])
    #         total += 1
    #         # total += len(df.iloc[i, 1])
    #     except Exception as e:
    #         print(e)
    #         print(df.iloc[i, 1])
    # print(total)
    #
    # error = 0
    row_list_org = []
    for i in trange(len(df), desc="Tokenization"):
        try:
            tmp = {'review': '|'.join(tokenize(df.iloc[i, 1])),
                    'label': df.iloc[i, 2]}
            row_list_org.append(tmp)
        except Exception as e:
            print(f"Index : {i}")
            print(e)
            print(df.iloc[i, 1])
            # error += 1
            continue
    df_org = pd.DataFrame(row_list_org)
    # print(error)

    row_list_bt = []
    # for i in trange(len(df), desc="Backtranslation"):
    for i in trange(len(df), desc="Backtranslation"):
        try:
            review = '|'.join(tokenize(translator.backtranslate(df.iloc[i, 1])))
            tmp = {'review': review,
                   'label': df.iloc[i, 2]}
            row_list_bt.append(tmp)
        except Exception as e:
            print(f"Index : {i}")
            print(e)
            print(df.iloc[i, 1])
            continue
    df_bt = pd.DataFrame(row_list_bt)


    row_list_rs = []
    for i in trange(len(df), desc="Random_swap"):
        try:
            tok_words = tokenize(df.iloc[i, 1])
            tmp = {'review': '|'.join(random_swap(tok_words, int(0.2 * len(tok_words)))),
                   'label': df.iloc[i, 2]}
            row_list_rs.append(tmp)
        except Exception as e:
            print(f"Index : {i}")
            print(e)
            print(df.iloc[i, 1])
            continue
    df_rs = pd.DataFrame(row_list_rs)

    result = df_org.append(df_bt, ignore_index=True)
    result = result.append(df_rs, ignore_index=True)
    result.to_csv("./src/data/aug_tok_train_s1000.csv")

if __name__ == '__main__':
    path = "./src/data/ratings_train_s1000.csv"
    main(file_path=path)