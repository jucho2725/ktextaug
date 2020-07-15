from ktextaug.Summarize import BackTranslate, random_swap, random_deletion, random_insertion, synonym_replacement
from ktextaug.aug_util import tokenize


from tqdm import tqdm, trange
import os
import pandas as pd
import random
random.seed(2020)

def main(file_path='review_total.csv', save_path=None):

    # init setting
    translator = BackTranslate()
    df = pd.read_csv(file_path, index_col=False)

    # params setting
    num_per_tech = 2 # is manually set.

    row_list_org = []
    for i in trange(len(df), desc="Tokenization"):
        try:
            tmp = {'id': df.iloc[i, 0],
                   'review': '|'.join(tokenize(df.iloc[i, 1])),
                    'label': df.iloc[i, 2]}
            row_list_org.append(tmp)
        except Exception as e:
            print(f"Index : {i}")
            print(e)
            print(df.iloc[i, 1])
            # error += 1
            continue
    df_org = pd.DataFrame(row_list_org)
    print(f"Checking : length of origin data {len(df_org)}")
    df_org.to_csv("../src/data/org_tok_train_s1000.csv")

    ### bt ###
    row_list_bt = []
    for i in trange(len(df), desc="Backtranslation 1: en"):
        try:
            review = '|'.join(tokenize(translator.backtranslate(df.iloc[i, 1])))
            tmp = {'id': df.iloc[i, 0],
                   'review': review,
                   'label': df.iloc[i, 2]}
            row_list_bt.append(tmp)
        except Exception as e:
            print(f"Index : {i}")
            print(e)
            print(df.iloc[i, 1])
            continue
    # translate to another language
    # needs to put more language in the basket
    language_basket = ['ja', 'fr']
    target_language = random.sample(language_basket, 1)[0]

    for i in trange(len(df), desc=f"Backtranslation 2: {target_language}"):
        try:
            review = '|'.join(tokenize(translator.backtranslate(df.iloc[i, 1], target_language=target_language)))
            tmp = {'id': df.iloc[i, 0],
                   'review': review,
                   'label': df.iloc[i, 2]}
            row_list_bt.append(tmp)
        except Exception as e:
            print(f"Index : {i}")
            print(e)
            print(df.iloc[i, 1])
            continue

    df_bt = pd.DataFrame(row_list_bt)
    bt = df_org.append(df_bt, ignore_index=True)
    bt.to_csv("../src/data/bt_tok_train_s1000.csv")
    print(f"Checking : length of origin + bt data {len(bt)}")


    ### random swap ###
    row_list_rs = []
    for _ in range(num_per_tech):
        for i in trange(len(df), desc="Random_swap"):
            try:
                tok_words = tokenize(df.iloc[i, 1])
                n_rs = max(1, int(0.1 * len(tok_words)))
                tmp = {'id': df.iloc[i, 0],
                       'review': '|'.join(random_swap(tok_words, n_rs)),
                       'label': df.iloc[i, 2]}
                row_list_rs.append(tmp)
            except Exception as e:
                print(f"Index : {i}")
                print(e)
                print(df.iloc[i, 1])
                continue
    df_rs = pd.DataFrame(row_list_rs)
    rs = df_org.append(df_rs, ignore_index=True)
    rs.to_csv("../src/data/rs_tok_train_s1000.csv")
    print(f"Checking : length of origin + rs data {len(rs)}")

    ### random insertion ###
    row_list_ri = []
    for _ in range(num_per_tech):
        for i in trange(len(df), desc="Random_insertion"):
            try:
                tok_words = tokenize(df.iloc[i, 1])
                n_ri = max(1, int(0.1 * len(tok_words)))
                tmp = {'id': df.iloc[i, 0],
                       'review': '|'.join(random_insertion(tok_words, n_ri)),
                       'label': df.iloc[i, 2]}
                row_list_ri.append(tmp)
            except Exception as e:
                print(f"Index : {i}")
                print(e)
                print(df.iloc[i, 1])
                continue
    df_ri = pd.DataFrame(row_list_ri)
    ri = df_org.append(df_ri, ignore_index=True)
    ri.to_csv("../src/data/ri_tok_train_s1000.csv")
    print(f"Checking : length of origin + ri data {len(ri)}")

    ### random deletion ###
    row_list_rd = []
    for _ in range(num_per_tech):
        for i in trange(len(df), desc="Random_deletion"):
            try:
                tok_words = tokenize(df.iloc[i, 1])
                p_rd = 0.1
                tmp = {'id': df.iloc[i, 0],
                       'review': '|'.join(random_deletion(tok_words, p_rd)),
                       'label': df.iloc[i, 2]}
                row_list_rd.append(tmp)
            except Exception as e:
                print(f"Index : {i}")
                print(e)
                print(df.iloc[i, 1])
                continue
    df_rd = pd.DataFrame(row_list_rd)
    rd = df_org.append(df_rd, ignore_index=True)
    rd.to_csv("../src/data/rd_tok_train_s1000.csv")
    print(f"Checking : length of origin + rd data {len(rd)}")

    ### synonym replacement ###
    row_list_sr = []
    for _ in range(num_per_tech):
        for i in trange(len(df), desc="Synonym replacement"):
            try:
                tok_words = tokenize(df.iloc[i, 1])
                n_sr = max(1, int(0.1 * len(tok_words)))
                tmp = {'id': df.iloc[i, 0],
                      'review': '|'.join(synonym_replacement(tok_words, n_sr)),
                       'label': df.iloc[i, 2]}
                row_list_sr.append(tmp)
            except Exception as e:
                print(f"Index : {i}")
                print(e)
                print(df.iloc[i, 1])
                continue
    df_sr = pd.DataFrame(row_list_sr)
    sr = df_org.append(df_sr, ignore_index=True)
    sr.to_csv("../src/data/sr_tok_train_s1000.csv")
    print(f"Checking : length of origin + sr data {len(sr)}")


    ### matching same id from noise file ###
    train_df = pd.read_table("../src/data/ratings_train.txt", encoding='utf-8') # 150000 data
    ndf = pd.read_csv("../src/data/noise_data.csv")
    sampled_df = train_df.loc[train_df['id'].isin(df['id'].tolist())]
    sampled_ndf = ndf.loc[sampled_df.index, :]

    row_list_noise = []
    for t in range(num_per_tech):
        for i in trange(len(sampled_ndf), desc="Matching noise data"):
            tmp = {'id': sampled_df.iloc[i, 0],
                  'review': '|'.join(tokenize(sampled_ndf.iloc[i, t])),
                   'label': sampled_df.iloc[i, 2]}
            row_list_noise.append(tmp)
    df_noise = pd.DataFrame(row_list_noise)
    noise = df_org.append(df_noise, ignore_index=True)
    noise.to_csv("../src/data/noise_tok_train_s1000.csv")
    print(f"Checking : length of origin + noise data {len(noise)}")

    total = df_org.append(df_rs, ignore_index=True)
    total = total.append(df_ri, ignore_index=True)
    total = total.append(df_rd, ignore_index=True)
    total = total.append(df_sr, ignore_index=True)
    total = total.append(df_bt, ignore_index=True)
    total = total.append(df_noise, ignore_index=True)
    total.to_csv("../src/data/total_tok_train_s1000.csv")
    print(f"Checking : length of origin + bt data {len(total)}")

if __name__ == '__main__':
    path = "../src/data/ratings_train_s1000.csv"
    main(file_path=path)