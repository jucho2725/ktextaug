from test.Summarize import (
    random_swap,
    random_deletion,
    random_insertion,
    synonym_replacement,
)

# from ktextaug.aug_util import tokenize

from tqdm import trange
import pandas as pd
import random
import multiprocessing

# from transformers import ElectraTokenizer
# tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-multilingual-cased", do_lower_case=False
)

random.seed(2020)

def tokenize(text):
    return tokenizer.tokenize(text)

def ri_proc(i, df):
    ### random insertion ###
    try:
        tok_words = tokenize(df.iloc[i, 1])
        n_ri = max(1, int(0.1 * len(tok_words)))
        tmp = {
            "id": df.iloc[i, 0],
            "review": "|".join(random_insertion(tok_words, n_ri)),
            "label": df.iloc[i, 2],
        }
    except Exception as e:
        print(f"Index : {i}")
        print(e)
        print(df.iloc[i, 1])
        tmp = {
            "id": df.iloc[i, 0],
            "review": "|".join(tok_words),
            "label": df.iloc[i, 2],
        }
    return tmp


def sr_proc(i, df):
    try:
        tok_words = tokenize(df.iloc[i, 1])
        n_sr = max(1, int(0.1 * len(tok_words)))
        tmp = {
            "id": df.iloc[i, 0],
            "review": "|".join(synonym_replacement(tok_words, n_sr)),
            "label": df.iloc[i, 2],
        }
    except Exception as e:
        print(f"Index : {i}")
        print(e)
        print(df.iloc[i, 1])
        tmp = {
            "id": df.iloc[i, 0],
            "review": "|".join(tok_words),
            "label": df.iloc[i, 2],
        }
    return tmp


def main(file_path="review_total.csv", save_path=None):

    # init setting
    df = pd.read_csv(file_path, index_col=False)

    # params setting
    num_per_tech = 2  # is manually set.
    sample_type = "1000"
    sample_num = "1"
    model = "multilingual"

    row_list_org = []
    for i in trange(len(df), desc="Tokenization"):
        try:
            tmp = {
                "id": df.iloc[i, 0],
                "review": "|".join(tokenize(df.iloc[i, 1])),
                "label": df.iloc[i, 2],
            }
            row_list_org.append(tmp)
        except Exception as e:
            print(f"Index : {i}")
            print(e)
            print(df.iloc[i, 1])
            # error += 1
            continue
    df_org = pd.DataFrame(row_list_org)
    print(f"Checking : length of origin data {len(df_org)}")
    df_org.to_csv(
        f"../src/data/{sample_type}/{model}/org_{model[:3]}_train_s{sample_type}_{sample_num}.csv",
        index=False,
    )

    import parmap

    num_cores = multiprocessing.cpu_count()

    input_indices = list(range(len(df)))
    input_indices = [[x] for x in input_indices]

    li_ri = parmap.starmap(
        ri_proc, input_indices, df, pm_pbar=True, pm_processes=num_cores
    )
    li_sr = parmap.starmap(
        sr_proc, input_indices, df, pm_pbar=True, pm_processes=num_cores
    )

    ### random insertion ###
    df_ri = pd.DataFrame(li_ri)
    ri = df_org.append(df_ri, ignore_index=True)
    ri.to_csv(
        f"../src/data/{sample_type}/{model}/ri_{model[:3]}_train_s{sample_type}_{sample_num}.csv",
        index=False,
    )
    print(f"Checking : length of origin + ri data {len(ri)}")
    ### synonym replacement ###
    df_sr = pd.DataFrame(li_sr)
    sr = df_org.append(df_sr, ignore_index=True)
    sr.to_csv(
        f"../src/data/{sample_type}/{model}/sr_{model[:3]}_train_s{sample_type}_{sample_num}.csv",
        index=False,
    )
    print(f"Checking : length of origin + sr data {len(sr)}")

    row_list_bt = []
    df_nottok_bt = pd.read_csv("../src/data/bt_nottok_train_s10000.csv", index_col=False)
    # print(df_nottok_bt.head())
    for i in trange(len(df_nottok_bt), desc=f"Tokenize from backtranslated data: "):
        try:
            review = "|".join(tokenize(df_nottok_bt.iloc[i, 1]))
            tmp = {
                "id": df_nottok_bt.iloc[i, 0],
                "review": review,
                "label": df_nottok_bt.iloc[i, 2],
            }
            row_list_bt.append(tmp)
        except Exception as e:
            print(f"Index : {i}")
            print(e)
            print(df.iloc[i, 1])
            continue

    df_bt = pd.DataFrame(row_list_bt)
    bt = df_org.append(df_bt, ignore_index=True)
    bt.to_csv(
        f"../src/data/{sample_type}/{model}/bt_{model[:3]}_train_s{sample_type}_{sample_num}.csv",
        index=False,
    )
    print(f"Checking : length of origin + bt data {len(bt)}")

    ### random swap ###
    row_list_rs = []
    for _ in range(num_per_tech):
        for i in trange(len(df), desc="Random_swap"):
            try:
                tok_words = tokenize(df.iloc[i, 1])
                n_rs = max(1, int(0.1 * len(tok_words)))
                tmp = {
                    "id": df.iloc[i, 0],
                    "review": "|".join(random_swap(tok_words, n_rs)),
                    "label": df.iloc[i, 2],
                }
                row_list_rs.append(tmp)
            except Exception as e:
                print(f"Index : {i}")
                print(e)
                print(df.iloc[i, 1])
                continue
    df_rs = pd.DataFrame(row_list_rs)
    rs = df_org.append(df_rs, ignore_index=True)
    rs.to_csv(
        f"../src/data/{sample_type}/{model}/rs_{model[:3]}_train_s{sample_type}_{sample_num}.csv",
        index=False,
    )
    print(f"Checking : length of origin + rs data {len(rs)}")

    ### random deletion ###
    row_list_rd = []
    for _ in range(num_per_tech):
        for i in trange(len(df), desc="Random_deletion"):
            try:
                tok_words = tokenize(df.iloc[i, 1])
                p_rd = 0.1
                tmp = {
                    "id": df.iloc[i, 0],
                    "review": "|".join(random_deletion(tok_words, p_rd)),
                    "label": df.iloc[i, 2],
                }
                row_list_rd.append(tmp)
            except Exception as e:
                print(f"Index : {i}")
                print(e)
                print(df.iloc[i, 1])
                continue
    df_rd = pd.DataFrame(row_list_rd)
    rd = df_org.append(df_rd, ignore_index=True)
    rd.to_csv(
        f"../src/data/{sample_type}/{model}/rd_{model[:3]}_train_s{sample_type}_{sample_num}.csv",
        index=False,
    )
    print(f"Checking : length of origin + rd data {len(rd)}")

    ### matching same id from noise file ###
    train_df = pd.read_table(
        "../src/data/ratings_train.txt", encoding="utf-8"
    )  # 150000 data
    ndf = pd.read_csv("../src/data/noise_data.csv")
    sampled_df = train_df.loc[train_df["id"].isin(df["id"].tolist())]
    sampled_ndf = ndf.loc[sampled_df.index, :]

    row_list_noise = []
    for t in range(num_per_tech):
        for i in trange(len(sampled_ndf), desc="Matching noise data"):
            tmp = {
                "id": sampled_df.iloc[i, 0],
                "review": "|".join(tokenize(sampled_ndf.iloc[i, t])),
                "label": sampled_df.iloc[i, 2],
            }
            row_list_noise.append(tmp)
    df_noise = pd.DataFrame(row_list_noise)
    noise = df_org.append(df_noise, ignore_index=True)
    noise.to_csv(
        f"../src/data/{sample_type}/{model}/noise_{model[:3]}_train_s{sample_type}_{sample_num}.csv",
        index=False,
    )
    print(f"Checking : length of origin + noise data {len(noise)}")

    total = df_org.append(df_rs, ignore_index=True)
    total = total.append(df_ri, ignore_index=True)
    total = total.append(df_rd, ignore_index=True)
    total = total.append(df_sr, ignore_index=True)
    total = total.append(df_bt, ignore_index=True)
    total = total.append(df_noise, ignore_index=True)
    total.to_csv(
        f"../src/data/{sample_type}/{model}/total_{model[:3]}_train_s{sample_type}_{sample_num}.csv",
        index=False,
    )
    print(f"Checking : length of origin + bt data {len(total)}")


if __name__ == "__main__":
    path = "../src/data/ratings_train_s1000.csv"
    main(file_path=path)
