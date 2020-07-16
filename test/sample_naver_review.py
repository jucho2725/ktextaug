import pandas as pd
from ktextaug.backtranslate import BackTranslate
from ktextaug.aug_util import tokenize
from tqdm import tqdm
import random

random_state = list(range(10))
translator = BackTranslate()

def preprocess(df: pd.DataFrame):
    df.dropna(inplace=True) # 5개 제거

    nonalpha_l = []
    for idx in range(len(df)):
        try:
            tmp = float(df.iloc[idx, 1])
            nonalpha_l.append(idx)
        except:
            continue
    # print(len(nonalpha_l)) # 88 개 제거
    df.drop(inplace=True, axis=0, index=nonalpha_l)
    df.reset_index(drop=True)

    # len_l = [idx for idx in range(len(df)) if len(df.iloc[idx, 1]) < 3]
    return df

def check_dup(df: pd.DataFrame, source: pd.DataFrame):
    df.reset_index(drop=True)

    drop_l = []
    pbar = tqdm(total=len(df), desc="Checking Duplicates")
    for i in range(len(df)):
        if df.iloc[i, 1] == translator.backtranslate(df.iloc[i, 1]):
            drop_l.append(i)
        pbar.update(1)

    print("num of drop rows ", len(drop_l))
    if not drop_l:
        return df
    else:
        df.drop(inplace=True, axis=0, index=[df.index[i] for i in drop_l])
        df.reset_index(drop=True)
        tmp = source.sample(n=len(drop_l), random_state=random.sample(random_state, 1)[0])
        return df.append(check_dup(tmp, source))

    return df


def main(train, test ,num_of_sample):
    # train 불러오고 pos 와 neg 를 나눔
    df = pd.read_table(train)
    df.reset_index()
    df = preprocess(df)

    df_pos = df[df['label'] == 1]
    df_neg = df[df['label'] == 0]

    # test 불러오고 test 내 pos 와 neg 의 비율 구함
    df_test = pd.read_table(test)
    rate_pos = len(df_test[df_test['label'] == 1]) / len(df_test)

    rs_p = df_pos.sample(n= int(num_of_sample*rate_pos), random_state=random.sample(random_state, 1)[0])
    # rs_p = check_dup(rs_p, df)

    rs_n = df_neg.sample(n= num_of_sample - int(num_of_sample*rate_pos), random_state=random.sample(random_state, 1)[0])
    # rs_n = check_dup(rs_n, df)

    result = rs_p.append(rs_n)
    result = result.sample(frac=1).reset_index(drop=True) # shuffle
    print(len(result))
    #
    result.to_csv("../src/data/ratings_train_s10000.csv", index=False)
    print("done")

if __name__ == '__main__':
    train_path = "../src/data/ratings_train.txt"
    test_path = "../src/data/ratings_test.txt"
    num_of_sample = 10000
    main(train=train_path,
         test=test_path,
         num_of_sample=num_of_sample)