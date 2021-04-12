"""
Author : Eddie, Jeon
Editor : Jin Uk, Cho
Last update : 10th, Apr, 2021
"""

import random
import re


consonant = [
    "ㄱ",    "ㄲ",    "ㄴ",    "ㄷ",    "ㄸ",    "ㄹ",
    "ㅁ",    "ㅂ",    "ㅃ",    "ㅅ",    "ㅆ",    "ㅇ",
    "ㅈ",    "ㅉ",    "ㅊ",    "ㅋ",    "ㅌ",    "ㅍ",
    "ㅎ",
]
vowel = [
    "ㅏ",    "ㅐ",    "ㅑ",    "ㅒ",    "ㅓ",    "ㅔ",
    "ㅕ",    "ㅖ",    "ㅗ",    "ㅘ",    "ㅙ",    "ㅚ",
    "ㅛ",    "ㅜ",    "ㅝ",    "ㅞ",    "ㅟ",    "ㅠ",
    "ㅡ",    "ㅢ",    "ㅣ",
]
final_consonant = [
    " ",    "ㄱ",    "ㄲ",    "ㄳ",    "ㄴ",    "ㄵ",
    "ㄶ",    "ㄷ",    "ㄹ",    "ㄺ",    "ㄻ",    "ㄼ",
    "ㄽ",    "ㄾ",    "ㄿ",    "ㅀ",    "ㅁ",    "ㅂ",
    "ㅄ",    "ㅅ",    "ㅆ",    "ㅇ",    "ㅈ",    "ㅊ",
    "ㅋ",    "ㅌ",    "ㅍ",    "ㅎ",
]
exceptions = ["ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅗ"]
pairs = {
    "ㅏ": "ㅑ",
    "ㅑ": "ㅏ",
    "ㅓ": "ㅕ",
    "ㅕ": "ㅓ",
    "ㅗ": "ㅛ",
    "ㅛ": "ㅗ",
    "ㅜ": "ㅠ",
    "ㅠ": "ㅜ",
}

def jamo_split(char):
    base = ord(char) - ord("가")
    c = base // 588
    v = (base - 588 * c) // 28
    f_c = base - 588 * c - 28 * v
    return [consonant[c], vowel[v], final_consonant[f_c]]

def jamo_merge(jamo_list):
    c, v, f_c = [
        _list.index(j)
        for _list, j in zip(
            [consonant, vowel, final_consonant], jamo_list
        )
    ]
    return chr(f_c + 588 * c + 28 * v + ord("가"))

def noise_generate(content, prob=None, option="01"):
    """ 
    generate noise using jamo split and merge
    option
        01: (분리 변형) 가 -> ㄱㅏ, default prob is 0.3
        02: (대체 변형) 가 -> 갸 or 까, default prob is 0.8
    """
    assert option in ["01", "02"], "option must be either '01' or '02'."
    if not prob: # if probability is not given
        prob = 0.3 if option == "01" else 0.8

    if option == "01":
        condition = lambda xlist: ((xlist[-1] == " ") and (xlist[-2] not in exceptions))

        output = [
            jamo_split(ch) if re.match("[가-힣]", ch) else [ch, "", ""]
            for ch in content
        ]
        output = [
            "".join(out).strip()
            if condition(out) and (random.random() < prob)
            else content[i]
            for i, out in enumerate(output)
        ]
    else:
        condition = lambda xlist: ((xlist[-1] == " ") and (xlist[-2] in pairs))
        output = [
            jamo_split(ch) if re.match("[가-힣]", ch) else [ch, "", ""]
            for ch in content
        ]
        output = [
            jamo_merge([out[0], pairs[out[1]], out[2]]) 
            if condition(out) and (random.random() < prob)
            else content[i]
            for i, out in enumerate(output)
        ]

    return "".join(output)
