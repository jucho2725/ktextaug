import re
import random

consonant = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
vowel = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ',
         'ㅢ', 'ㅣ']
final_consonant = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ',
                   'ㅂ',
                   'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
pairs = {'ㅏ': 'ㅑ', 'ㅑ': 'ㅏ', 'ㅓ': 'ㅕ', 'ㅕ': 'ㅓ', 'ㅗ': 'ㅛ', 'ㅛ': 'ㅗ', 'ㅜ': 'ㅠ', 'ㅠ': 'ㅜ', }

oral_consonant = ['ㄱ', 'ㄷ', 'ㄹ', 'ㅂ', 'ㅅ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅎ']
nasal_consonant = ['ㅁ', 'ㄴ', 'ㅇ']
liquid_consonant = ['ㄹ']


def load_pairs(path):
    with open(path, 'r', encoding='utf-8') as r:
        contents = [l.split('\t') for l in r.read().split('\n')]
    dictionary = {k: v for k, v in contents if k != v and k.strip() != ''}
    return dictionary


def jamo_split(char):
    base = ord(char) - ord('가')
    c = base // 588
    v = (base - 588 * c) // 28
    f_c = base - 588 * c - 28 * v
    return [consonant[c], vowel[v], final_consonant[f_c]]


def jamo_merge(jamo_list):
    if jamo_list[1:] == ['', '']:
        return jamo_list[0]
    c, v, f_c = [_list.index(j) for _list, j in zip([consonant, vowel, final_consonant], jamo_list)]
    return chr(f_c + 588 * c + 28 * v + ord('가'))


def splitting_noise(content, prob=0.1):
    exceptions = ['ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅗ']
    condition = lambda xlist: ((xlist[-1] == ' ') and (xlist[-2] not in exceptions))

    output = [jamo_split(ch) if re.match('[가-힣]', ch) else [ch, '', ''] for ch in content]
    output = [''.join(out).strip() if condition(out) and (random.random() < prob) else content[i] for i, out in
              enumerate(output)]

    return ''.join(output)


def vowel_noise(content, prob=0.1):
    output = [jamo_split(ch) if re.match('[가-힣]', ch) else [ch, '', ''] for ch in content]
    condition = lambda xlist: ((xlist[-1] == ' ') and (xlist[-2] in pairs))
    output = [
        jamo_merge([out[0], pairs[out[1]], out[2]]) if condition(out) and (random.random() < prob) else
        content[i] for i, out in
        enumerate(output)]
    return ''.join(output)


def palatalization(fc, nc):
    palatal = {'ㄷ': 'ㅈ', 'ㅌ': 'ㅊ'}
    if (fc[-1] in palatal) and nc[:-1] == ['ㅇ', 'ㅣ']:
        nc[0] = palatal[fc[-1]]
        fc[-1] = ' '
    return fc, nc


def linking(fc, nc):
    formal_morpheme = [jamo_split(mor) for mor in ['이', '을', '를', '은', '았', '었', '아', '어']]
    links = {'ㄻ': 'ㄹㅁ', 'ㅄ': 'ㅂㅆ', 'ㄳ': 'ㄱㅅ', 'ㄽ': 'ㄹㅅ', 'ㅊ': ' ㅊ', 'ㅂ': ' ㅂ', 'ㅍ': ' ㅂ', 'ㄷ': ' ㄹ', 'ㄹ': ' ㄹ',
             'ㄹㅎ': ' ㄹ'}
    if (fc[-1] in links) and (nc in formal_morpheme):
        fc[-1], nc[0] = links[fc[-1]]
    return fc, nc


def liquidization(fc, nc):
    liquid_set = {'ㄴㄹ': 'ㄹㄹ', 'ㄹㄴ': 'ㄹㄹ', 'ㄾㄴ': 'ㄹㄹ'}
    exception_set = {'ㄴㄹㅕㄱ': 'ㄴㄴ'}

    if fc[-1] + ''.join(nc) in exception_set:
        fc[-1], nc[0] = exception_set[fc[-1] + ''.join(nc)]
        return fc, nc
    else:
        if fc[-1] + nc[0] in liquid_set:
            fc[-1], nc[0] = liquid_set[fc[-1] + nc[0]]
        return fc, nc


def nasalization(fc, nc):
    nasalization_set = {'ㅂㅁ': 'ㅁㅁ', 'ㄷㄴ': 'ㄴㄴ', 'ㄱㅁ': 'ㅇㅁ', 'ㄱㄴ': 'ㅇㄴ', 'ㅇㄹ': 'ㅇㄴ',
                        'ㅁㄹ': 'ㅁㄴ', 'ㄲㄴ': 'ㅇㄴ', 'ㅂㄹ': 'ㅁㄴ', 'ㄱㄹ': 'ㅇㄴ', 'ㅊㄹ': 'ㄴㄴ',
                        'ㄺㄴ': 'ㅇㄴ', 'ㅍㄴ': 'ㅁㄴ'}
    fc_c = fc[-1] + nc[0]
    if fc_c in nasalization_set:
        fc[-1], nc[0] = nasalization_set[fc_c]
    return fc, nc


def assimilation(fc, nc):
    # assimilation not employed in the nasalization function. each other has the similar rules.
    reverse_assimil = {'ㄺㄴ': 'ㅇㄴ'}
    fc_c = fc[-1] + nc[0]
    if fc_c in reverse_assimil:
        fc[-1], nc[0] = reverse_assimil[fc_c]
    return fc, nc


def phonological_process(content, prob=0.3):
    uncased = [jamo_split(ch) if re.match('[가-힣]', ch) else [ch, '', ''] for ch in content]

    for i in range(len(uncased) - 1):
        if random.random() < prob:
            uncased[i], uncased[i + 1] = palatalization(uncased[i], uncased[i + 1])
            uncased[i], uncased[i + 1] = linking(uncased[i], uncased[i + 1])
            uncased[i], uncased[i + 1] = liquidization(uncased[i], uncased[i + 1])
            uncased[i], uncased[i + 1] = nasalization(uncased[i], uncased[i + 1])

    content = ''.join([jamo_merge(unc) for unc in uncased])
    return content


def noise_add(text, prob, tokenizer, rng, noise_mode=['jamo_split', 'vowel_change', 'phonological_change'],
              **kwargs):
    if isinstance(text, list):
        text = tokenizer.convert_tokens_to_string(text)

    fns_dict = {'jamo_split': splitting_noise,
                'vowel_change': vowel_noise,
                'phonological_change': phonological_process}

    noised_corpus = text
    idx = 0
    while noised_corpus == text or idx < len(noise_mode):
        noised_corpus = fns_dict[rng.sample(noise_mode, k=1)[0]](text, prob=prob)
        idx += 1
    return noised_corpus
