import random
import re


class NoiseGenerator(object):
    def __init__(self):
        self.consonant = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.vowel = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ',
                      'ㅢ', 'ㅣ']
        self.final_consonant = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ',
                                'ㅂ',
                                'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.exceptions = ['ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅗ']
        self.pairs = {'ㅏ': 'ㅑ', 'ㅑ': 'ㅏ', 'ㅓ': 'ㅕ', 'ㅕ': 'ㅓ', 'ㅗ': 'ㅛ', 'ㅛ': 'ㅗ', 'ㅜ': 'ㅠ', 'ㅠ': 'ㅜ', }

    def jamo_split(self, char):
        base = ord(char) - ord('가')
        c = base // 588
        v = (base - 588 * c) // 28
        f_c = base - 588 * c - 28 * v
        return [self.consonant[c], self.vowel[v], self.final_consonant[f_c]]

    def jamo_merge(self, jamo_list):
        c, v, f_c = [_list.index(j) for _list, j in zip([self.consonant, self.vowel, self.final_consonant], jamo_list)]
        return chr(f_c + 588 * c + 28 * v + ord('가'))

    def noise_generate1(self, content, prob=0.3):
        condition = lambda xlist: ((xlist[-1] == ' ') and (xlist[-2] not in self.exceptions))

        output = [self.jamo_split(ch) if re.match('[가-힣]', ch) else [ch, '', ''] for ch in content]
        output = [''.join(out).strip() if condition(out) and (random.random() < prob) else content[i] for i, out in
                  enumerate(output)]

        return ''.join(output)

    def noise_generate2(self, content, prob=0.3):
        output = [self.jamo_split(ch) if re.match('[가-힣]', ch) else [ch, '', ''] for ch in content]
        condition = lambda xlist: ((xlist[-1] == ' ') and (xlist[-2] in self.pairs))
        output = [
            self.jamo_merge([out[0], self.pairs[out[1]], out[2]]) if condition(out) and (random.random() < prob) else
            content[i] for i, out in
            enumerate(output)]
        return ''.join(output)
