import torch.cuda
from torch import nn
from tqdm.auto import trange
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

from transformers import MarianMTModel, MarianTokenizer


class BackTranslation(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    :param target_language: support lanaguage [de, en, es, fi, fr, hu, ru, sv]
    see more details in https://huggingface.co/Helsinki-NLP
   """
    def __init__(self, target_language: str,
                 device='auto'):
        super(BackTranslation, self).__init__()

        target_model_name = f'Helsinki-NLP/opus-mt-ko-{target_language}'
        self.tar_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
        self.tar_model = MarianMTModel.from_pretrained(target_model_name)

        source_model_name = f'Helsinki-NLP/opus-mt-{target_language}-ko'
        self.src_tokenizer = MarianTokenizer.from_pretrained(source_model_name)
        self.src_model = MarianMTModel.from_pretrained(source_model_name)

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.tar_model.to(device)
        self.src_model.to(device)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens,\
                         'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):              #{key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):      #Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])      #Sum of length of individual strings

    def back_translate(self, text_or_corpus: Union[List[str], str],
                       batch_size,
                       temperature=1.3,
                       show_progress_bar=False):
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel()==logging.INFO or logger.getEffectiveLevel()==logging.DEBUG)

        assert isinstance(text_or_corpus, (str, list))
        if isinstance(text_or_corpus, str):
            sentences = [text_or_corpus]
        else:
            sentences = text_or_corpus

        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        result = []
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            result.append(self._back_translate(sentences_batch, temperature=temperature))
        return result

    def _translate(self, texts, model, tokenizer, temperature=1.3):
        with torch.no_grad():
            template = lambda text: f"{text}"
            src_texts = [template(text) for text in texts]

            encoded = tokenizer.prepare_seq2seq_batch(src_texts,
                                                      truncation=True,
                                                      max_length=512, return_tensors="pt").to(self.device)
            encoded['temperature'] = temperature
            encoded['do_sample'] = True

            translated = model.generate(**encoded)

            translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

            return translated_texts

    def _back_translate(self, texts, temperature=1.3):
        # Translate from source to target language
        fr_texts = self._translate(texts, self.tar_model, self.tar_tokenizer,
                             temperature=temperature)

        # Translate from target language back to source language
        back_translated_texts = self._translate(fr_texts, self.src_model, self.src_tokenizer,
                                          temperature=temperature)
        return back_translated_texts
