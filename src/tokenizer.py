import re
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

class SimpleTokenizer():
    def __init__(self, vocab=None, **kwargs):
        #super().__init__(**kwargs)
        self.special_tokens = {'<|startoftext|>': 0, '<|endoftext|>': 1, '<|pad|>': 2, '<|unk|>': 3}
        self.vocab = vocab if vocab is not None else self.special_tokens.copy()
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token = '<|pad|>'
        self.pad_token_id = self.special_tokens[self.pad_token]
        self.unk_token = '<|unk|>'
        self.unk_token_id = self.special_tokens[self.unk_token]
        self.tokenizer_initialized = bool(vocab)

    def _tokenize(self, text):
        tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        return tokens

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        return self.inverse_vocab.get(index, self.unk_token)

    def build_vocab(self, texts):
        token_counter = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            token_counter.update(tokens)
        
        self.vocab = self.special_tokens.copy()
        index = len(self.special_tokens)
        for token, _ in token_counter.items():
            if token not in self.vocab:
                self.vocab[token] = index
                index += 1
        self.inverse_vocab = {index: token for token, index in self.vocab.items()}
        self.tokenizer_initialized = True

    def encode(self, text, add_special_tokens=True, **kwargs):
        tokens = self._tokenize(text)
        if add_special_tokens:
            tokens = ['<|startoftext|>'] + tokens + ['<|endoftext|>']
        token_ids = [self._convert_token_to_id(token) for token in tokens]
        return token_ids

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True, **kwargs):
        tokens = [self._convert_id_to_token(token_id) for token_id in token_ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        text = ' '.join(tokens)
        return text

    def pad(self, encoded_inputs, max_length=None, padding=True, pad_to_multiple_of=None, return_tensors=None, **kwargs):
        if max_length is None:
            max_length = max(len(inputs) for inputs in encoded_inputs['input_ids'])

        for key in encoded_inputs:
            padded = []
            for sequence in encoded_inputs[key]:
                if len(sequence) < max_length:
                    sequence += [self.pad_token_id] * (max_length - len(sequence))
                else:
                    sequence = sequence[:max_length]
                padded.append(sequence)
            encoded_inputs[key] = padded

        if return_tensors == 'pt':
            for key in encoded_inputs:
                encoded_inputs[key] = torch.tensor(encoded_inputs[key])
        
        return encoded_inputs

    def __call__(self, text, padding=True, truncation=True, max_length=512, return_tensors='pt', **kwargs):
        if isinstance(text, str):
            text = [text]

        encoded_inputs = {'input_ids': [], 'attention_mask': []}
        for t in text:
            encoded_input = self.encode(t, add_special_tokens=True)
            attention_mask = [1] * len(encoded_input)
            encoded_inputs['input_ids'].append(encoded_input)
            encoded_inputs['attention_mask'].append(attention_mask)

        padded_inputs = self.pad(encoded_inputs, max_length=max_length, padding=padding, return_tensors=return_tensors)
        return padded_inputs

    def get_vocab(self):
        return self.vocab

        