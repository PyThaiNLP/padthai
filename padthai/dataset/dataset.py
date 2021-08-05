# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class ListDataset(Dataset):
    """
    List dataset wrapper.

    Original code from https://link.medium.com/4FfbALWz8gb
    """
    def __init__(
        self, txt_list: list, tokenizer: GPT2Tokenizer,
        max_length: int,
        bos_token: str = None,
        eos_token: str = None
    ):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            if bos_token is None and eos_token is None:
                encodings_dict = tokenizer(
                    txt, truncation=True,
                    max_length=max_length,
                    padding="max_length"
                )
            elif bos_token is not None and eos_token is not None:
                # if both bos_token and eos_token specified
                encodings_dict = tokenizer(
                    bos_token + txt + eos_token,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length"
                )
            else:
                raise Exception("bos_token and eos_token should be both specified")
            self.input_ids.append(torch.tensor(list(encodings_dict['input_ids'])))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]