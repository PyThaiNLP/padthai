# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset


class ListDataset(Dataset):
    """
    List dataset wrapper.

    Original code from https://link.medium.com/4FfbALWz8gb
    """
    def __init__(self, txt_list: list, tokenizer, max_length: int):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer(
                txt, truncation=True,
                max_length=max_length, padding="max_length"
            )
            self.input_ids.append(torch.tensor(list(encodings_dict['input_ids'])))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]