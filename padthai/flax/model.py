# -*- coding: utf-8 -*-
import os
import torch
from typing import List
from torch.utils.data import random_split
from .dataset import ListDataset
from transformers import (
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    GPT2LMHeadModel
)
torch.manual_seed(42)


class FlaxGPT2FewShot:
    """
    Few-Shot Learning using GPT-2 from Flax Community

    Default model: https://huggingface.co/flax-community/gpt2-base-thai
    Original code from https://link.medium.com/4FfbALWz8gb
    """
    def __init__(
        self,
        model_dir: str,
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu").type,
        pretrained: str = "flax-community/gpt2-base-thai"
    ):
        """
        :param str model_dir: path of model directory
        :param str device: device
        :param str device: device (cuda is default)
        """
        self.device = device
        self.model_dir = model_dir
        self.pretrained = pretrained
        if not os.path.exists(self.model_dir):
            self._init_model()
        else:
            self.load_model()

    def _init_model(self) -> None:
        """
        initialize GPT-2 model
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.pretrained
        )
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.pad_token = self.tokenizer.pad_token
        self.tokenizer.save_pretrained(self.model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(
            self.pretrained
        ).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def load_model(self):
        """
        Load model from path of model directory
        """
        self.model_dir = self.model_dir
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.model_dir
        )
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.pad_token = self.tokenizer.pad_token
        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_dir
        ).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def train(
        self,
        data: List[str],
        logging_dir: str,
        num_train_epochs: int = 10,
        train_size: float = 0.95,
        batch_size: int = 2,
        save_every_epochs: bool = True,
        max_length: int = None
    ):
        """
        Train model

        :param str data: List for text
        :param str logging_dir: logging directory
        :param int num_train_epochs: Number train epochs
        :param str train_size: size of train set
        :param int batch_size: batch size
        :param bool save_every_epochs: save model every epochs
        :param int max_length: maximum token length, default as None
        """
        if save_every_epochs:
            self.evaluation_strategy = "epoch"
        else:
            self.evaluation_strategy = "no"
        self.data = data
        if max_length is None:
            self.max_length = max(
                [len(self.tokenizer.encode(i)) for i in self.data]
            )
        else:
            self.max_length = max_length
        self.dataset = ListDataset(
            self.data,
            self.tokenizer,
            max_length=self.max_length
        )
        self.train_size = int(train_size * len(self.dataset))
        _, self.val_dataset = random_split(
            self.dataset, [
                self.train_size,
                len(self.dataset) - self.train_size
            ]
        )
        self.training_args = TrainingArguments(
            output_dir=self.model_dir,
            do_train=True,
            do_eval=True,
            evaluation_strategy="epoch",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            logging_strategy="epoch",
            save_strategy=self.evaluation_strategy,
            per_device_eval_batch_size=batch_size,
            logging_dir=logging_dir
        )
        self.train = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            eval_dataset=self.val_dataset,
            data_collator=lambda data: {
                'input_ids': torch.stack([f[0] for f in data]),
                'attention_mask': torch.stack([f[1] for f in data]),
                'labels': torch.stack([f[0] for f in data])
            }
        )
        self.train.train()
        self.train.evaluate()
        self.train.save_model(self.model_dir)

    def remove_bos(self, txt: str) -> str:
        return txt.replace(self.bos_token, '')

    def remove_eos(self, txt: str) -> str:
        return txt.replace(self.eos_token, '')

    def remove_bos_eos(self, txt: str) -> str:
        return self.remove_eos(self.remove_bos(txt))

    def gen(
        self,
        text: str,
        top_k: int = 50,
        max_length: int = 89,
        top_p: float = 0.95,
        keep_bos: bool = False,
        keep_eos: bool = False,
        temperature: int = 1,
        num_return_sequences: int = 5,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        :param str text: text
        :param int top_k: top k
        :param int max_length: max length of return sequences
        :param float top_p: top p
        :param bool keep_bos: keep beginning of a sentence
        :param bool keep_eos: keep end of a sentence
        :param int temperature: temperature
        :param int num_return_sequences: number of return sequences
        :param bool skip_special_tokens: skip special tokens
        :return: return sequences
        :rtype: List[str]
        """
        self.generated = self.tokenizer(
            text, return_tensors="pt"
        ).input_ids.to(self.device)
        self.sample_outputs = self.model.generate(
            self.generated,
            do_sample=True,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=num_return_sequences
        )
        _temp = [
            self.tokenizer.decode(
                i, skip_special_tokens=skip_special_tokens
            ) for i in self.sample_outputs
        ]
        if not keep_bos and not keep_eos:
            return [self.remove_bos_eos(i) for i in _temp]
        elif not keep_bos:
            return [self.remove_bos(i) for i in _temp]
        elif not keep_eos:
            return [self.remove_eos(i) for i in _temp]
        else:
            return _temp
