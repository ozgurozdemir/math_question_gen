from transformers import AutoTokenizer

import torch
import json
import os
import random
from itertools import islice, chain, cycle

class MathDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer: str="AnReu/math_albert", maxlen=128, replace_cls=True, seq2seq=False):
        self.data_path = data_path
        self.dataset   = self.__read_files__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.maxlen = maxlen
        self.replace_cls = replace_cls

        self.seq2seq = seq2seq

        self.subj_to_id = {
            "Algebra": 0, "Counting & Probability": 1,
            "Geometry": 2, "Intermediate Algebra": 3,
            "Number Theory": 4, "Prealgebra": 5, "Precalculus": 6
        }

        self.id_to_subj = {self.subj_to_id[k]: k for k in self.subj_to_id}

        self.subj_to_token = {"Algebra": "[ALGEBRA]", "Counting & Probability": "[COUNTPROB]", "Geometry": "[GEO]",
                              "Intermediate Algebra": "[INTERALGEBRA]", "Number Theory": "[NUMTHEORY]",
                              "Prealgebra": "[PREALGEBRA]", "Precalculus": "[PRECALCULUS]"}

        for token in self.subj_to_token.values():
            self.tokenizer.add_tokens(token)


        self.level_to_id = {
            "Level ?": 0, "Level 1": 1,
            "Level 2": 2, "Level 3": 3,
            "Level 4": 4, "Level 5": 5
        }
        self.id_to_level = {self.level_to_id[k]: k for k in self.level_to_id}


    def __read_files__(self):
        dataset = []
        for subject in os.listdir(self.data_path):
            for fname in os.listdir(f"{self.data_path}/{subject}"):
                with open(f"{self.data_path}/{subject}/{fname}", "r") as file:
                    dataset.append(json.load(file))

        return dataset

    def __prepare_dataset_item__(self, idx):
        item = {}

        if self.seq2seq:
            cls = self.tokenizer(self.dataset[idx]["type"], max_length=25, padding='max_length', truncation=True)
            item["input_ids"]       = torch.Tensor(cls["input_ids"]).type(torch.int16)
            item["attention_mask"]  = torch.Tensor(cls["attention_mask"]).type(torch.int16)

            problem = self.tokenizer(self.dataset[idx]["problem"], max_length=self.maxlen, padding='max_length', truncation=True)
            item["labels"]  = torch.Tensor(problem["input_ids"]).type(torch.int16)

            # replace CLS token with the target subject
            if self.replace_cls:
                item["labels"][0] = self.tokenizer.get_vocab()[self.subj_to_token[self.dataset[idx]["type"]]]

        else:
            problem = self.tokenizer(self.dataset[idx]["problem"], max_length=self.maxlen, padding='max_length', truncation=True)
            item["input_ids"]  = torch.Tensor(problem["input_ids"]).type(torch.int16)

            # replace CLS token with the target subject
            if self.replace_cls:
                item["input_ids"][0] = self.tokenizer.get_vocab()[self.subj_to_token[self.dataset[idx]["type"]]]

            item["attention_mask"]  = torch.Tensor(problem["attention_mask"]).type(torch.int16)
            item["labels"]     = self.subj_to_id[self.dataset[idx]["type"]]

        return item


    def __getitem__(self, idx):
        return self.__prepare_dataset_item__(idx)

    def __len__(self):
        return len(self.dataset)

