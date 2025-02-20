import numpy as np
import pandas as pd
import random
import re
import pickle
from rdkit import Chem
import sys
import time
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HELM-Triple",
                                          ignore_mismatched_sizes=True)


class Experience(object):
    def __init__(self, voc, max_size=18):
        self.memory = []
        self.max_size = max_size
        self.voc = voc

    def add_experience(self, experience):
        self.memory.extend(experience)
        if len(self.memory) > self.max_size:
            idxs, helm = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in helm:
                    idxs.append(i)
                    helm.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]

            self.memory.sort(key=lambda x: x[1], reverse=True)
            self.memory = self.memory[:self.max_size]
            print("\nBest score in memory: {:.2f}".format(self.memory[0][1]))

    def sample(self, n):
        if len(self.memory) < n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[1] for x in self.memory]
            sample = np.random.choice(len(self), size=n, replace=False, p=scores / np.sum(scores))
            sample = [self.memory[i] for i in sample]
            helm = [x[0] for x in sample]
            scores = [x[1] for x in sample]
            prior_likelihood = [x[2] for x in sample]

            HELM = [tokenizer.decode(g,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True).replace(" ", "") for g in helm]

        return HELM, np.array(scores), np.array(prior_likelihood)


def unique(arr):
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))


def seq_to_helm(seqs):
    helm = [tokenizer.decode(g,
                             skip_special_tokens=True,
                             clean_up_tokenization_spaces=True) for g in seqs]
    helms = []
    for i in helm:
        helms.append([i])
    return helms































