import streamlit as st
from random import shuffle, seed
from collections import defaultdict
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
# import torchsort
from commune import Module
import ray
import asyncio

# from commune.bittensor.cortex.metric import causal_lm_loss, ranknet_loss
from commune.utils import *
from sklearn import metrics
from scipy.stats import kendalltau
import torch
from torch import nn
from commune.ray.actor_pool import ActorPool
Module.new_event_loop()
import bittensor

class DatasetModule(Module):
    def __init__(self,config=None, tokenizer=None, dataset=None, **kwargs):
        Module.__init__(self, config=config, **kwargs)

        self.load_tokenizer(tokenizer)
        self.load_dataset(dataset)


    def load_tokenizer(self, tokenizer=None): 
        tokenizer = tokenizer if tokenizer else self.config['tokenizer']
        
        st.write(tokenizer)
        self.tokenizer = self.launch_module(**tokenizer)
        return self.tokenizer

    def load_dataset(self, dataset=None):
        dataset = dataset if dataset else self.config['dataset']
        self.dataset = self.launch_module(**dataset)
        return self.dataset

    def filter_dataset(self, fn, dataset=None):
        if dataset == None:
            dataset = self.dataset
        for split in dataset.keys():
            dataset[split] = dataset[split].filter(fn)
        return dataset

    @property
    def info(self):
        return self.dataset[self.splits[0]]._info.__dict__

    @property
    def features(self):
        return self.dataset[self.splits[0]]._info.__dict__['features']

    @property
    def device(self):
        device = self.config.get('device', 'cpu')
        if 'cuda' in device:
            assert torch.cuda.is_available()
        return device

    @device.setter
    def device(self, device):
        if 'cuda' in device:
            assert torch.cuda.is_available()
        self.config['device'] = device
        return device

    def to(self, device):
        self.device = device
        return self.device

    default_receptor_path = 'bittensor.receptor.pool.module.ReceptorPoolModule'

    def tokenize(self, text, padding=True, *args, **kwargs):
        device = kwargs.pop('device', self.device)
        return torch.tensor(self.tokenizer(text=text, padding=padding)['input_ids']).to(device)

    @property
    def splits(self):
        return list(self.dataset.keys())


    def split_size(self, split=None):
        if split == None:
            split = self.splits[0]
        return len(self.dataset[split])

    def split_size_map(self):
        return {split: self.dataset[split] for split in self.splits}

    def __getitem__(self, idx=None, split='train', sequence_length=128):
        
        text_field = self.config['dataset']['text_field']
        dataset_length =  self.split_size(split)
        if idx == None:
            idx = random.randint(1,dataset_length-1)    

        final_sample  = ''
        while len(final_sample.split()) < sequence_length:
            if split == None:
                split = self.splits[0]
                
            sample = self.dataset[split][idx][text_field]
            final_sample += sample if len(final_sample) == 0 else '\n' + sample
            idx = (idx + 1 ) % dataset_length
        
        final_sample = ' '.join(final_sample.split()[:sequence_length])

        return final_sample

    def sample(self, batch_size=10, sequence_length=16, random=True, idx_list = None, tokenize=True, padding=True,  **kwargs):
        if idx_list != None:
            assert isinstance(idx_list, list)
            batch_size = len(idx_list)
            samples =  [self.__getitem__(idx=idx_list[i] ,**kwargs) for i in range(batch_size)]

        elif idx_list == None:
            samples =  [self.__getitem__(idx=None if random else i,**kwargs) for i in range(batch_size)]
        else:
            raise NotImplementedError(type(idx_list))

        if tokenize:
            samples = self.tokenize(samples, padding=padding)
            samples = samples[:,:sequence_length]
            remainder = sequence_length - samples.shape[1]
            if remainder > 0:
                filler = torch.full(size=(samples.shape[0], remainder),fill_value=0).to(samples.device)
                samples = torch.cat([samples, filler], dim=1)
        return samples
    
    def resolve_device(self, device=None):
        if device == None:
            device = self.device
        return device

    @staticmethod
    def ray_job_generator(running_jobs):
        while running_jobs:
            finished_jobs, running_jobs = ray.wait(running_jobs)
            for finished_job in finished_jobs:
                yield ray.get(finished_job)

    classmethod
    def test_model_sample(cls):
        self = cls()
        batch_size=12
        seqeunce_length = 256
        x = self.sample(batch_size=batch_size,seqeunce_length=seqeunce_length )
        assert x.shape[0] == batch_size
        assert x.shape[1] == seqeunce_length

if __name__ == '__main__':
    module = DatasetModule.deploy(actor={'refresh': False}, load=True, wrap=True)
    generator = DatasetModule.ray_job_generator([module.sample(batch_size=32, sequence_length=256, ray_get=False) for i in range(100)])
    for x in generator:
        st.write(x.shape)

