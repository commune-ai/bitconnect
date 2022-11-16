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
from huggingface_hub import HfApi
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
    def __init__(self,config=None, tokenizer=None, dataset=None, load=False,**kwargs):
        Module.__init__(self, config=config, **kwargs)
        self.hf_api = HfApi(self.config.get('hub'))

        if load:
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

    def sample(self, batch_size=10, sequence_length=16, random=True, idx_list = None, tokenize=True, padding=True,  **kwargs)->dict:
        if idx_list != None:
            assert isinstance(idx_list, list)
            batch_size = len(idx_list)
            samples =  [self.__getitem__(idx=idx_list[i] ,**kwargs) for i in range(batch_size)]

        elif idx_list == None:
            samples =  [self.__getitem__(idx=None if random else i,**kwargs) for i in range(batch_size)]
        else:
            raise NotImplementedError(type(idx_list))
        output = {'text': samples}

        if tokenize:
            sample_tokens = self.tokenize(samples, padding=padding)
            sample_tokens = samples[:,:sequence_length]
            remainder = sequence_length - samples.shape[1]
            if remainder > 0:
                filler = torch.full(size=(samples.shape[0], remainder),fill_value=0).to(samples.device)
                samples = torch.cat([samples, filler], dim=1)
            output['tokens'] = sample_tokens


        return output
    
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


    def list_datasets(self,return_type = 'dict', filter_fn=None, *args, **kwargs):
        datasets = self.hf_api.list_datasets(*args,**kwargs)
        filter_fn = self.resolve_filter_fn(filter_fn=filter_fn)
        if return_type in 'dict':
            datasets = list(map(lambda x: x.__dict__, datasets))
            if filter_fn != None and callable(filter_fn):
                datasets = list(filter(filter_fn, datasets))
        elif return_type in ['pandas', 'pd']:
            datasets = list(map(lambda x: x.__dict__, datasets))
            df = pd.DataFrame(datasets)
            df['num_tags'] = df['tags'].apply(len)
            df['tags'] = df['tags'].apply(lambda tags: {tag.split(':')[0]:tag.split(':')[1] for tag in tags  }).tolist()
            for tag_field in ['task_categories']:
                df[tag_field] = df['tags'].apply(lambda tag:tag.get(tag_field) )
            df['size_categories'] = df['tags'].apply(lambda t: t.get('size_categories'))
            df = df.sort_values('downloads', ascending=False)
            if filter_fn != None and callable(filter_fn):
                df = self.filter_df(df=df, fn=filter_fn)
            return df
        else:
            raise NotImplementedError

    
        return datasets

    @property
    def task_categories(self):
        return list(self.datasets['task_categories'].unique())
    @property
    def pipeline_tags(self): 
        df = self.list_models(return_type='pandas')
        return df['pipeline_tag'].unique()
    @property
    def pipeline_tags_count(self):
        count_dict = dict(self.models_df['pipeline_tag'].value_counts())
        return {k:int(v) for k,v in count_dict.items()}

    @staticmethod
    def resolve_filter_fn(filter_fn):
        if filter_fn != None:
            if callable(filter_fn):
                fn = filter_fn

            if isinstance(filter_fn, str):
                filter_fn = eval(f'lambda r : {filter_fn}')
        
            assert(callable(filter_fn))
        return filter_fn
    @property
    def models(self):
        df = pd.DataFrame(self.list_models(return_type='dict'))
        return df
    @property
    def datasets(self):
        df = pd.DataFrame(self.list_datasets(return_type='dict'))
        return df



    def list_models(self,return_type = 'pandas',filter_fn=None, *args, **kwargs):
        models = self.hf_api.list_models(*args,**kwargs)
       
        filter_fn = self.resolve_filter_fn(filter_fn=filter_fn)


        if return_type in 'dict':
            models = list(map(lambda x: x.__dict__, models))
            if filter_fn != None and callable(filter_fn):
                models = list(filter(filter_fn, models))

        elif return_type in ['pandas', 'pd']:

            models = list(map(lambda x: x.__dict__, models))
            models = pd.DataFrame(models)
            if filter_fn != None and callable(filter_fn):
                models = self.filter_df(df=models, fn=filter_fn)

        else:
            raise NotImplementedError

        return models


    @property
    def task_categories(self):
        return list(self.datasets['task_categories'].unique())
    @property
    def pipeline_tags(self): 
        df = self.list_models(return_type='pandas')
        return df['pipeline_tag'].unique()



    def dataset_tags(self, limit=10, **kwargs):
        df = self.list_datasets(limit=limit,return_type='pandas', **kwargs)
        tag_dict_list = df['tags'].apply(lambda tags: {tag.split(':')[0]:tag.split(':')[1] for tag in tags  }).tolist()
        tags_df =  pd.DataFrame(tag_dict_list)
        df = df.drop(columns=['tags'])
        return pd.concat([df, tags_df], axis=1)

    @staticmethod
    def filter_df(df, fn):
        indices =  df.apply(fn, axis=1)
        return df[indices]

if __name__ == '__main__':
    module = DatasetModule.deploy(actor=False, load=False, wrap=True)
    st.write(module.pipeline_tags)
    st.write(module.task_categories)
    # for x in :
    #     st.write(x.shape)
