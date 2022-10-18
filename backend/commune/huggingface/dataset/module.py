import streamlit as st
from random import shuffle, seed
from collections import defaultdict
import pandas as pd
import bittensor
import torch
from torch import nn
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
# import torchsort
from commune.bittensor import BitModule
from commune import BaseModule
import ray
from commune.bittensor.cortex.metric import causal_lm_loss, ranknet_loss
from commune.utils import *
from sklearn import metrics
from scipy.stats import kendalltau


import torch
from torch import nn

from commune.ray.actor_pool import ActorPool


class DatasetModule(BaseModule):
    __file__ = __file__
    default_config_path = 'huggingface.dataset'
    def __init__(self, config=None, load=True, **kwargs):
        BitModule.__init__(self, config=config,  **kwargs)

        if type(load) in [dict]:
            self.load(**load)
        else:
            self.load(load)

    default_load_keys = [ 'dataset', 'tokenizer']
    def load(self, keys=True, load_kwargs={}, load_args={}, **kwargs):
        
        if keys in [False, None]:
            return

        if keys == True:
            keys = self.default_load_keys
        
        load_keys = keys

        for load_key in load_keys:
            st.write(load_key, 'load')
            load_kwargs.get(load_key, {}) 
            load_fn = getattr(self, f'load_{load_key}', None)
            assert load_fn != None, f'{load_key} is suppose to be a function'
            load_fn_kwargs = load_kwargs.get(load_key, self.config.get(load_key, {}))
            load_fn(**load_fn_kwargs)

    def load_dataset(self, path, params, **kwargs):
        dataset_class = self.import_object(path)

        split = params.get('split')
        if isinstance(split, str):
            params['split'] = {split: split}
        elif isinstance(split, list):
            params['split'] = {s:s for s in params['split']}  
        

        self.dataset = dataset_class(**params)

    @property
    def info(self):
        return self.dataset[self.splits[0]]._info.__dict__

    @property
    def features(self):
        return self.dataset[self.splits[0]]._info.__dict__['features']


    def load_tokenizer(self, 
                        path='bittensor.tokenizer', 
                        params=dict(version=bittensor.__version__), **kwargs): 
        tokenizer_class = self.import_object(path)
        self.tokenizer = tokenizer_class(**params)

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


    def __getitem__(self, idx=None, tokenize=False, split=None):
        if split == None:
            split = self.splits[0]
        
        text_field = self.config['dataset']['text_field']
        dataset_length = len(self.dataset[split])
        if idx == None:
            idx = random.randint(1,dataset_length-1)

        assert idx <= dataset_length, f'{idx}<={dataset_length} '
    
        sample = self.dataset[split][idx][text_field]
        if tokenize == True:
            sample = self.tokenize(text=sample, padding=True)
        return sample

    def sample(self, batch_size=1, random=True, **kwargs):
        return self.tokenize([self.__getitem__(idx=None if random else i, tokenize=False,**kwargs) for i in range(batch_size)])

    def resolve_device(self, device=None):
        if device == None:
            device = self.device
        
        return device




if __name__ == '__main__':


    import ray

    # st.write(BenchmarkModule.metagraph)
    # module = BenchmarkModule.deploy(actor={'refresh': False, 'name': f'benchmark'})
    # module = DatasetModule.deploy(actor={'refresh': False}, load=True, wrap=True)
    # DatasetModule.ray_restart()
    module = DatasetModule.deploy(actor=False, load=True, wrap=False)
    # ray.get(module.put_batch.remote('fam', [1]*100 ))
    # st.write(ray.get(module.get_batch.remote('fam', 10)))
    # ray.get(module.load_receptor_pool.remote(actor=False))
    # # st.write(module.actor)
    # # topic='train'

    # ray.get(module.delete.remote('receptor_pool'))
    
    st.write(module.dataset)
    for i in range(10):
        resp = module.sample(batch_size=100, split='train')
        
        # del module.receptor_pool
        st.write(resp.shape, i)
    
    # finished_results = []
    # while running_jobs:
    #     finished_jobs, running_jobs = ray.wait(running_jobs)
    #     if len(finished_jobs)>0:
    #         for job in  finished_jobs:
    #             finished_results += [ray.get(job)]
    #             st.write(len(finished_results))

    # st.write(ray.get(module.sample_generator.remote('train')))

    # st.write(ray.get(queue_server.delete_all.remote()))

    # all_synapses = ray.get(module.getattr.remote('available_synapses'))

    # selected_synapses = st.multiselect('Select Synapses',all_synapses,  all_synapses[:1])
    # module.start_sample_loop.remote(topic='train', synapse=selected_synapses, timeout=1.0, success_only=True, refresh_cache=True, refresh_queue=True)
    # st.write(ray.get(module.sample_cache_count.remote('train')))

    # all_synapses = ray.get(module.getattr.remote('available_synapses'))
    # selected_synapses = st.multiselect('Select Synapses',all_synapses,  all_synapses[:1])
    # ray.get(module.start_sample_loop.remote(topic='test', synapse=selected_synapses, timeout=1, success_only=True, refresh_cache=True, refresh_queue=True))
    # st.write(ray.get(module.sample_cache_count.remote('train')))
    # # st.write(ray.get(module.stop_sample_loop.remote('train')))
    # # st.write(ray.get(module.running_loops.remote()))
    # st.write(ray.get(module.sample_loop.remote('train')))


    # st.write(ray.get(module.sample_loop.remote('test')))
    # st.write()
    
    # st.write(random.uniform(0,1))
    # random.randint()

    # with st.expander('tensors', False):
    #     st.write(results[0])
    # with st.expander('return type', False):
    #     st.write(results[1])

    
