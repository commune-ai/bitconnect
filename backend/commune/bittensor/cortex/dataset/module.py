""" Manages a pool of grpc connections as receptors
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

import gc
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
import time
from copy import deepcopy
import math
from typing import Tuple, List, Union
from threading import Lock

import numpy as np
import torch
from loguru import logger
import concurrent
import bittensor
from commune.utils import round_sig
import bittensor.utils.networking as net
from concurrent.futures import ThreadPoolExecutor
from commune import BaseModule
from commune.bittensor.receptor.receptor.module import ReceptorModule
import ray
import psutil

import asyncio
logger = logger.opt(colors=True)


class GeneratorObject(object):
    def __init__(self, generator=None, sample_count=0, jobs=[]):
        self.generator = generator
        self.sample_count = sample_count
        self.jobs = jobs


    def sample(self):
        self.sample_count -= 1
        return next(self.generator)
    
    def __reduce__(self):
        deserializer = GeneratorObject
        serialized_data = (None, self.sample_count, self.jobs)
        return deserializer, serialized_data

class DatasetModule (BaseModule, torch.nn.Module ):
    """ Manages a pool of grpc connections as receptors
    """
    default_config_path = 'bittensor.cortex.dataset'

    def __init__(
        self, 
        config = None,
        override= {},
        **kwargs
    ):
        torch.nn.Module.__init__(self)
        BaseModule.__init__(self, config=config, override=override, **kwargs)
        self.load()
        


    def load(self):
        self.load_bitmodule()
        self.load_receptor_pool()
        self.load_dataset()
        self.load_queue()

    def refresh_module(self, module, *args, **kwargs):
        return getattr(self, f'load_{module}')(refresh=True, *args, **kwargs)

    @property
    def available_synapses(self):
        return self.bitmodule.getattr('available_synapses')

    def load_dataset(self, refresh=False):
        dataset_class =  BaseModule.get_object('huggingface.dataset.module.DatasetModule')
        self.dataset = dataset_class.deploy(actor={'refresh': refresh, 'name':'huggingface.dataset'}, load=['tokenizer', 'dataset'], wrap = True)

    def load_receptor_pool(self, refresh=False):
        receptor_pool_class =  BaseModule.get_object('bittensor.receptor.pool.module.ReceptorPoolModule')
        self.receptor_pool = receptor_pool_class.deploy(actor={'refresh': refresh},wallet=self.bitmodule.getattr('wallet'), wrap=True)
        return self.receptor_pool

    def load_queue(self, refresh=False):
        queue_config = self.config.get('queue')
        queue_class =  BaseModule.get_object(queue_config['module'])
        queue_config['actor']['refresh'] = refresh
        self.queue = queue_class.deploy(actor=queue_config['actor'], wrap=True)
        return self.queue

    def load_bitmodule(self, refresh=False, network=None, wallet = None):
        module_class = BaseModule.get_object('bittensor.base.module.BitModule')
        network = network if network != None else self.config['network']
        wallet = wallet if wallet != None else self.config['wallet']
        self.bitmodule = module_class.deploy(actor={'refresh': refresh}, override={'network': network, 'wallet': wallet}, load=True, wrap = True)
        self.sync()
        return self.bitmodule

    def monitor_module_memory(self, module='receptor_pool'):
        threshold_mode = self.config[module]['memory_threshold']['mode'] # ratio, percent, etc
        threshold_value = self.config[module]['memory_threshold']['value']
        receptor_pool_memory_value = self.receptor_pool.memory_usage(threshold_mode)
        if receptor_pool_memory_value > threshold_value:
            getattr(self, f'load_{module}')(refresh=True)
            st.write('refreshed', receptor_pool_memory_value, threshold_value)
            time.sleep(1)
            st.write(self.receptor_pool.getattr('name'))



    def sync(self):
        self.bitmodule.sync()
        self.metagraph_state = self.bitmodule.getattr('metagraph_state')

    # def __str__(self):
    #     return "ReceptorPool({},{})".format(len(self.receptors), self.max_active_receptors)

    # def __repr__(self):
    #     return self.__str__()
    
    def __exit__(self):
        for receptor in self.receptors:
            receptor.__del__()

    # Here is a useful solution that works for various operating systems, including Linux, Windows, etc.:

    ############################
    ##### Forward Function #####
    ############################
    # Forward function queries (n_queried) random endpoints with the inputs
    # then waits timeout before checking for success from each query.
    # The function returns a list of booleans True or false depending on the query result.

    def str2synapse(self, synapse_str, *args, **kwargs):
        synapse =getattr(bittensor.synapse, synapse_str)
        # st.write(synapse)
        if synapse_str in ['TextCausalLMNext','TextCausalLM']:
            synapse =synapse(decode_topk=False, *args, **kwargs)
        else:
            synapse = synapse(*args, **kwargs)
        
        return synapse


    sample_generator_count = 0
    sample_generator_count_max = 5

    def sample_generator_loop(self, loops=10,  splits=['train', 'test'], **kwargs):
        for i in range(loops):
            for split in splits:
                st.write(i, split )
                st.write('memory', self.receptor_pool.memory_usage('ratio'), self.queue.memory_usage('ratio'))
                kwargs['split'] = kwargs['queue'] = split
                self.sample_generator(**kwargs)
                st.write(self.queue.size_map())
                gc.collect()
        
    generator_map = {}
    def start_generator(self, split='train', refresh=False, **kwargs):
        if split in self.generator_map and refresh == False:
            return 
        self.generator_map[split] =GeneratorObject(generator=None)
        self.generator_map[split].generator = self.sample_generator(split=split, **kwargs)   
    def kill_generator(self, split='train'):
        self.generator_map.pop(split,None)
        ray.get(jobs)

    rm_generator = kill_generator
    def ls_generators(self):
        return list(self.generator_map.keys())
    def generate_sample(self, split='train'):
        return self.generator_map[split].sample()

    
    sample_store = {}

    def sample_generator(self, 
                    num_samples=10,
                    max_concurrent_calls = 5,
                    batch_size=1,
                    batch_multiplier=1,
                    min_success=100,
                    endpoint_ids=None , 
                    num_endpoints=20,
                    random_sample=True,
                    timeout=1,
                    seq_len=16,
                    seq_multiplier=1,
                    padding_fill_value = 1,
                    synapse='TextCausalLM',
                    include_metagraph = True, 
                    metagraph_features=['stake', 'ranks', 'trust', 'consensus', 'incentive', 'emission', 'dividends'],
                    success_only=True,
                    split=None,
                    queue=None):

        self.monitor_module_memory()
        self.sample_generator_count += 1
        
        running_jobs_dict = {}
        sample_kwargs = dict(batch_size=batch_size,
                batch_multiplier=batch_multiplier,
                min_success=min_success,
                endpoint_ids=endpoint_ids , 
                num_endpoints=num_endpoints,
                random_sample=random_sample,
                timeout=timeout,
                seq_len=seq_len,
                seq_multiplier=seq_multiplier,
                padding_fill_value =padding_fill_value,
                synapse=synapse,
                split=split,
                include_metagraph=include_metagraph,
                metagraph_features=metagraph_features,
                success_only=success_only)

        max_concurrent_calls = min(max_concurrent_calls, num_samples)
        if max_concurrent_calls == -1:
            max_concurrent_calls = num_samples
        running_jobs = []

        job2results = {}
        
        sample_list = []
        for i in range(num_samples):
            results_dict = self.sample(**sample_kwargs, ray_get=False)
            job = results_dict.pop('job')
            job2results[job] = results_dict
            running_jobs = list(job2results.keys())

            if split in self.generator_map:
                self.generator_map[split].sample_count += (batch_multiplier*seq_multiplier)
                self.generator_map[split].jobs = running_jobs

            st.write(i)
            while len(running_jobs) >= max_concurrent_calls or (i == num_samples-1 and len(running_jobs) > 0):
                finished_jobs, running_jobs = ray.wait(running_jobs)

                if len(finished_jobs) > 0:
                    
                    for job in finished_jobs:
                        results = ray.get(job)
                        results_dict = job2results.pop(job)
                        del job

                        results = self.process_results(results=results, synapse=synapse)
                        results_dict.update(results)

                        if len(results_dict['tensor']) == 0:
                            continue
                        
                        if include_metagraph:
                            if metagraph_features != None:
                                metagraph_state = self.metagraph_state
                                for k in metagraph_features:
                                    results_dict[k] = metagraph_state[k][results_dict['endpoint']]

                        batch_chunked_tensors = torch.chunk(results['tensor'], chunks=batch_multiplier, dim=1)
                        for batch_chunked_tensor in batch_chunked_tensors:

                            chunked_tensors = torch.chunk(batch_chunked_tensor, chunks=seq_multiplier, dim=2)
                            for chunked_tensor in chunked_tensors:
                                
                                sample_result =  {**results_dict, **{'tensor': chunked_tensor}},
                                # yield sample_result
                                sample_list.append(sample_result)
       
        return sample_list
       
       
       
        # self.kill_generator(split)
        # if queue != None:

        #     queue_kwargs = {}
        #     if isinstance(queue, str):
        #         queue_kwargs['topic'] = queue

        #     elif isinstance(queue, dict):
        #         queue_kwargs = queue
        #     for finished_result in finished_results:

        #         queue_kwargs['item'] = finished_result
        #         job = self.queue.put(**queue_kwargs)


        # return finished_results
        # self.sample_generator_count -= 1 
        # return finished_results
        


    def sample_queue(self, 
                    num_samples=2,
                    batch_size=1,
                    batch_multiplier=1,
                    min_success=100,
                    endpoint_ids=None , 
                    num_endpoints=20,
                    random_sample=True,
                    timeout=1,
                    seq_len=16,
                    seq_multiplier=1,
                    padding_fill_value = 1,
                    synapse='TextCausalLM',
                    success_only=True,
                    split='train'):

        self.monitor_module_memory()

        # while self.sample_generator_count >= self.sample_generator_count_max:
        #     time.sleep(1)
        
        self.sample_generator_count += 1


        
        running_jobs_dict = {}
        sample_kwargs = dict(batch_size=batch_size,
                batch_multiplier=batch_multiplier,
                min_success=min_success,
                endpoint_ids=endpoint_ids , 
                num_endpoints=num_endpoints,
                random_sample=random_sample,
                timeout=timeout,
                seq_len=seq_len,
                seq_multiplier=seq_multiplier,
                padding_fill_value =padding_fill_value,
                synapse=synapse,
                split=split,
                success_only=success_only,
                ray_get=False)

            
        for i in range(num_samples):

            job = self.sample(**sample_kwargs)
            running_jobs_dict[job] = True
        running_jobs = list(running_jobs_dict.keys())

        self.generator_jobs[split] = running_jobs
        finished_results = []
        while len(running_jobs)>0:
            finished_jobs, running_jobs = ray.wait(running_jobs)
            self.generator_jobs[split] = running_jobs
            if len(finished_jobs) > 0:
                for job in finished_jobs:

                    results = ray.get(job)
                    results = self.process_results(results=results, synapse=synapse)
                    batch_chunked_tensors = torch.chunk(results['tensor'], chunks=batch_multiplier, dim=1)
                    for batch_chunked_tensor in batch_chunked_tensors:

                        chunked_tensors = torch.chunk(batch_chunked_tensor, chunks=seq_multiplier, dim=2)
                        for chunked_tensor in chunked_tensors:
                            sample_result = {**results, **{'tensor': chunked_tensor}}
                            self.queue.put(topic=split, item=sample_result)
                            
                            # finished_results.append(sample_result)
                            # if generator:
                            # if not split in self.sample_store:
                            #     self.sample_store[split] = []
                            # self.sample_store[split].append(sample_result)
                            # yield sample_result
        
        # self.kill_generator(split)
        # if queue != None:

        #     queue_kwargs = {}
        #     if isinstance(queue, str):
        #         queue_kwargs['topic'] = queue

        #     elif isinstance(queue, dict):
        #         queue_kwargs = queue
        #     for finished_result in finished_results:

        #         queue_kwargs['item'] = finished_result
        #         job = self.queue.put(**queue_kwargs)


        # return finished_results
        # self.sample_generator_count -= 1 
        # return finished_results
        

    def process_results(self, results, synapse):
        results_dict = {'tensor':[], 'code':[], 'latency':[], 'endpoint': []}

        num_responses = len(results[0])
        for i in range(num_responses):
            tensor = results[0][i][0]
            code = results[1][i][0]
            latency = results[2][i][0]
            endpoint = results[3][i]

            if str(synapse) in ['TextCausalLMNext']:
                if tensor.shape[-1] != 2:
                    continue

            results_dict['tensor'].append(tensor)
            results_dict['code'].append(code)
            results_dict['latency'].append(latency)
            results_dict['endpoint'].append(endpoint)

        if len(results_dict['tensor'])>0:
            results_dict['tensor'] = torch.stack(results_dict['tensor'])
            results_dict['code'] = torch.tensor(results_dict['code'])
            results_dict['latency'] = torch.tensor(results_dict['latency'])
            results_dict['endpoint'] = torch.tensor(results_dict['endpoint'])
        else:
            results_dict['tensor'] = torch.tensor([])
            results_dict['code'] = torch.tensor([])
            results_dict['latency'] = torch.tensor([])
            results_dict['endpoint'] =  torch.tensor([])


        return results_dict


    def splits(self):
        return self.dataset.getattr('splits')

    def sample(self,
                 batch_size=1,
                 batch_multiplier=1,
                 min_success=100,
                 endpoint_ids=None , 
                 num_endpoints=30,
                 random_sample=True,
                 timeout=4,
                 seq_len=16,
                 seq_multiplier=1,
                 padding_fill_value = 1,
                 synapse='TextCausalLM',
                 include_metagraph = True, 
                 metagraph_features=['stake', 'ranks', 'trust', 'consensus', 'incentive', 'emission', 'dividends'],
                 success_only=True,
                 split=None,
                 ray_get=True):


        assert synapse in self.available_synapses, f'{synapse} should be in available synapses {self.available_synapses}'
        synapse = self.str2synapse(synapse)
        macro_batch_size = batch_size*batch_multiplier
        
        input_tokens = self.dataset.sample(batch_size=macro_batch_size, split=split)

        
        endpoints = self.bitmodule.get_endpoints(endpoint_ids=endpoint_ids , 
                                                num_endpoints=num_endpoints, 
                                                random_sample=random_sample)

        # endpoint_ids = [e.uid for e in endpoints]

        # metagraph_dict = {k:self.metagraph_state[k][endpoint_ids]  for k in ['stake', 'incentive', 'consensus', 'trust','ranks', 'dividends', 'emmision']}


        # ensure sequence is sequence length
        input_tokens_seq_len = input_tokens.shape[1]
        if seq_len > input_tokens_seq_len:
            input_tokens_seq_len = input_tokens.shape[1]
            seq_filler = torch.full(size=(input_tokens.shape[0], seq_len - input_tokens_seq_len), fill_value=padding_fill_value)
            input_tokens = torch.cat([input_tokens, seq_filler], dim=1)
        else:
            input_tokens = input_tokens[:,:seq_len]



        forward_kwargs = dict(
                              inputs= [input_tokens] ,
                              synapses=[synapse],
                              timeout=timeout, 
                              endpoints=endpoints,
                              min_success=min_success,
                              return_success_only=success_only
                              )

        results_dict = {'input': input_tokens}
        result = self.receptor_pool.forward(**forward_kwargs, ray_get=ray_get)
        if ray_get:
            # result is a response from the receptor_pool
            output_dict = self.process_results(results=result, synapse=synapse)
            results_dict.update(output_dict)

            if include_metagraph and len(output_dict['tensor']) > 0:
                if metagraph_features != None:
                    metagraph_state = self.metagraph_state
                    for k in metagraph_features:
                        results_dict[k] = metagraph_state[k][results_dict['endpoint']]
        else:
            # result is a ray job
            results_dict['job'] = result

        return results_dict

if __name__ == '__main__':
    
    import streamlit as st
    import time


    module = DatasetModule.deploy(actor=False, wrap=True)
    # st.write(module.list_actors(detail=True))
    # st.write(module.refresh_module('receptor_pool'))
    # st.write(module.list_actors())


    # module.sample_generator( num_samples=10, max_concurrent_calls=5, batch_multiplier=1, batch_size=5, seq_len=10, seq_multiplier=1, timeout=4, num_endpoints=100, split='train')
    
    
    # st.write(module.generate_sample('train'))
    
    sample_dict = module.sample(num_endpoints=100)
    st.write({k:v.shape for k,v in sample_dict.items()})

    # st.write(module.getattr('generator_map')['train'].__dict__)


