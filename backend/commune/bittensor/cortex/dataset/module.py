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


    def monitor_module_memory(self, module='receptor_pool'):
        threshold_mode = self.config[module]['memory_threshold']['mode'] # ratio, percent, etc
        threshold_value = self.config[module]['memory_threshold']['value']
        receptor_pool_memory_value = self.receptor_pool.memory_usage(threshold_mode)
        if receptor_pool_memory_value > threshold_value:
            getattr(self, f'load_{module}')(refresh=True)
            st.write('refreshed', receptor_pool_memory_value, threshold_value)
            time.sleep(1)
            st.write(self.receptor_pool.getattr('name'))

        

    def load_bitmodule(self, refresh=False):
        module_class = BaseModule.get_object('bittensor.base.module.BitModule')
        self.bitmodule = module_class.deploy(actor={'refresh': refresh}, override={'network': self.config['network'], 'wallet': self.config['wallet']},load=True, wrap = True)
        self.sync()
        return self.bitmodule

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

    def get_total_requests(self):
        return self.total_requests
    def get_receptors_state(self):
        r""" Return the state of each receptor.
            Returns:
                states (:obj:`List[grpc.channel.state]`)
                    The state of receptor.
        """
        return {hotkey: v.state() for hotkey, v in self.receptors.items()}



    # Here is a useful solution that works for various operating systems, including Linux, Windows, etc.:
    @staticmethod
    def st_test_1():

        success_count = 0
        elapsed_time = 0
        use_ray = False

        module = DatasetModule.deploy(actor=False)
        # module.load_dataset(refresh=False)
        # module.load_bitmodule(refresh=False)
        # module.bitmodule.set_wallet(name='const', hotkey='Tiberius')

        from ray.experimental.state.api import summarize_objects
        print(summarize_objects())

        with st.sidebar.expander('Receptor Pool', True):
            refresh = st.button('Refresh')
            receptor_pool = module.load_receptor_pool(refresh=refresh)
            

        # with st.expander('Text', False):
        #     input_text = st.text_area('Input Text')

        # metrics_dict = {}
        # with st.sidebar.expander('Query', True):

        #     with st.form('Fire'):
        #         batch_size = st.slider('batch size', 1, 128, 5)
        #         num_endpoints = st.slider('num endpoints', 1, 1000, 50)
        #         timeout = st.select_slider('timeout', list(np.arange(0.0, 5.0, 0.1)), 1.0)
        #         batch_count = st.select_slider('batch count', list(range(1,10)), 1)
        #         min_success = st.select_slider('min_success',list(np.arange(0.0, 1.0, 0.1)) , 0.5)

        #         all_synapses = module.bitmodule.getattr('available_synapses')
        #         synapse2idx = {s:s_i for s_i, s in enumerate(all_synapses)}
        #         synapses = st.multiselect('synapspe', all_synapses, ['TextLastHiddenState'])
        #         synapses = list(map(module.bitmodule.str2synapse, synapses))

        
        #         submit_button = st.form_submit_button('Fire')


        #         if submit_button:
        #             metrics_dict = module.get_samples(synapses=synapses, timeout=timeout, batch_count=batch_count, num_endpoints=num_endpoints, min_success=min_success , batch_size=batch_size)
    

        # if len(metrics_dict)>0: 
        #     for k in ['tokens', 'samples', 'queries', 'successes', 'upload_mb', 'download_mb']: 
        #         metrics_dict[f"{k}_per_second"] = round_sig(metrics_dict[k] / (metrics_dict['elapsed_seconds']), 3)



        #     total_metrics =  len(metrics_dict)
        #     num_cols = 3
        #     num_rows = total_metrics// num_cols
        #     last_row_cols = total_metrics % num_cols

        #     rows = []

        #     for i, k in enumerate(metrics_dict.keys()):
        #         if i % num_cols == 0:
        #             row_column_count = num_cols
        #             rows.append(st.columns([1]*row_column_count))
                
        #         rows[-1][(i % num_cols)].metric(f'{k}', metrics_dict[k])

        #     st.write(f'Num successful returns: {success_count} (Time: {elapsed_time})')

        #     import gc
        #     gc.collect()


        # with st.sidebar.expander('Ray', True):
        #     restart_ray_cluster = st.button('Restart Ray Cluster')
        #     if restart_ray_cluster:
        #         BaseModule.ray_restart()
        #     stop_ray_cluster = st.button('Stop Ray Cluster')
        #     if stop_ray_cluster:
        #         BaseModule.ray_stop()

        #     start_ray_cluster = st.button('Start Ray Cluster')
        #     if start_ray_cluster:
        #         BaseModule.ray_start()




        # with st.expander('Resource Usage'):
        #     actor_name =module.receptor_pool.getattr('actor_name')
        #     memory_dict = {
        #         actor_name: receptor_pool.memory_usage(mode='percent'),
        #         'other': receptor_pool.memory_used(mode='percent') - receptor_pool.memory_usage(mode='percent'),
        #         'free': receptor_pool.memory_available(mode='percent'),
        #     }

        #     import plotly.express as px
        
        #     fig = px.pie( values=list(memory_dict.values()), names=list(memory_dict.keys()), title='Memory Usage')
        #     st.write(fig)
    
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
    generator_jobs = {}
    def start_generator(self, key=None, **kwargs):
        if key == None:
            key = kwargs.get('split')
        self.generator_map[key] = self.sample_generator(**kwargs) 

    def kill_generator(self, key=None):
        self.generator_map.pop(key,None)
        jobs = self.generator_jobs.pop(key,None)
        ray.get(jobs)

    rm_generator = kill_generator
    def ls_generators(self):
        return list(self.generator_map.keys())

    def generate_sample(self, key):
        return next(self.generator_map[key])

    def num_generator_jobs(self, key):
        return len(self.generator_jobs.get(key, []))
    def sample_generator(self, 

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
                    success_only=False,
                    generator=False,
                    split=None,
                    queue=None):

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
                        if str(synapse) not in ['TextCausalLM', 'TextLastHiddenState']:
                            seq_multiplier = 1

                        chunked_tensors = torch.chunk(batch_chunked_tensor, chunks=seq_multiplier, dim=2)
                        for chunked_tensor in chunked_tensors:
                            sample_result = {**results, **{'tensor': chunked_tensor}}
                            # finished_results.append(sample_result)
                            # if generator:
                            yield sample_result
        
        self.kill_generator(split)
        # queue_kwargs = {}
        # if isinstance(queue, str):
        #     queue_kwargs['topic'] = queue

        # elif isinstance(queue, dict):
        #     queue_kwargs = queue
        # for finished_result in finished_results:

        #     queue_kwargs['item'] = finished_result
        #     job = self.queue.put(**queue_kwargs)

        # self.sample_generator_count -= 1 
        # return finished_results
        

    def process_results(self, results, synapse):
        results_dict = {'tensor':[], 'code':[], 'latency':[]}

        num_responses = len(results[0])
        for i in range(num_responses):
            tensor = results[0][i][0]
            code = results[1][i][0]
            latency = results[2][i][0]


            if str(synapse) in ['TextCausalLMNext']:
                if tensor.shape[-1] != 2:
                    continue

            results_dict['tensor'].append(tensor)
            results_dict['code'].append(code)
            results_dict['latency'].append(latency)


        results_dict['tensor'] = torch.stack(results_dict['tensor'])
        results_dict['code'] = torch.tensor(results_dict['code'])
        results_dict['latency'] = torch.tensor(results_dict['latency'])

        return results_dict


    def splits(self):
        return self.dataset.getattr('splits')

    def sample(self,
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

        
        if ray_get:
            results = self.receptor_pool.forward(**forward_kwargs, ray_get=ray_get)
            results = self.process_results(results=results, synapse=synapse)
            return results
        else:
            results_job = self.receptor_pool.forward(**forward_kwargs, ray_get=ray_get)
            return results_job

if __name__ == '__main__':
    
    import streamlit as st
    import time
    st.set_page_config(layout="wide")
    # DatasetModule.st_test_1()

    # DatasetModule.ray_restart()
    module = DatasetModule.deploy(actor={'refresh': False}, wrap=True)

    # module.load_dataset(refresh=True)
    # module.load_queue(refresh=True)

    # st.write(module.queue)
    # st.write(module.queue.memory_usage())
    # module.bitmodule.set_wallet(name='const', hotkey='Tiberius')
    # st.write(module.bitmodule.getattr('wallet'))
    # with st.sidebar.expander('Receptor Pool', True):
    #     refresh = st.button('Refresh')
    #     receptor_pool = module.load_receptor_pool(refresh=refresh)
    # st.write(module.receptor_pool.memory_usage('ratio')
    # st.write(module.str2synapse(module.available_synapses[0]))
    # st.write(module.sample( batch_multiplier=2, batch_size=6, seq_len=10, seq_multiplier=2, timeout=2, num_endpoints=10)['tensor'].shape)
    # st.write(module.receptor_pool.memory_usage('ratio'))

    # st.write(module.getattr('queue').size('fam'))
    # st.write(module.getattr('sample_generator_count'))
    # module.sample_generator_loop( loops=20,  batch_multiplier=4, batch_size=6, seq_len=10, seq_multiplier=2, timeout=2, num_endpoints=10, split='train', queue='train')
    # module.start_generator( num_samples=100, batch_multiplier=4, batch_size=6, seq_len=10, seq_multiplier=2, timeout=2, num_endpoints=10, split='train')

    # st.write(module.num_generator_jobs('train'))

    for i in range(100):
        st.write(module.generate_sample('train'))
    # for sample in :
    #     st.write(sample['tensor'].shape)

    # for sample in enumerate(module.sample_generator(  batch_multiplier=4, batch_size=6, seq_len=10, seq_multiplier=2, timeout=2, num_endpoints=10, split='train', queue='train'):
        

    # st.write(module.getattr('queue').delete('fam'))    
    # st.write(module.getattr('queue').put('train', 'bro'))
    
    # st.write(module.queue.memory_usage('ratio'))
    # # # st.write(module.queue.delete('test'))

    # st.write(module.getattr('queue').size_map())
    # for i in range(100):
    #     st.write(module.queue.memory_usage('ratio'))
    #     module.getattr('queue').get('test')
    #     st.write(module.getattr('queue').size_map())
    # for i in range(module.getattr('queue').size_map('train')):
    #     module.getattr('queue').get('train')
    #     module.getattr('queue').size('train')
    #     st.write(i)
    # st.write(module.getattr('queue').get('demo')['tensor'].shape)
    # st.write(module.load_receptor_pool(True))  
    # st.write(module.getattr('dataset').memory_usage('ratio'))
    # BaseModule.ray_restart()