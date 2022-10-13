""" Manages a pool of grpc connections as receptors
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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


    def load(self):
        self.load_dataset()
        self.load_receptor_pool()
        self.load_bitmodule()


    def load_dataset(self, refresh=False):
        dataset_class =  BaseModule.get_object('huggingface.dataset.module.DatasetModule')
        self.dataset = dataset_class.deploy(actor={'refresh': refresh}, load=['tokenizer', 'dataset'], wrap = True)

    def load_receptor_pool(self, refresh=False):
        receptor_pool_class =  BaseModule.get_object('bittensor.receptor.pool.module.ReceptorPoolModule')
        self.receptor_pool = receptor_pool_class.deploy(actor={'refresh': refresh},wallet=self.bitmodule.getattr('wallet'), wrap=True)
        return self.receptor_pool

    def load_bitmodule(self, refresh=False):
        module_class = BaseModule.get_object('bittensor.base.module.BitModule')
        self.bitmodule = module_class.deploy(actor={'refresh': refresh},load=True, wrap = True)
        return self.bitmodule

    def __str__(self):
        return "ReceptorPool({},{})".format(len(self.receptors), self.max_active_receptors)

    def __repr__(self):
        return self.__str__()
    
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


    def get_samples(self, num_endpoints=10, batch_count=10, batch_size=10, timeout=1, synapses=None, min_success=0.5):

        return_success_only = True
        results = None
        job2inputs_dict = {}
        metrics_dict=dict(
            samples = 0, 
            tokens = 0,
            queries = 0, 
            successes= 0,
            elapsed_seconds =  -1,
            upload_mb=0,
            download_mb=0, 

        )

        io_1 = psutil.net_io_counters()
        start_bytes_sent, start_bytes_recv = io_1.bytes_sent, io_1.bytes_recv

        with BaseModule.timer('Time: {t}', streamlit=False) as t: 
            inputs_batch = []
            forward_kwargs_list = []

            for i in range(batch_count):
                endpoints = self.bitmodule.get_endpoints(num_endpoints=num_endpoints)
                inputs = self.dataset.sample(batch_size=batch_size)
                
                inputs_batch.append(inputs)
                metrics_dict['queries'] += len(endpoints)
                forward_kwargs_list.append(dict(inputs= inputs ,synapses=synapses, timeout=timeout, endpoints=endpoints,
                            min_success=min_success,return_success_only=True))


            for forward_kwargs in forward_kwargs_list:
                job = self.receptor_pool.forward(**forward_kwargs, ray_get=False)
                job2inputs_dict[job] = inputs

        
            running_jobs = list(job2inputs_dict.keys())
            while len(running_jobs)>0:
                finished_jobs, running_jobs = ray.wait(running_jobs)
                if len(finished_jobs) > 0:
                    for job in finished_jobs:
                        results = ray.get(job)
                        successes = len(results[0])
                        metrics_dict['successes'] += successes
                        inputs = job2inputs_dict.pop(job)
                        metrics_dict['samples'] += inputs.shape[0] *  successes
                        st.write(inputs.shape)
                        metrics_dict['tokens'] += inputs.shape[0] * inputs.shape[1] *  successes
                    
                    del finished_jobs
            metrics_dict['elapsed_seconds'] = t.elapsed_seconds


        io_2 = psutil.net_io_counters()
        total_bytes_sent = round_sig(io_2.bytes_sent - start_bytes_sent, 3)
        total_bytes_recved = round_sig(io_2.bytes_recv - start_bytes_recv,3) 

        metrics_dict['upload_mb'] = total_bytes_sent * 10e-6
        metrics_dict['download_mb'] = total_bytes_recved *  10e-6


        for k in ['tokens', 'samples', 'queries', 'successes', 'upload_mb', 'download_mb']: 
            metrics_dict[f"{k}_per_second"] = round_sig(metrics_dict[k] / (metrics_dict['elapsed_seconds']), 3)

        return metrics_dict
        

    # Here is a useful solution that works for various operating systems, including Linux, Windows, etc.:
    @staticmethod
    def st_test_1():

        success_count = 0
        elapsed_time = 0
        use_ray = False

        module = DatasetModule.deploy(actor=False)
        st.write('bro')
        module.load_dataset(refresh=False)
        module.load_bitmodule(refresh=False)
        module.bitmodule.set_wallet(name='const', hotkey='Tiberius')

        st.write(module.bitmodule.getattr('wallet'))


        with st.sidebar.expander('Receptor Pool', True):
            refresh = st.button('Refresh')
            receptor_pool = module.load_receptor_pool(refresh=refresh)
            

        with st.expander('Text', False):
            input_text = st.text_area('Input Text')

        metrics_dict = {}
        with st.sidebar.expander('Query', True):

            with st.form('Fire'):
                batch_size = st.slider('batch size', 1, 128, 5)
                num_endpoints = st.slider('num endpoints', 1, 1000, 50)
                timeout = st.select_slider('timeout', list(np.arange(0.0, 5.0, 0.1)), 1.0)
                batch_count = st.select_slider('batch count', list(range(1,10)), 1)
                min_success = st.select_slider('min_success',list(np.arange(0.0, 1.0, 0.1)) , 0.5)

                all_synapses = module.bitmodule.getattr('available_synapses')
                synapse2idx = {s:s_i for s_i, s in enumerate(all_synapses)}
                synapses = st.multiselect('synapspe', all_synapses, ['TextLastHiddenState'])
                synapses = list(map(module.bitmodule.str2synapse, synapses))

        
                submit_button = st.form_submit_button('Fire')


                if submit_button:
                    metrics_dict = module.get_samples(synapses=synapses, timeout=timeout, batch_count=batch_count, num_endpoints=num_endpoints, min_success=min_success , batch_size=batch_size)
    

        if len(metrics_dict)>0: 
            for k in ['tokens', 'samples', 'queries', 'successes', 'upload_mb', 'download_mb']: 
                metrics_dict[f"{k}_per_second"] = round_sig(metrics_dict[k] / (metrics_dict['elapsed_seconds']), 3)



            total_metrics =  len(metrics_dict)
            num_cols = 3
            num_rows = total_metrics// num_cols
            last_row_cols = total_metrics % num_cols

            rows = []

            for i, k in enumerate(metrics_dict.keys()):
                if i % num_cols == 0:
                    row_column_count = num_cols
                    rows.append(st.columns([1]*row_column_count))
                
                rows[-1][(i % num_cols)].metric(f'{k}', metrics_dict[k])

            st.write(f'Num successful returns: {success_count} (Time: {elapsed_time})')

            import gc
            gc.collect()


        with st.sidebar.expander('Ray', True):
            restart_ray_cluster = st.button('Restart Ray Cluster')
            if restart_ray_cluster:
                BaseModule.ray_restart()
            stop_ray_cluster = st.button('Stop Ray Cluster')
            if stop_ray_cluster:
                BaseModule.ray_stop()

            start_ray_cluster = st.button('Start Ray Cluster')
            if start_ray_cluster:
                BaseModule.ray_start()




        with st.expander('Resource Usage'):
            actor_name =module.receptor_pool.getattr('actor_name')
            memory_dict = {
                actor_name: receptor_pool.memory_usage(mode='percent'),
                'other': receptor_pool.memory_used(mode='percent') - receptor_pool.memory_usage(mode='percent'),
                'free': receptor_pool.memory_available(mode='percent'),
            }

            import plotly.express as px
        
            fig = px.pie( values=list(memory_dict.values()), names=list(memory_dict.keys()), title='Memory Usage')
            st.write(fig)
    
    ############################
    ##### Forward Function #####
    ############################
    # Forward function queries (n_queried) random endpoints with the inputs
    # then waits timeout before checking for success from each query.
    # The function returns a list of booleans True or false depending on the query result.

if __name__ == '__main__':
    
    import streamlit as st
    import time
    st.set_page_config(layout="wide")
    DatasetModule.st_test_1()
    # BaseModule.ray_restart()
   
   