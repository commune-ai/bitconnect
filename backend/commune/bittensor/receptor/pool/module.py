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
import pandas as pd
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

class ReceptorPoolModule (BaseModule, torch.nn.Module ):
    """ Manages a pool of grpc connections as receptors
    """
    default_config_path = 'bittensor.receptor.pool'

    def __init__(
        self, 
        wallet: 'bittensor.Wallet',
        max_worker_threads: int = 150,
        max_active_receptors: int= 150,
        compression: str= None,
        config = None,
        override= {},
    ):
        torch.nn.Module.__init__(self)
        BaseModule.__init__(self, config=config, override=override)




        self.wallet = wallet
        self.receptors = {}
        self.max_worker_threads = max_worker_threads
        self.max_active_receptors = max_active_receptors

        self.cull_mutex = Lock()
        self.max_processes = 10
        self.compression = compression
        self.total_requests = 0


        if self.wallet == None:
            default_wallet = {'name': 'const', 'hotkey': 'Tiberius'}
            self.wallet = bittensor.wallet(**self.config.get('wallet', default_wallet))




        
        try:
            self.external_ip = str(net.get_external_ip())
        except Exception:
            self.external_ip = None

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


    def rm_receptor(self, key):
        self.receptors[ key ].close()
        del self.receptors[ key ]
        return key

    def rm_all(self):
        for key in  deepcopy(list(self.receptors.keys())):
            self.rm_receptor(key=key)


    refresh = rm_all
    def forward(
            self, 
            endpoints: List [ 'bittensor.Endpoint' ],
            synapses: List[ 'bittensor.Synapse' ],
            inputs: List [ torch.Tensor ],
            timeout: int,
            min_success = 5,
            return_success_only=False, 
            refresh=False,
            return_type = 'tuple',
            max_workers=10
        ) -> Tuple[List[torch.Tensor], List[int], List[float]]:
        r""" Forward tensor inputs to endpoints.
            Args:
                endpoints (:obj:`List[ bittensor.Endpoint ]` of shape :obj:`(num_endpoints)`, `required`):
                    List of remote endpoints which match length of inputs. Tensors from x are sent forward to these endpoints.
                synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                    Responses are packed in this ordering. 
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    TODO(const): Allow multiple tensors.
                    List of tensors to send to corresponsing endpoints. Tensors are of arbitrary type and shape depending on the
                    modality.
                timeout (int):
                    Request timeout.
            Returns:
                forward_outputs (:obj:`List[ List[ torch.FloatTensor ]]` of shape :obj:`(num_endpoints * (num_synapses * (shape)))`, `required`):
                    Output encodings of tensors produced by remote endpoints. Non-responses are zeroes of common shape.
                forward_codes (:obj:`List[ List[bittensor.proto.ReturnCodes] ]` of shape :obj:`(num_endpoints * ( num_synapses ))`, `required`):
                    dendrite backward call return ops.
                forward_times (:obj:`List[ List [float] ]` of shape :obj:`(num_endpoints * ( num_synapses ))`, `required`):
                    dendrite backward call times
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(endpoints) != len(inputs):
            if len(inputs) == 1:
                inputs = len(endpoints)*inputs
            else:
                raise ValueError('Endpoints must have the same length as passed inputs. Got {} and {}'.format(len(endpoints), len(inputs)))


        receptors = [ self._get_or_create_receptor_for_endpoint( endpoint ) for endpoint in endpoints ]

        # Init argument iterables.
        call_args = []
        for idx, receptor in enumerate( receptors ):
            call_args.append({ 
                'receptor': receptor, 
                'inputs': inputs [ idx ] ,
                'synapses': synapses, 
                'timeout': timeout
            }) 

        # Init function.
        def call_forward( args ):
            return args['receptor'].forward( args['synapses'], args['inputs'], args['timeout'] )
        
        responses=[]
        # Unpack responses
        forward_outputs = []
        forward_codes = []
        forward_times = []

        assert min_success > 0
        if min_success <= 1:
            min_success = int(min_success*len(endpoints))

        # Submit calls to receptors.
        with concurrent.futures.ThreadPoolExecutor( max_workers = max_workers ) as executor:
            future_map = {}

            for idx, call_arg in enumerate(call_args):
                future = executor.submit( call_forward, call_arg)
                future_map[future] = call_arg

            success_response_cnt = 0
            for i,future in enumerate(concurrent.futures.as_completed(future_map)):
                response = future.result()
                


                            
                if response[1][0] == 1:
                    success_response_cnt += 1

                    forward_outputs.append( response[0] )
                    forward_codes.append( response[1] )
                    forward_times.append( response[2] )
                else:
                    if not return_success_only:
                        forward_outputs.append( response[0] )
                        forward_codes.append( response[1] )
                        forward_times.append( response[2] )
                                
                if success_response_cnt >= min_success:
                    for receptor in receptors:
                        receptor.semaphore.release()
                    self._destroy_receptors_over_max_allowed()
                    # ---- Return ----

                    if return_type in ['tuple',  tuple]:
                        return forward_outputs, forward_codes, forward_times
                    elif return_typpe in ['dict', dict]:
                        return dict(
                            outputs=forward_outputs,
                            codes =forward_codes,
                            times = forward_times
    
                        )


        # Release semephore.
        for receptor in receptors:
            receptor.semaphore.release()
        self._destroy_receptors_over_max_allowed()
        # ---- Return ----

        if refresh:
            self.refresh()
        

            
        return forward_outputs, forward_codes, forward_times

    def backward(
                self, 
                endpoints: List [ 'bittensor.Endpoint' ],
                synapses: List[ 'bittensor.Synapse' ],
                inputs: List [ torch.Tensor ],
                grads: List [ List[ torch.FloatTensor ] ],
                timeout: int
            ) -> Tuple[List[torch.Tensor], List[int], List[float]]:
        r""" Backward tensor inputs to endpoints.
            Args:
                endpoints (:obj:`List['bittensor.Endpoint']` of shape :obj:`(num_endpoints)`, `required`):
                    List of remote endpoints which match length of x. Tensors from x are sent backward to these endpoints.
                synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                    Responses are packed in this ordering. 
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of tensors to send to corresponsing endpoints. Tensors are of arbitrary type and shape depending on the
                    synapse.
                grads (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of list of grad tensors where each grad corresponds to a synapse call on an endpoint.
                
                timeout (int):
                    request timeout.
            Returns:
                backward_outputs (:obj:`List[ List[ torch.FloatTensor] ]` of shape :obj:`num_endpoints * (batch_size, sequence_len, -1)]`, `required`):
                    Gradients returned from the backward call one per endpoint.
                backward_codes (:obj:`List[ List[ bittensor.proto.ReturnCodes ] ]` of shape :obj:`(num_endpoints)`, `required`):
                    List of list of Backward call return ops, one per endpoint and synapse.
                backward_times (:obj:`List[float]` of shape :obj:`(num_endpoints)`, `required`):
                    List of list of Backward call times one per endpoint and synapse.
        """
        if len(endpoints) != len(inputs):
            raise ValueError('Endpoints must have the same length as passed inputs. Got {} and {}'.format(len(endpoints), len(inputs)))
        if len(endpoints) != len(grads):
            raise ValueError('Endpoints must have the same length as passed grads_dy. Got {} and {}'.format(len(endpoints), len(grads)))
        for grads_per_synapse in grads:
            if len(grads_per_synapse) != len(synapses):
                raise ValueError('Gradients must have the same length as passed synapses. Got {} and {}'.format(len(grads_per_synapse), len(synapses)))

        # Init receptors.
        receptors = [ self._get_or_create_receptor_for_endpoint( endpoint ) for endpoint in endpoints ]

        # Init argument iterables.
        call_args = []
        for idx, receptor in enumerate( receptors ):
            call_args.append({ 
                'receptor': receptor, 
                'synapses': synapses, 
                'inputs': inputs [ idx ] ,
                'grads': grads [ idx ] ,
                'timeout': timeout
            }) 

        # Init function.
        def call_backward( args ):
            return args['receptor'].backward ( 
                synapses = args['synapses'], 
                inputs = args['inputs'], 
                grads = args['grads'], 
                timeout = args['timeout'] 
            )

        # Submit calls to receptors.
        with concurrent.futures.ThreadPoolExecutor( max_workers = len(endpoints) ) as executor:
            responses = executor.map ( call_backward, call_args, timeout=10*timeout )

        # Release semephore.
        for receptor in receptors:
            receptor.semaphore.release()
            
        # Unpack responses
        backward_outputs = []
        backward_codes = []
        backward_times = []
        for response in responses:
            backward_outputs.append( response[0] )
            backward_codes.append( response[1] )
            backward_times.append( response[2] )

        # ---- Kill receptors ----
        self._destroy_receptors_over_max_allowed()
        # ---- Return ----
        return backward_outputs, backward_codes, backward_times

    def _destroy_receptors_over_max_allowed( self ):
        r""" Destroys receptors based on QPS until there are no more than max_active_receptors.
        """
        with self.cull_mutex:
            # ---- Finally: Kill receptors over max allowed ----
            while len(self.receptors) > self.max_active_receptors:
                min_receptor_qps = math.inf
                receptor_to_remove = None
                for next_receptor in self.receptors.values():
                    next_qps = next_receptor.stats.forward_qps.value
                    sema_value = next_receptor.semaphore._value
                    if (min_receptor_qps > next_qps) and (sema_value == self.max_processes):
                        receptor_to_remove = next_receptor
                        min_receptor_qps = next_receptor.stats.forward_qps.value
                        
                if receptor_to_remove != None:
                    try:
                        bittensor.logging.destroy_receptor_log(receptor_to_remove.endpoint)
                        self.receptors[ receptor_to_remove.endpoint.hotkey ].close()
                        del self.receptors[ receptor_to_remove.endpoint.hotkey ]
                    except KeyError:
                        pass
                elif receptor_to_remove == None:
                    break

    def _get_or_create_receptor_for_endpoint( self, endpoint: 'bittensor.Endpoint' ) -> 'bittensor.Receptor':
        r""" Finds or creates a receptor TCP connection associated with the passed Neuron Endpoint
            Returns
                receptor: (`bittensor.Receptor`):
                    receptor with tcp connection endpoint at endpoint.ip:endpoint.port
        """
        # ---- Find the active receptor for this endpoint ----
        # if endpoint.hotkey in self.receptors:
        #     receptor = self.receptors[ endpoint.hotkey ]

        #     # Change receptor address.
        #     if receptor.endpoint.ip != endpoint.ip or receptor.endpoint.port != endpoint.port:
        #         #receptor.close()
        #         bittensor.logging.update_receptor_log( endpoint )
        #         receptor = bittensor.receptor (
        #             endpoint = endpoint, 
        #             wallet = self.wallet,
        #             external_ip = self.external_ip,
        #             max_processes = self.max_processes
        #         )            
        #         self.receptors[ receptor.endpoint.hotkey ] = receptor

        # # ---- Or: Create a new receptor ----
        # else:
        bittensor.logging.create_receptor_log( endpoint )
        receptor = bittensor.receptor (
                endpoint = endpoint, 
                wallet = self.wallet,
                external_ip = self.external_ip,
                max_processes = self.max_processes,
                compression = self.compression
        )
        # self.receptors[ receptor.endpoint.hotkey ] = receptor
            
        receptor.semaphore.acquire()
        return receptor

    # Here is a useful solution that works for various operating systems, including Linux, Windows, etc.:

    def set_wallet(self, name:str=None, hotkey:str=None,**kwargs):

        wallet_config = self.config.get('wallet', self.default_wallet_config)
        wallet_config['name'] = wallet_config['name'] if name == None else name
        wallet_config['hotkey'] = wallet_config['hotkey'] if hotkey == None else hotkey

        self.wallet = bittensor.wallet(**wallet_config)
        
        self.config['wallet'] = wallet_config

        return self.wallet
    @staticmethod
    def st_test_1():
        return_info_dict = {'codes': [], 'latency': []}
        dataset_class =  BaseModule.get_object('bittensor.cortex.dataset2.module.DatasetModule')
        st.write(dataset_class)
        dataset = dataset_class.deploy(actor={'refresh': False}, load=True, wrap = True)
        success_count = 0
        elapsed_time = 0
        use_ray = False

        with st.sidebar.expander('Receptor Pool', True):
            refresh = st.button('Refresh')
            receptor_pool = ReceptorPoolModule.deploy(actor={'refresh': refresh}, wallet=None, wrap=True)

        with st.expander('Text', False):
            input_text = st.text_area('Input Text')

        with st.sidebar.expander('Query', True):

            with st.form('Fire'):
                batch_size = st.slider('batch size', 1, 128, 5)
                bundle_batch_factor = st.slider('batch multiplier', 1, 10, 1)
                sequence_length = st.slider('sequence length', 1, 256, 10)
                sequence_multiplier = st.slider('sequence multiplier', 1, 10, 1)
                num_endpoints = st.slider('num endpoints', 1, 100, 50)
                timeout = st.select_slider('timeout', list(np.arange(0.0, 5.0, 0.1)), 1.0)
                batch_count = st.slider('num samples',1, 100, 1)
                min_success = st.slider('min_success',2, 100 , 30)


                all_synapses = dataset.getattr('available_synapses')
                synapse2idx = {s:s_i for s_i, s in enumerate(all_synapses)}

                synapse_str = st.selectbox('synapspe', all_synapses, synapse2idx['TextLastHiddenState'])
                synapse =getattr(bittensor.synapse, synapse_str)
                # st.write(synapse)
                if synapse_str in ['TextCausalLMNext','TextCausalLM']:
                    synapse =synapse(decode_topk=False)
                else:
                    synapse = synapse()
                # synapses = [synapse]




                return_success_only = True
                submit_button = st.form_submit_button('Fire')

                results = None
                job2inputs_dict = {}
                metrics_dict=dict(
                    samples = 0, 
                    tokens = 0,
                    queries = 0, 
                    steps=0,
                    successes= 0,
                    elapsed_seconds =  -1,
                    upload_mb=0,
                    download_mb=0, 

                )

                if submit_button:
                    io_1 = psutil.net_io_counters()
                    start_bytes_sent, start_bytes_recv = io_1.bytes_sent, io_1.bytes_recv

                    with BaseModule.timer('Time: {t}', streamlit=False) as t: 
                        inputs_batch = []
                        forward_kwargs_list = []
                        st.write(len(job2inputs_dict),'initial ')
     
                        for i in range(batch_count):
                            endpoints = dataset.get_endpoints(num_endpoints=num_endpoints)
                            inputs = torch.ones([batch_size*bundle_batch_factor, sequence_length], dtype=torch.int64)
                            # inputs = dataset.sample_raw_batch(batch_size=batch_size, tokenize=False)  
                            # inputs [Batch, Seq Length]

                            # inputs = dataset.tokenize(inputs, padding=True)
                            inputs_batch.append(inputs)
                            metrics_dict['queries'] += len(endpoints)
                            forward_kwargs_list.append(dict(inputs= inputs ,synapses=[synapse], timeout=timeout, endpoints=endpoints,
                                        min_success=min_success,return_success_only=True))


                        for forward_kwargs in forward_kwargs_list:
                            job = receptor_pool.forward(**forward_kwargs, ray_get=False)
                            job2inputs_dict[job] = inputs

                    

                        


                        running_jobs = list(job2inputs_dict.keys())


                        while len(running_jobs)>0:
                            # running_jobs = list(job2inputs_dict.keys())
                            st.write(len(running_jobs))

                            finished_jobs, running_jobs = ray.wait(running_jobs)
                            if len(finished_jobs) > 0:
                                for job in finished_jobs:
                                    results = ray.get(job)

                                    # st.write(results[0][0][0].shape)
                                    return_info_dict['codes'] += [c[0] for c in results[1]]
                                    return_info_dict['latency'] += [c[0] for c in results[2]]
                                    successes = len([c for c in results[1] if c[0] == 1])

                                    metrics_dict['successes'] += successes
                                    inputs = job2inputs_dict.pop(job)
                                    metrics_dict['samples'] += inputs.shape[0] *  successes

                                    if successes > 0:
                                        st.write(results[0][0][0].shape)
                                        metrics_dict['steps'] += 1*bundle_batch_factor
                                    metrics_dict['tokens'] += inputs.shape[0] * inputs.shape[1] *  successes
                                
                                del finished_jobs
                        metrics_dict['elapsed_seconds'] = t.elapsed_seconds
                        return_info_df = pd.DataFrame(return_info_dict)

                        # Measure state after.
                    io_2 = psutil.net_io_counters()
                    total_bytes_sent = round_sig(io_2.bytes_sent - start_bytes_sent, 3)
                    total_bytes_recved = round_sig(io_2.bytes_recv - start_bytes_recv,3) 

                    metrics_dict['upload_mb'] = total_bytes_sent * 10e-6
                    metrics_dict['download_mb'] = total_bytes_recved *  10e-6
    
                for k in ['tokens', 'samples', 'queries', 'successes', 'upload_mb', 'download_mb', 'steps']: 
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

        import plotly.express as px

    
        return_info_df = pd.DataFrame(return_info_dict)

        with st.expander('Response Charts'):

            code_count_dict = {k:int(v) for k,v in dict(return_info_df['codes'].value_counts()).items()}
            fig = px.pie( values=list(code_count_dict.values()), names=list(code_count_dict.keys()), title='Return Code Statistics')
            st.write(fig)

            fig = px.histogram(return_info_df['latency'], title='Latency Statistics')
            st.write(fig)


        

        with st.expander('Resource Usage'):
            actor_name =receptor_pool.getattr('actor_name')
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
    ReceptorPoolModule.st_test_1()
    st.write('fam')
    # BaseModule.ray_restart()
   
   