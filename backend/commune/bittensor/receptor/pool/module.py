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

import asyncio
logger = logger.opt(colors=True)

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
        self.max_worker_threads = max_worker_threads
        self.max_active_receptors = max_active_receptors
        self.receptors = {}
        self.cull_mutex = Lock()
        self.max_processes = 10
        self.compression = compression
        self.total_requests = 0


        
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
    def forward (
            self, 
            endpoints: List [ 'bittensor.Endpoint' ],
            synapses: List[ 'bittensor.Synapse' ],
            inputs: List [ torch.Tensor ],
            timeout: int,
            min_success = 5,
            return_success_only=False, 
            refresh=False,
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
        if min_success < 1:
            min_success = int(min_success*len(endpoints))

        # Submit calls to receptors.
        with concurrent.futures.ThreadPoolExecutor( max_workers = len(endpoints) ) as executor:
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
                    return forward_outputs, forward_codes, forward_times


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

if __name__ == '__main__':
    import streamlit as st
    st.set_page_config(layout="wide")
    # BaseModule.ray_restart()
    dataset_class =  BaseModule.get_object('bittensor.cortex.dataset.module.DatasetModule')
    dataset = dataset_class.deploy(actor={'refresh': False}, load=['env', 'tokenizer', 'dataset'], wrap = True)
   
    



    success_count = 0
    elapsed_time = 0


    receptor_pool = ReceptorPoolModule.deploy(actor={'refresh': False}, wallet=dataset.getattr('wallet'), wrap=True)


    with st.sidebar.expander('Receptor Pool', True):
        refresh = st.button('Refresh')
        if refresh:
            receptor_pool = ReceptorPoolModule.deploy(actor={'refresh': True}, wallet=dataset.getattr('wallet'), wrap=True)


        st.write('Actor_name',receptor_pool.actor_name)



    import time

        # st.write(t.elapsed_time)
    with st.sidebar.expander('Query', True):

        with st.form('Fire'):
            batch_size = st.slider('batch size', 1, 32, 5)
            num_endpoints = st.slider('num endpoints', 1, 200, 50)
            timeout = st.select_slider('timeout', list(np.arange(0.0, 5.0, 0.1)), 1.0)
            batch_count = st.select_slider('batch count', list(range(1,10)), 1)
            min_success = st.select_slider('min_success',list(np.arange(0.0, 1.0, 0.1)) , 0.5)

            all_synapses = dataset.getattr('available_synapses')
            synapse2idx = {s:s_i for s_i, s in enumerate(all_synapses)}
            synapses = st.multiselect('synapspe', all_synapses, ['TextLastHiddenState'])
            synapses = list(map(dataset.str2synapse, synapses))

            return_success_only = True
            submit_button = st.form_submit_button('Fire')

            results = None
            running_jobs = []
            metrics_dict=dict(
                samples = 0, 
                tokens = 0,
                queries = 0, 
                successes= 0,
                elapsed_seconds =  -1,

            )

            if submit_button:
                with BaseModule.timer('Time: {t}', streamlit=False) as t: 
                    for i in range(batch_count):
                        endpoints = dataset.get_endpoints(num_endpoints=num_endpoints)


                        inputs = dataset.sample_raw_batch(batch_size=batch_size, tokenize=False)
                        
                        # inputs [Batch, Seq Length]
                        inputs = dataset.tokenize(inputs, padding=True)
                        job = receptor_pool.forward(inputs= inputs ,synapses=synapses, timeout=timeout, endpoints=endpoints, min_success=min_success,return_success_only=True, ray_get=False)
                        running_jobs.append(job)

                        metrics_dict['queries'] += len(endpoints)
                        metrics_dict['samples'] += inputs.shape[0] *  len(endpoints)
                        metrics_dict['tokens'] += inputs.numel() *  len(endpoints)
                        
     
   
                    results_batch = ray.get(running_jobs)
                    del running_jobs
                    del job
                    metrics_dict['elapsed_seconds'] = t.elapsed_seconds
                for results in results_batch:
                    metrics_dict['successes'] += len(results[0])

            for k in ['tokens', 'samples', 'queries', 'successes']: 
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
