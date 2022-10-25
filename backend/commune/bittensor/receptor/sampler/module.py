
""" Manages a pool of grpc connections as self.receptors
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
import random
from commune.utils import round_sig
import bittensor.utils.networking as net
from concurrent.futures import ThreadPoolExecutor
from commune import Module
from commune.bittensor.base.module import BitModule
from commune.bittensor.receptor.receptor.module import ReceptorModule
import ray

import asyncio
logger = logger.opt(colors=True)

class SamplerModule (BitModule ):
    """ Manages a pool of grpc connections as self.receptors
    """
    default_config_path = 'bittensor.receptor.sampler'

    def __init__(
        self, 
        config = None,
        **kwargs
    ):
        BitModule.__init__(self, config=config, **kwargs)
        # A list of pre-instantiated endpoints with stub connections.
        self.sync()
        self.receptors = self.get_receptors()

        
    def get_receptors(self):
        # A list of pre-instantiated endpoints with stub connections.
        return  [bittensor.receptor( wallet = self.wallet, endpoint = self.metagraph.endpoint_objs[i] ) for i in range(self.metagraph.n)]


    def forward(self, inputs, synapse= None,  n_queried=10,  sample_ids=None, timeout=1):
        
        # Preable.
        futures = []
        result = [False for _ in range(n_queried)]
        
        # Randomly select (n_queried) peers to sample.
        if sample_ids == None:
            sampled_ids = random.sample( range(4096), n_queried )
    
        for i in sampled_ids:
            
            # Build the request.
            synapse = bittensor.synapse.TextCausalLMNext()    
            grpc_request = bittensor.proto.TensorMessage (
                version = bittensor.__version_as_int__,
                hotkey = self.wallet.hotkey.ss58_address,
                tensors = [ synapse.serialize_forward_request_tensor ( inputs )],
                synapses = [ synapse.serialize_to_wire_proto() ],
                requires_grad = False,
            )

            
            # Fire off the future.
            futures.append( self.receptors[i].stub.Forward.future(
                request = grpc_request, 
                timeout = timeout,
                metadata = (
                    ('rpc-auth-header','Bittensor'),
                    ('bittensor-signature', self.receptors[i].sign() ),
                    ('bittensor-version',str(bittensor.__version_as_int__)),
                    ('request_type', str(bittensor.proto.RequestType.FORWARD)),
                )
            )
            )

            # Logging
            bittensor.logging.rpc_log ( 
                axon = False, 
                forward = True, 
                is_response = False, 
                code = bittensor.proto.ReturnCode.Success, 
                call_time = 0, 
                pubkey = self.receptors[i].endpoint.hotkey, 
                uid = self.receptors[i].endpoint.uid, 
                inputs = list(inputs.shape), 
                outputs = None, 
                message = 'Success',
                synapse = synapse.synapse_type
            )
            
        # We force the wait period. 
        time.sleep(timeout)
        
        # Iterate over the futures and check for success.
        for i,f in enumerate( futures ):
            try:
                if f.done():
                    fresult = f.result()
                    if fresult.return_code == 1:
                        # We return a True on success and skip deserialization.
                        response_tensor = synapse.deserialize_forward_response_proto ( inputs, fresult.tensors[0] )
                        result[i] = True

                        # Success logging.
                        bittensor.logging.rpc_log ( 
                            axon = False, 
                            forward = True, 
                            is_response = True, 
                            code = bittensor.proto.ReturnCode.Success, 
                            call_time = timeout, 
                            pubkey = self.receptors[i].endpoint.hotkey, 
                            uid = self.receptors[i].endpoint.uid, 
                            inputs = list(inputs.shape), 
                            outputs = list(response_tensor.shape), 
                            message = 'Success',
                            synapse = synapse.synapse_type
                        )

                    else:
                        # Timeout Logging.
                        bittensor.logging.rpc_log ( 
                            axon = False, 
                            forward = True, 
                            is_response = True, 
                            code = bittensor.proto.ReturnCode.Timeout, 
                            call_time = timeout, 
                            pubkey = self.receptors[i].endpoint.hotkey, 
                            uid = self.receptors[i].endpoint.uid, 
                            inputs = list(inputs.shape), 
                            outputs = None, 
                            message = 'Timeout',
                            synapse = synapse.synapse_type
                        )

            except Exception as e:
                # Unknown error logging.
                bittensor.logging.rpc_log ( 
                    axon = False, 
                    forward = True, 
                    is_response = True, 
                    code = bittensor.proto.ReturnCode.UnknownException, 
                    call_time = timeout, 
                    pubkey = self.receptors[i].endpoint.hotkey, 
                    uid = self.receptors[i].endpoint.uid, 
                    inputs = list(inputs.shape), 
                    outputs = None, 
                    message = str(e),
                    synapse = synapse.synapse_type
                )
            
        # Return the list of booleans.
        return result


    def sample(self, max_workers=10, n_steps=10, batch_size=64, sequence_len=20, n_queried=10, timeout=4):
        start_time = time.time()
        io_1 = psutil.net_io_counters()
        start_bytes_sent, start_bytes_recv = io_1.bytes_sent, io_1.bytes_recv


        inputs = torch.ones([batch_size, sequence_len], dtype=torch.int64)

        # Run each query concurrently then get results as completed.
        exp_results = []
        exp_futures = []

        def call_forward(**kwargs): 
            fn = kwargs.pop('fn')
            return fn(**kwargs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(n_steps):
                st.write(i)
                kwargs = dict(fn=self.forward, inputs=inputs, n_queried = n_queried, timeout=timeout)
                exp_futures.append(executor.submit(call_forward, **kwargs))
            for future in concurrent.futures.as_completed(exp_futures):
                exp_results.append( future.result() )   

        io_2 = psutil.net_io_counters()
        total_bytes_sent, total_bytes_recved = io_2.bytes_sent - start_bytes_sent, io_2.bytes_recv - start_bytes_recv
        end_time = time.time()


        ########################
        ##### Show results #####
        ########################
        def get_size(bytes):
            for unit in ['', 'K', 'M', 'G', 'T', 'P']:
                if bytes < 1024:
                    return f"{bytes:.2f}{unit}B"
                bytes /= 1024
                

        total_success = sum([sum(ri) for ri in exp_results])
        total_sent = n_queried * n_steps
        total_failed = total_sent - total_success

        total_seconds =  end_time - start_time
        print ('\nElapsed:', total_seconds, 's') 
        print ('\nSteps:', n_steps ) 
        print ('Step speed:', n_steps / (total_seconds), "/s" ) 

        print ('\nQueried:', n_queried )
        print ('Query speed:', total_sent / (total_seconds), "/s" ) 

    
        return   {"Success": total_success, "Failed": total_failed, 
                  "Elapsed": total_seconds, 
                  "Step speed": n_steps / (total_seconds),
                  "Success Speed": total_success/total_seconds,
                  "Query Speed": total_sent/total_seconds,
                  "Rate": total_success / (total_success + total_failed),
                  'Total Upload':  get_size( total_bytes_sent ), 
                  "Total Download": get_size( total_bytes_recved ),
                  "Upload Speed": get_size( total_bytes_sent/ total_seconds),
                  "Download Speed": get_size( total_bytes_recved/ total_seconds)
            }       


if __name__ == '__main__':
    # import streamlit as st
    # import time
    # import psutil
    # import tqdm
    # st.set_page_config(layout="wide")
    # module = SamplerModule()
    # # The tensor we are going to send over the wire

    # # module.forward(inputs=inputs)
    # # Module.ray_restart()
   
    # ##########################
    # ##### Run experiment #####
    # ##########################
    # # Measure state before.

    # st.write(module.sample( max_workers = 10,
    #                     n_steps= 10, 
    #                     batch_size= 10,
    #                     sequence_len = 20,
    #                     n_queried = 10,
    #                     timeout = 4))



    # # Measure state after.



    ##################
    ##### Import #####
    ##################
    import torch
    import concurrent.futures
    import time
    import psutil
    import random
    import argparse
    from tqdm import tqdm
    import bittensor


    ##########################
    ##### Get args ###########
    ##########################
    parser = argparse.ArgumentParser( 
        description=f"Bittensor Speed Test ",
        usage="python3 speed.py <command args>",
        add_help=True
    )
    parser.add_argument(
        '--max_workers', 
        dest='max_workers', 
        type=int,
        default=10,
        help='''Maximum concurrent workers on threadpool'''
    )
    parser.add_argument(
        '--timeout', 
        dest='timeout', 
        type=int,
        default=4,
        help='''Timeout for each set of queries (we always wait this long)'''
    )
    parser.add_argument(
        "--n_queried", 
        dest='n_queried', 
        type=int,
        default=10,
        help='''The number of endpoints we query each step.'''
    )
    parser.add_argument(
        "--n_steps", 
        dest='n_steps', 
        type=int,
        default=10,
        help='''The number of concurrent steps we run.'''
    )
    parser.add_argument(
        "--batch_size", 
        dest='batch_size', 
        type=int,
        default=10,
        help='''Input batch size'''
    )
    parser.add_argument(
        "--sequence_length", 
        dest='sequence_length', 
        type=int,
        default=20,
        help='''Input sequence length'''
    )
    bittensor.wallet.add_args(parser)
    bittensor.logging.add_args(parser)
    bittensor.subtensor.add_args(parser)
    config = bittensor.config(parser = parser)
    print (config)

    ##########################
    ##### Setup objects ######
    ##########################
    # Sync graph and load power wallet.
    bittensor.logging( config = config )
    subtensor = bittensor.subtensor( config = config )
    graph = bittensor.metagraph( subtensor = subtensor ).sync()
    wallet = bittensor.wallet( config = config )

    ################################
    ##### Experiment arguments #####
    ################################
    # A list of pre-instantiated endpoints with stub connections.
    receptors = [bittensor.receptor( wallet = wallet, endpoint = graph.endpoint_objs[i] ) for i in range(graph.n)]

    # Timeout for each set of queries (we always wait this long)
    timeout = config.timeout

    # The number of endpoints we query each step.
    n_queried = config.n_queried

    # The number of concurrent steps we run.
    n_steps = config.n_steps

    # The number of workers in the thread pool
    max_workers = config.max_workers

    # The tensor we are going to send over the wire
    inputs = torch.ones([config.batch_size, config.sequence_length], dtype=torch.int64)

    ############################
    ##### Forward Function #####
    ############################
    # Forward function queries (n_queried) random endpoints with the inputs
    # then waits timeout before checking for success from each query.
    # The function returns a list of booleans True or false depending on the query result.
    def forward():
        
        # Preable.
        futures = []
        result = [False for _ in range(n_queried)]
        
        # Randomly select (n_queried) peers to sample.
        for i in random.sample( range(4096), n_queried ):
            # Build the request.
            synapse = bittensor.synapse.TextCausalLMNext()    
            grpc_request = bittensor.proto.TensorMessage (
                version = bittensor.__version_as_int__,
                hotkey = wallet.hotkey.ss58_address,
                tensors = [ synapse.serialize_forward_request_tensor ( inputs )],
                synapses = [ synapse.serialize_to_wire_proto() ],
                requires_grad = False,
            )
            
            # Fire off the future.
            futures.append( receptors[i].stub.Forward.future(
                request = grpc_request, 
                timeout = timeout,
                metadata = (
                    ('rpc-auth-header','Bittensor'),
                    ('bittensor-signature', receptors[i].sign() ),
                    ('bittensor-version',str(bittensor.__version_as_int__)),
                    ('request_type', str(bittensor.proto.RequestType.FORWARD)),
                )
            )
            )

            # Logging
            bittensor.logging.rpc_log ( 
                axon = False, 
                forward = True, 
                is_response = False, 
                code = bittensor.proto.ReturnCode.Success, 
                call_time = 0, 
                pubkey = receptors[i].endpoint.hotkey, 
                uid = receptors[i].endpoint.uid, 
                inputs = list(inputs.shape), 
                outputs = None, 
                message = 'Success',
                synapse = synapse.synapse_type
            )
            
        # We force the wait period. 
        time.sleep(timeout)
        
        # Iterate over the futures and check for success.
        for i,f in enumerate( futures ):
            try:
                if f.done():
                    fresult = f.result()
                    if fresult.return_code == 1:
                        # We return a True on success and skip deserialization.
                        response_tensor = synapse.deserialize_forward_response_proto ( inputs, fresult.tensors[0] )
                        result[i] = True

                        # Success logging.
                        bittensor.logging.rpc_log ( 
                            axon = False, 
                            forward = True, 
                            is_response = True, 
                            code = bittensor.proto.ReturnCode.Success, 
                            call_time = timeout, 
                            pubkey = receptors[i].endpoint.hotkey, 
                            uid = receptors[i].endpoint.uid, 
                            inputs = list(inputs.shape), 
                            outputs = list(response_tensor.shape), 
                            message = 'Success',
                            synapse = synapse.synapse_type
                        )

                    else:
                        # Timeout Logging.
                        bittensor.logging.rpc_log ( 
                            axon = False, 
                            forward = True, 
                            is_response = True, 
                            code = bittensor.proto.ReturnCode.Timeout, 
                            call_time = timeout, 
                            pubkey = receptors[i].endpoint.hotkey, 
                            uid = receptors[i].endpoint.uid, 
                            inputs = list(inputs.shape), 
                            outputs = None, 
                            message = 'Timeout',
                            synapse = synapse.synapse_type
                        )

            except Exception as e:
                # Unknown error logging.
                bittensor.logging.rpc_log ( 
                    axon = False, 
                    forward = True, 
                    is_response = True, 
                    code = bittensor.proto.ReturnCode.UnknownException, 
                    call_time = timeout, 
                    pubkey = receptors[i].endpoint.hotkey, 
                    uid = receptors[i].endpoint.uid, 
                    inputs = list(inputs.shape), 
                    outputs = None, 
                    message = str(e),
                    synapse = synapse.synapse_type
                )
            
        # Return the list of booleans.
        return result


    ##########################
    ##### Run experiment #####
    ##########################
    # Measure state before.
    start_time = time.time()
    io_1 = psutil.net_io_counters()
    start_bytes_sent, start_bytes_recv = io_1.bytes_sent, io_1.bytes_recv


    # Run each query concurrently then get results as completed.
    exp_results = []
    exp_futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in tqdm(range(n_steps), desc='Submitting', leave=True):
            exp_futures.append(executor.submit(forward))
        for future in tqdm(concurrent.futures.as_completed(exp_futures), desc='Filling', leave=True):
            exp_results.append( future.result() )     
            
    # Measure state after.
    io_2 = psutil.net_io_counters()
    total_bytes_sent, total_bytes_recved = io_2.bytes_sent - start_bytes_sent, io_2.bytes_recv - start_bytes_recv
    end_time = time.time()

    ########################
    ##### Show results #####
    ########################
    def get_size(bytes):
        for unit in ['', 'K', 'M', 'G', 'T', 'P']:
            if bytes < 1024:
                return f"{bytes:.2f}{unit}B"
            bytes /= 1024
            

    total_success = sum([sum(ri) for ri in exp_results])
    total_sent = n_queried * n_steps
    total_failed = total_sent - total_success

    total_seconds =  end_time - start_time
    print ('\nElapsed:', total_seconds, 's') 
    print ('\nSteps:', n_steps ) 
    print ('Step speed:', n_steps / (total_seconds), "/s" ) 

    print ('\nQueried:', n_queried )
    print ('Query speed:', total_sent / (total_seconds), "/s" ) 

    print ('\nSuccess', total_success) 
    print ('Failed', total_failed ) 
    print ('Rate', total_success / (total_success + total_failed))

    print("\nTotal Upload:", get_size( total_bytes_sent ))
    print("Total Download:", get_size( total_bytes_recved ))
    print("\nUpload Speed:", get_size( total_bytes_sent/ total_seconds), "/s")
    print("Download Speed:", get_size( total_bytes_recved/ total_seconds), "/s")