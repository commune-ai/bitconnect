
from __future__ import print_function
##################
##### Import #####
##################
import torch
import concurrent.futures
import time
import psutil
import sys
import random
import argparse
from tqdm import tqdm
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())
import bittensor
import glob
import queue
import streamlit as st
import numpy as np
import aiohttp
import json
import os
from fsspec.asyn import AsyncFileSystem, sync, sync_wrapper
from commune import Module
##########################
##### Get args ###########
##########################
from typing import *
from munch import Munch

from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

import commune.sandbox.dataset.constant as constant

class DatasetTesting:


    def run_trial(self,  
                block_size=2000, 
                sequence_length=256,
                 batch_size=32, 
                 dataset_name = 'default', 
                 load_dataset=True, 
                 save_dataset=False, 
                 run_generator=True,  
                 num_samples=100,
                 cache_size: int = 10, 
                 cache_calls_per_block: int=100
                 ):
        dataset = bittensor.dataset( block_size=block_size, sequence_length=sequence_length, 
                                batch_size=batch_size, dataset_name = dataset_name,
                                 load_dataset=load_dataset, save_dataset=save_dataset,
                                 run_generator=run_generator, cache_size = cache_size, 
                                 cache_calls_per_block=cache_calls_per_block
                                    )
        
        with Module.timer() as t:
            for i in range(num_samples):
                # st.write(Module.get_memory_info())
                # st.write('dataset size: ',total_size(dataset.__dict__))
                # st.write('Write')
                st.write(list(next(dataset).values())[0].shape)
                st.write('', i / t.elapsed_time.total_seconds())
    

    def run_experiment(self,
                        params=dict(
                            block_size= [1000, 5000, 10000, 20000],
                            sequence_length = [64, 128, 256, 512],
                            batch_size = [16,32, 64, 128],
                         ) ):
        pass

if __name__ == '__main__':

    wallet = bittensor.wallet(hotkey='default', name='default')
    
    wallet.create(coldkey_use_password=False)
    # DatasetTesting.test_change_data_size()
    st.write(DatasetTesting().run_trial())



    # dataset = bittensor.dataset(num_batches=20, block_size=10000, sequence_length=256, batch_size=64, dataset_name = 'default', load_dataset=True, save_dataset=False, run_generator=False)
    
    # for i in range(10):
    #     st.write({k:v.shape for k,v in next(dataset).items()})


