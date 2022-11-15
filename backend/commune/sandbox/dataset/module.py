
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

import commune.sandbox.dataset.constant as constant

class DatasetTesting:


    def run_trial(self,  
                block_size=2000, 
                sequence_length=256,
                 batch_size=32, 
                 dataset_name = 'default', 
                 load_dataset=False, 
                 save_dataset=True, 
                 run_generator=False,  
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
    

    @classmethod
    def test_change_data_size(cls):
        data_sizes = [(10,1000), (15, 2000),(30, 3000), (25,4000)]
        dataset = bittensor.dataset(num_batches = constant.dataset.num_batches, dataset_name = constant.dataset.dataset_name, run_generator=False, no_tokenizer=False)
        for data_size in data_sizes:
            dataset.set_data_size(*data_size)
            sample_dict = next(dataset)
            for k,v in sample_dict.items():
                v.shape[0] == data_size[0]
            
        dataset = bittensor.dataset(num_batches = constant.dataset.num_batches, dataset_name = constant.dataset.dataset_name, run_generator=False, no_tokenizer=True)

        for data_size in data_sizes:
            raw_text_sample = next(dataset)
            len(raw_text_sample)  == data_size[1]
        
        dataset.close() 


    def run_experiment(self,
                        params=dict(
                            block_size= [1000, 5000, 10000, 20000],
                            sequence_length = [64, 128, 256, 512],
                            batch_size = [16,32, 64, 128],
                         ) ):
        pass

if __name__ == '__main__':


    # DatasetTesting.test_change_data_size()
    # st.write(DatasetTesting().run_trial())


    # st.write('FUCK')

    dataset = bittensor.dataset(num_batches=20, block_size=10000, sequence_length=64, batch_size=32, dataset_name = 'default', load_dataset=False, save_dataset=False, run_generator=False)
    
    for i in range(100):
        st.write(i)
        st.write({k:v.shape for k,v in next(dataset).items()})


