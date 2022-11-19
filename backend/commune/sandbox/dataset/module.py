
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
                block_size=1000, 
                sequence_length=256,
                max_blocks_per_dataset=10e9,
                 batch_size=32, 
                 dataset_name = 'default', 
                 load_dataset=False, 
                 save_dataset=True, 
                 cache_size: int = 50, 
                 num_samples=1000,
                 cache_calls_per_update=10000,
                 ):
        dataset = bittensor.dataset( block_size=block_size, sequence_length=sequence_length, 
                                batch_size=batch_size, dataset_name = dataset_name,
                                max_blocks_per_dataset=max_blocks_per_dataset,
                                cache_calls_per_update=cache_calls_per_update,
                                 load_dataset=load_dataset, save_dataset=save_dataset,
                                  cache_size = cache_size, 
                                    )
        
        next(dataset)
        with Module.timer() as t:
            for i in range(num_samples):
                # st.write(Module.get_memory_info())
                # st.write('dataset size: ',total_size(dataset.__dict__))
                # st.write('Write')
                next(dataset)
                st.write('', i / t.elapsed_time.total_seconds())
    

    @classmethod
    def test_change_data_size(cls):
        data_sizes = [(10,1000), (15, 2000),(30, 3000), (25,4000)]
        dataset = bittensor.dataset(num_batches = constant.dataset.num_batches, dataset_name = constant.dataset.dataset_name, no_tokenizer=False)
        for data_size in data_sizes:
            dataset.set_data_size(*data_size)
            sample_dict = next(dataset)
            for k,v in sample_dict.items():
                v.shape[0] == data_size[0]
            
        dataset = bittensor.dataset(num_batches = constant.dataset.num_batches, dataset_name = constant.dataset.dataset_name, no_tokenizer=True)

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




    @staticmethod
    def test_next_tokenized_sample():
        batch_size = 10
        sequence_length = 128
        block_size = 1000
        num_batches = 10


        dataset = bittensor.dataset (
            block_size = block_size,
            batch_size = batch_size,
            sequence_length = sequence_length,
            num_batches=num_batches,
            no_tokenizer=False
        )

        for i in range(num_batches):
            input = next(dataset)
            assert input['input_ids'].shape[0] == input['attention_mask'].shape[0] == batch_size
            assert input['input_ids'].shape[1] == input['attention_mask'].shape[1] == sequence_length
            dataset.close()


    @staticmethod
    def test_next_raw_sample():
        batch_size = 10
        sequence_length = 128
        block_size = 1000
        num_batches = 10
        dataset = bittensor.dataset (
            block_size = block_size,
            batch_size = batch_size,
            sequence_length = sequence_length,
            num_batches=num_batches,
            no_tokenizer = True
        )

        input = next(dataset)
        assert len(input) == batch_size
        for i in range(len(input)):
            assert len(input[i].split()) == sequence_length

        dataset.close()



    

def test_change_data_size():
    # (batch_size, block_size, buffer_size)
    data_sizes = [(10,1000, 100), (15, 2000, 1000), (25,3000, 4000)]
    dataset = bittensor.dataset(dataset_name = ['ArXiv'], no_tokenizer=False)

    for data_size in data_sizes:
        st.write(data_size)
        dataset.set_data_size(*data_size)
        sample = next(dataset)
        assert sample.shape[0] == data_size[0]
        assert dataset.block_size == data_size[1]
        assert dataset.buffer_size == data_size[2]
        
if __name__ == '__main__':


    Module.new_event_loop()
    # DatasetTesting.test_change_data_size()
    # st.write(DatasetTesting().run_trial())
    test_change_data_size()

    dataset = bittensor.dataset(block_size=4000,no_tokenizer=True, 
                             buffer_size=2000, sequence_length=256, batch_size=32, buffer_calls_per_update=100,
                             dataset_name = ['ArXiv'], load_dataset=False, save_dataset=True)
    
    with Module.timer() as t:
        cnt = 0
        previous_seconds =  0
        for i in range(10000):
            cnt += 1
            if cnt % 100 == 0:
                seconds = t.elapsed_time.total_seconds() - previous_seconds
                st.write(seconds)
                st.write(cnt/ seconds)

                cnt = 0
                previous_seconds = t.elapsed_time.total_seconds() 

            raw_text_sample = next(dataset)
            
    # st.write(DatasetTesting.test_next_raw_sample())
