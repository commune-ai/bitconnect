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
import streamlit as st
import numpy as np

##########################
##### Get args ###########
##########################
parser = argparse.ArgumentParser( 
    description=f"Bittensor Speed Test ",
    usage="python3 speed.py <command args>",
    add_help=True
)
bittensor.wallet.add_args(parser)
bittensor.logging.add_args(parser)
bittensor.subtensor.add_args(parser)
config = bittensor.config(parser = parser)
config.wallet.name = 'const'
config.wallet.hotkey = 'Tiberius'
##########################
##### Setup objects ######
##########################
# Sync graph and load power wallet.


from commune import Module
class Sandbox(Module):
    def __init__(self, config=None):
        Module.__init__(self, config=config)

        # config = bittensor.config()
        # st.write(config)
        self.subtensor = bittensor.subtensor( config = config )
        self.graph = bittensor.metagraph( subtensor = self.subtensor )
        self.graph.load()
        st.write(self.sync_delay)
        self.tokenizer = bittensor.tokenizer()
        if self.sync_delay > self.config.get('delay_threshold', 100):
            self.graph.sync()
            self.graph.save()
        
        self.wallet = bittensor.wallet(**self.config.get('wallet'))
        self.receptor_pool = bittensor.receptor_pool(wallet=self.wallet)
    @property
    def current_block(self):
        return self.subtensor.block
    
    @property
    def synced_block(self): 
        return self.graph.block.item()

    @property
    def sync_delay(self):
        return self.current_block - self.synced_block
    

    def get_receptors(self, n = 10,uids=None):
        if uids == None:
            uids = list(range(n))
        
        receptors = []
        for uid in uids:
            receptors += [bittensor.receptor( wallet = self.wallet, endpoint = self.graph.endpoint_objs[uid])]
        return receptors
    

    def get_random_endpoints(self, n = 10 ):
        endpoints =self.graph.endpoint_objs
        random_ids =  list(np.random.randint(0, len(endpoints), (n)))
        return [endpoints[i] for i in random_ids]

    def get_endpoints(self, n=10, uids:list=[]):

        if len(uids) == 0:
            uids = list(range(n))
        endpoints =self.graph.endpoint_objs
        selected_endpoints = []
        for uid in uids:
            selected_endpoints += [endpoints[uid]]

        return selected_endpoints


    def tokenize(self, text:str, dtype = torch.int64, device='cpu'):
        # must be a string, or a list of strings
        if isinstance(text, str):
            text = [text]
        assert all(isinstance(t, str) for t in text)
        token_ids =  self.tokenizer(text)['input_ids']
        token_ids = torch.Tensor(token_ids).type(dtype).to(device)
        return token_ids

    @staticmethod
    def str2synapse(synapse:str, *args, **kwargs):
        return getattr(bittensor.synapse, synapse)(*args, **kwargs)
    @property
    def available_synapses(self):
        return [f for f in dir(bittensor.synapse) if f.startswith('Text')]


    def streamlit(self,
            sequence_length = 10,
            batch_size = 2,
            timeout= 1,
            synapse = 'TextLastHiddenState',
            num_endpoints = 10
        ):
        ################################
        ##### Experiment arguments #####
        ################################
        # A list of pre-instantiated endpoints with stub connections.
        # receptors = self.receptor_([0,2,4,5])

        input_text = [' '.join(['ola']*sequence_length) for i in range(batch_size)][0]
        inputs = self.tokenize(input_text)
        synapses = [bittensor.synapse.TextCausalLMNext() ]
        endpoints = self.get_random_endpoints(num_endpoints)
        results = self.receptor_pool.forward(
                            endpoints=endpoints,
                            synapses= synapses,
                            timeout=timeout,
                            inputs= [inputs]*len(endpoints),
                            return_type='tuple',
                            graph=self.graph
                        )
            
        st.write(results)
        

        # st.write(list(map(lambda x: bittensor.utils.codes.code_to_string(x[0]), results[1])))



# Sandbox.speed_test()
Sandbox().streamlit()

