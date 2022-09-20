
import streamlit as st
from random import shuffle, seed
from collections import defaultdict
import argparse
import bittensor
import torch
from torch import nn
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from commune.bittensor import BitModule


import torch
from torch import nn
from sentence_transformers import SentenceTransformer

class NeuronModule(BitModule):
    __file__ = __file__
    default_config_path = 'bittensor.neuron'
    def __init__(self, config=None, **kwargs):
        BitModule.__init__(self, config=config, **kwargs)
        self.load_state()

    @property
    def debug(self):
        return self.config.get('debug', False)

    def load_state(self):
        self.load_neuron()

    def load_neuron(self):
        self.sync(force_sync=False)
        st.write(self.wallet, self.graph, self.subtensor)
        self.neuron =  bittensor.neurons.core_server.neuron(subtensor=self.subtensor, 
                                                wallet=self.wallet,
                                                 axon=None, metagraph = self.graph)

    @classmethod
    def argparse(cls):
        parser = argparse.ArgumentParser(description='Gradio API and Functions')

        '''
        if --no-api is chosen
        '''
        parser.add_argument('--hotkey', type=str, default='default')
        parser.add_argument('--network', type=str, default='nobunaga')
        args =  parser.parse_args()
        override = {'wallet.hotkey': args.hotkey , 'network': args.network}
        return override



                                                
                        



if __name__ == '__main__':
    override = NeuronModule.argparse()
    st.write(override)
    module = NeuronModule.deploy(actor=False, override=override)
    # st.write(module.neuron)
    # module.register()

    module.neuron.run()


    # st.write(module.get_endpoints())
    # st.write(module.synapses)
    # module.run()