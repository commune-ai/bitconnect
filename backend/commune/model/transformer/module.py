# make sure you're logged in with `huggingface-cli login`


import os
import sys
from copy import deepcopy
import streamlit as st
sys.path.append(os.environ['PWD'])
from commune.utils import dict_put, get_object, dict_has
from commune import Module
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import torch
import os
import io
import glob
import numpy as np
import uuid
import pandas as pd
from PIL import Image
import torch
import ray
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, LMSDiscreteScheduler


class TransformerModel(Module):

    def __init__(self, config=None,  **kwargs):
        Module.__init__(self, config=config, **kwargs)

        self.load_model()

    @property
    def hf_token(self):
        return self.config['hf_token']


    def load_model(self):
        self.model = self.launch(**self.config['model'])
        self.tokenizer = self.launch(**self.config['tokenizer'])

        self.model.to(self.device)

    @property
    def device(self):
        return self.config.get('device', 'cuda')

    def predict(self, input:str="This is the first sentence. This is the second sentence."):
        input_ids = self.tokenizer(
                input, add_special_tokens=False, return_tensors="pt"
            ).input_ids
        
        outputs =  self.model.generate(input_ids.to(self.device))

        return self.tokenizer.decode(outputs[0])

    @classmethod
    def streamlit(self):
        dataset = Module.launch('dataset.huggingface', actor=True, wrap=True)
        # st.write(dataset.sample())
        model = TransformerModel.deploy(actor={'refresh': False, 'resources': {'num_gpus': 0.2, 'num_cpus':2}, }, wrap=True)
        st.write(model.tokenizer.decode(model.predict()[0]))
        

if __name__ == '__main__':
    TransformerModel.run()


