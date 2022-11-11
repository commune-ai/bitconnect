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

    def __init__(self, config=None, model=None, tokenizer=None,  **kwargs):
        Module.__init__(self, config=config, **kwargs)

        self.load_model(model)
        self.load_tokenizer(tokenizer)

    @property
    def hf_token(self):
        return self.config['hf_token']

    def load_tokenizer(self, tokenizer=None):
        self.tokenizer = tokenizer if tokenizer else self.launch(**self.config['tokenizer'])

    def load_model(self, model=None):
        self.model =  model if model else self.launch(**self.config['model'])
        self.model.to(self.device)

    @property
    def device(self):
        return self.config.get('device', 'cuda')

    def predict(self, input:str="This is the first sentence. This is the second sentence.", tokenize=False):
        
        if tokenize:
            input = self.tokenizer(
                    input, add_special_tokens=False, return_tensors="pt"
                ).input_ids
        return self.model.generate(input)



    @classmethod
    def streamlit(self):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        Module.init_ray()
        dataset = Module.launch('dataset.huggingface', actor=False, wrap=True)
        # st.write(dataset.sample())
        # model = Module.launch('commune.model.transformer')
        model = TransformerModel.deploy(actor=False, wrap=True)
        st.write(model.device)
 
        x = torch.tensor(dataset.sample().to_list()).to('cuda')
        st.write(model.predict(x)[0])
        

if __name__ == '__main__':
    TransformerModel.run()


