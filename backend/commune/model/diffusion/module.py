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
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, LMSDiscreteScheduler


class DiffuserModule(Module):

    default_config_path = 'model.diffusion'
    default_mode = 'txt2img'
    default_device = 'cuda'

    def __init__(self, config=None,  **kwargs):
        Module.__init__(self, config=config, **kwargs)

        self.mode = kwargs.get('mode', self.config.get('mode', self.default_mode))
        self.load_pipeline(**self.config['pipeline'])

    @property
    def hf_token(self):
        return self.config['hf_token']
    
    @property
    def device(self): 
        device = self.config.get('device',self.default_device)
        if device == 'cuda' and not torch.cuda.is_available():
            device = self.config['device'] = 'cpu'
        return device

    @device.setter
    def device(self, device):
        if device == 'cuda':
            assert torch.cuda.is_available(), 'Cuda is not available bruh'
        self.config['device'] = device
        return device


    def resolve_scheduler(self, scheduler):
        if scheduler == None:
            if not hasattr(self, 'scheduler'):
                self.scheduler = self.load_scheduler()
            scheduler = self.scheduler
        elif isinstance(scheduler, dict):
            scheduler = self.load_scheduler(**scheduler)
        else:
            raise NotImplementedError

        return scheduler

    def resolve_device(self, device):
        if device == None:
            device = self.device
        else:
            self.device = device

        return device



    def load_scheduler(self, path=None, **params):
        default_scheduler = LMSDiscreteScheduler
        default_params = dict(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        if path == None:
            scheduler_class = LMSDiscreteScheduler
        else:
            scheduler_class = self.import_object(path)
        if len(params) == 0:
            params = default_params 
        self.scheduler = scheduler_class(**params)
        return self.scheduler

    def load_pipeline(self, 
                      path, 
                      scheduler=None, 
                      mode=default_mode, 
                      device=None, 
                      enable_attention_slicing=False,
                      memory_efficient_attention = True,
                      **params):
        # check if there is already the pipe in pipes
        # if yes and the selected model is same return
        # if model different - set up and add to pipes and return pipe


        default_params = dict(revision="fp16", torch_dtype=torch.float16)
        if len(params) == 0:
            params = default_params


        if mode == 'img2img':
            pipeline_class =  StableDiffusionImg2ImgPipeline
        elif mode == 'txt2img':
            pipeline_class = StableDiffusionPipeline
        else:
            raise NotImplemented 

        
            

        params['scheduler'] = self.resolve_scheduler(scheduler)
        params['use_auth_token'] = params.get('use_auth_token', self.hf_token)

        self.pipeline = pipeline_class.from_pretrained(path, **params).to(self.device)

        if enable_attention_slicing:
            self.pipeline.enable_attention_slicing()

        # if memory_efficient_attention:
        #     self.pipeline.enable_xformers_memory_efficient_attention()
        
        # self.pipeline.safety_checker = lambda images, clip_input: (images, False)
        
        return self.pipeline


    def predict(self, enable_attention_slicing=False, memory_efficient_attention=True, mode='txt2img', *args, **kwargs):
        if enable_attention_slicing:
            self.pipeline.enable_attention_slicing()
        else:
            self.pipeline.disable_attention_slicing()

        # if memory_efficient_attention:
        #     self.pipeline.enable_xformers_memory_efficient_attention()
        # else:
        #     self.pipeline.disable_xformers_memory_efficient_attention()

        return getattr(self, f'predict_{mode}', *args, **kwargs)

    def predict_txt2img(self, 
        prompt:str, 
        num_samples=1, 
        height=1024, 
        width=1024, 
        inf_steps=10, 
        guidance_scale=7.5, 
        seed=69, 
        strength=0.9, 
        save_path=None):


        # with torch.cuda.amp.autocast():
            
        images = self.pipeline([prompt] * num_samples, 
                    num_inference_steps=inf_steps, 
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    seed=seed).images

        if save_path:
            for i, image in enumerate(images):
                base_save_path, ext = os.path.splittext(save_path)
                image.save(f'{base_save_path}_{i}{ext}')

                
        return images

    def predict_img2img(self, 
        prompt,
        init_image,
        num_samples=1, 
        height=512, 
        width=512, 
        inf_steps=50, 
        guidance_scale=7.5, 
        seed=69, 
        strength=0.6, save_path=None):

        with torch.cuda.amp.autocast():
            images = self.pipeline([prompt] * num_samples, 
                        init_image=init_image,
                        strength=strength,
                        num_inference_steps=inf_steps, 
                        guidance_scale=guidance_scale,
                        seed=seed).images

        if save_path:
            for i, image in enumerate(images):
                base_save_path, ext = os.path.splittext(save_path)
                image.save(f'{base_save_path}_{i}{ext}')

        return images

    def image_to_np(self, image: Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        return byte_im

    @staticmethod
    def st_demo():
        module = DiffuserModule.deploy(actor={'refresh': False, 'resources': {'num_cpus': 2, 'num_gpus': 0.6}}, wrap=True)


        with st.form('Prompt'):
            # text = st.input_text('Input Text', 'd')
            cols = st.columns([2,3])
            with cols[0]:
                text = st.text_area('Input Prompt','a malibu beach house with a ferarri in the driveway' )
                steps = st.slider('Number of Steps', 0, 200, 20)
                dims = st.select_slider('height', list(range(256, 4096, 128)), )
            
            with cols[1]:
                submitted = st.form_submit_button("Sync")

                if submitted:
                    img = module.image_to_np(module.predict_txt2img(text,inf_steps=steps,  height=dims, width=dims)[0])
                else:
                    img = np.zeros([dims,dims])
                
                st.image(img)
    
    
if __name__ == '__main__':
    import ray
    DiffuserModule.st_demo()
    # module = DiffuserModule.deploy(actor={'refresh': False, 'resources': {'num_cpus': 2, 'num_gpus': 0.6}}, wrap=True)
    # module = DiffuserModule.deploy(actor={'refresh': False, 
    #                                       'resources': {'num_gpus': 0.5, 'num_cpus': 2}}, wrap=True)
    
    # st.write(module.list_actors(detail=True))
    # st.write(st.write(ray.get_actor('model.diffusion').getattr.remote('config')))
    # module.getattr('')
    # st.write(module.getattr('config'))
    # st.write(module.pipeline.to('cuda'))
    # st.write('fam')
    # st.write('fram')
    # st.write(module.kill_actor('model.diffusion.2'))
    # st.write(ray.get_actor('model.diffusion'))
    # # st.write(module.kill_actor('model.diffusion.2'))
    # st.write(module.list_actors())
    # # st.write(module.pipeline)
    # ray.kill(ray.get_actor('actor'))

    # st.write(module.forward('whadup fam, what are you sayin'))
    # DiffuserModule.ray_restart()

    # from ray.experimental.state.api import list_actors
    # # ray.kill(ray.get_actor('actor'))
    # st.write(list_actors(filters=[("state", "=", "ALIVE")]))

