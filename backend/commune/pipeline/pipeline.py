
from munch import Munch
import commune
import streamlit as st
import os

# pipeline_config = commune.load_config(os.path.dirname(__file__).replace(os.getenv('PWD'), ''))



dataset_block = {
    'module': 'dataset.text.huggingface',
    'fn': 'sample',
    'kwargs': {},
    'args': []
}

model_block = {
    'module': 'model.transformer',
    'fn': 'forward',
    'kwargs': {},
    'args': []
}


pipeline_blocks = [dataset_block, model_block ]
config = {'pipeline': pipeline_blocks}


class Pipeline:
    def __init__(self, pipeline=pipeline_blocks, config=config):
        self.config = Munch(config)
        self.pipeline = pipeline if pipeline != None else self.config.pipeline
        self.build_pipeline(self.pipeline)
    def build_pipeline(self, pipeline_config):
        if isinstance(pipeline_config, list):
            keys = list(range(len(pipeline_config)))
        elif isinstance(pipeline_config, dict): 
            keys = list(pipeline_config.keys())
        
        previous_key = None
        # building the pipeline
        for key in keys:
            process_block = pipeline_config[key]


            path = process_block['module']
            process_block['replica'] = process_block.get('replica', 0)
            block_name = path if process_block['replica'] == 0 else path + f".{process_block['replica']}"
            process_block['name'] = process_block.get('name',  path )
            process_block['actor'] = process_block.get('actor',  False )
            module_class = commune.load_module(process_block['name'])
            module_fn = getattr(module_class, process_block['fn'])
            if previous_key != None:
                input_modules = pipeline_blocks[previous_key]
                if not isinstance(input_modules, list):
                    input_modules = [input_modules]
                process_block['input_modules'] = list(map(lambda x: x['name'], input_modules ))

            previous_key = key

            st.write(process_block)



        
        

