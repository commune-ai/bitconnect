
from munch import Munch
import commune
import streamlit as st
import os

# pipeline_config = commune.load_config(os.path.dirname(__file__).replace(os.getenv('PWD'), ''))




class Pipeline:
    def __init__(self, pipeline, config={}):
        self.config = Munch(config)
        self.process_block = Munch({})
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

            launch_kwargs = dict(
                module = process_block['module'],
                fn = process_block.get('init_fn', None),
                actor =  process_block['actor']
            )
            module_block = commune.launch(**launch_kwargs)
            self.process_block[process_block['name']] = module_block
            module_class = commune.load_module(process_block['name'])
            module_fn = getattr(module_class, process_block['fn'])
            if previous_key != None:
                input_modules = pipeline_blocks[previous_key]
                if not isinstance(input_modules, list):
                    input_modules = [input_modules]
                process_block['input_modules'] = list(map(lambda x: x['name'], input_modules ))

            previous_key = key



if __name__ == '__main__':


    dataset_block = {
        'module': 'dataset.text.huggingface',
        'actor': {'cpus': 1, 'refresh': False},
        'fn': 'sample',
        'kwargs': {},
        'args': []
    }

    model_block = {
        'module': 'model.transformer',
        'actor': {'gpus': 0.1, 'refresh': False},
        'fn': 'forward',
        'kwargs': {},
        'args': []
    }

    import ray
    commune.ray_init()

    # st.write(commune.list_actors())
    pipeline_blocks = [dataset_block, model_block ]

    pipeline = Pipeline(pipeline_blocks)
    st.write(pipeline.process_block)



        
        

