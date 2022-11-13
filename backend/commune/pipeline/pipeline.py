
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
        self.pipeline_blocks = []
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
            process_block['module'] = module_block
            process_block['function'] = getattr(module_block, process_block['fn'])

            self.process_block[process_block['name']] = process_block

            if previous_key != None:
                input_modules = self.pipeline_blocks[previous_key]
                if not isinstance(input_modules, list):
                    input_modules = [input_modules]
                process_block['input_modules'] = list(map(lambda x: x['name'], input_modules ))

            previous_key = key
            self.pipeline_blocks.append(process_block)
            

    def run(self):
        input = {}
        for block in self.pipeline_blocks:
            fn = block.get('function')
            fn_args = block.get('args', [])
            fn_kwargs = block.get('kwargs', {})
            key_map = block.get('key_map', {})
            input = {key_map.get(k, k):v for k,v in input.items()}
            fn_kwargs = {**input, **fn_kwargs}
            output = fn(*fn_args, **fn_kwargs)

            st.write(output)
            input = output

    @staticmethod
    def test_sequential_pipeline():
        commune.init_ray()
        pipeline_blocks = [{
        'module': 'dataset.text.huggingface',
        'actor': {'cpus': 1, 'refresh': False},
        'fn': 'sample',
        'kwargs': {'tokenize': False},
         }, {
            'module': 'model.transformer',
            'actor': {'gpus': 0.1},
            'fn': 'forward',
            'key_map': {'text': 'input'},
            'kwargs': {},
        }]

        pipeline = Pipeline(pipeline_blocks)
        st.write(pipeline)
        st.write(pipeline.run())

if __name__ == '__main__':

    Pipeline.test_sequential_pipeline()



    
    # st.write(commune.list_actor_names())



        
        

