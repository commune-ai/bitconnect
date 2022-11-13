
from munch import Munch
import commune
import streamlit as st
import os
import ray

# pipeline_config = commune.load_config(os.path.dirname(__file__).replace(os.getenv('PWD'), ''))
import torch 


class BaseAggregator:
    def __init__(self, blocks, config={}):
        self.config = Munch(config)

        self.blocks = blocks if blocks != None else self.config.blocks
        self.build(self.blocks)

    def build(self,blocks):
        if isinstance(blocks, list):
            block_keys = list(range(len(blocks)))
        elif isinstance(blocks, dict): 
            block_keys = list(blocks.keys())
        
        previous_key = None
        # building the pipeline
        self.blocks = []
        for block_key in block_keys:
            process_block = blocks[block_key]
            path = process_block['module']
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
            previous_key = block_key
            self.blocks.append(process_block)


    @staticmethod
    def run_block(block, input={}):
        fn = block.get('function')
        fn_args = block.get('args', [])
        fn_kwargs = block.get('kwargs', {})
        key_map = block.get('key_map', {})
        input = {key_map.get(k, k):v for k,v in input.items()}
        fn_kwargs = {**input, **fn_kwargs}
        output = fn(*fn_args, **fn_kwargs)      
        return output
        
    def get_outputs(self, *args,**kwargs):
        outputs = []
        for block in self.blocks:
            output = self.run_block(block)
            outputs.append(output)
        return outputs

    @staticmethod
    def aggregate_outputs(outputs):

        if any([isinstance(o, ray._raylet.ObjectRef) for o in outputs]):
            if all([isinstance(o, ray._raylet.ObjectRef) for o in outputs]):
                outputs = ray.get(outputs)
            else:
                outputs = [ray.get(o) if isinstance(o, ray._raylet.ObjectRef) else o  for o in outputs ]
        
        aggregate_outputs = {}

        for output in outputs:
            for k,v in output.items():
                if k in aggregate_outputs:
                    aggregate_outputs[k] += [v]
                else:
                    aggregate_outputs[k] = [v]
        return aggregate_outputs

    def run(self, *args, **kwargs):
        outputs = self.get_outputs(*args , **kwargs)
        aggregate_outputs = self.aggregate_outputs(outputs)
        
        # stack outputs in 1st dimension
        outputs = {k:torch.stack(v) for k,v in aggregate_outputs.items()}
        outputs = {k: torch.sum(v, dim=0) for k,v in outputs.items()}
        return outputs

    @staticmethod
    def test_sequential_pipeline():
        commune.init_ray()
        blocks = [
 
         {
            'module': 'model.transformer',
            'actor': {'gpus': 0.1},
            'fn': 'forward',
            'kwargs': {'ray_get': False},
        }, 
         {
            'module': 'model.transformer',
            'actor': {'gpus': 0.1},
            'fn': 'forward',
            'kwargs': {'ray_get': False},
        },
        {
            'module': 'model.transformer',
            'actor': {'gpus': 0.1},
            'fn': 'forward',
            'kwargs': {'ray_get': False},
        }
        ]



        aggregator = BaseAggregator(blocks)
        st.write(aggregator.run())

if __name__ == '__main__':

    BaseAggregator.test_sequential_pipeline()
    # st.write(commune.list_actor_names())



        
        

