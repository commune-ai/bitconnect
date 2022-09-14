import streamlit as st
import fsspec
from fsspec.implementations.local import LocalFileSystem
from copy import deepcopy
import json
import os
from typing import *
import pandas as pd
import pickle
LocalFileSystem.root_market = '/'
class LocalModule(LocalFileSystem):
    default_cfg = {
    }
    def __init__(self, config=None):
        LocalFileSystem.__init__(self)
        self.config= self.resolve_config(config)
    
    def ensure_path(self, path):
        """
        ensures a dir_path exists, otherwise, it will create it 
        """
        file_extension = self.get_file_extension(path)
        if os.path.isfile(path):
            dir_path = os.path.dirname(path)
        elif os.path.isdir(path):
            dir_path = path
        elif len(file_extension)>0:
            dir_path = os.path.dirname(path)
        else:
            dir_path = os.path.dirname(path)

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def get_file_extension(path):
        return os.path.splitext(path)[1].replace('.', '')

    extension2mode = {
        'pkl':'pickle',
        'pickle':'pickle',
        'json': 'json',
        'csv': 'csv',
        'yaml': 'yaml',
        'pth': 'torch.state_dict',
        'onnx': 'onnx'
    }

    supported_modes = ['pickle', 'json']


    def resolve_mode_from_path(self, path):
        mode = self.extension2mode[self.get_file_extension(path)]
        assert mode in self.supported_modes
        return mode  

    def put_json(self, path, data):
            # Directly from dictionary
        self.ensure_path(path)
        if isinstance(data, dict):
            with open(path, 'w') as outfile:
                json.dump(data, outfile)
        
        elif isinstance(data, str):
            # Using a JSON string
            with self.open(path, 'w') as outfile:
                outfile.write(data)
        elif isinstance(data, pd.DataFrame):
            with open(path, 'w') as outfile:
                data.to_json(outfile)

    def get_json(self, path, handle_error = False):
        try:
            return json.loads(self.cat(path))
        except FileNotFoundError as e:
            if handle_error:
                return None
            else:
                raise e


    def put_pickle(self, path:str, data):
        with self.open(path,'wb') as f:
            pickle.dump(data, f, protocol= pickle.HIGHEST_PROTOCOL)
            
    def get_pickle(self, path, handle_error = False):
        try:
            with self.open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError as e:
            if handle_error:
                return None
            else:
                raise e

    def put_object(self, path:str, data:Any, mode:str=None,**kwargs):
        if mode == None:
            mode = self.resolve_mode_from_path(path)
        return getattr(self, f'put_{mode}')(path=path,data=data, **kwargs)

    def get_object(self, path:str, mode:str=None, **kwargs):
        if mode == None:
            mode = self.resolve_mode_from_path(path)
        return getattr(self, f'get_{mode}')(path=path, **kwargs)

    @staticmethod
    def funcs(module, return_dict=True):
        fn_list = dir(module)

        final_fns = []
        if return_dict:
            final_fns = {}

        for fn in fn_list:
            if not (fn.startswith('__') and fn.endswith('__')) and not fn.startswith('_'):
                
                fn_object = getattr(module, fn)
                if callable(fn_object):
                    if return_dict:
                        final_fns[fn] = fn_object
                    else:
                        final_fns.append(fn)

        return final_fns

    def resolve_config(self,config):
        if config == None:
            config = self.default_cfg
        else:
            assert isinstance(config, dict)
        
        return config

if __name__ == '__main__':
    # module = LocalModule()
    # st.write(module)
    # module.put_json(path='/tmp/commune/bro.json', data={'bro': 1})
    # module.put_pickle(path='/tmp/commune/bro.pkl', data={'bro': 1})

    # st.write(module.get_pickle(path='/tmp/commune/bro.pkl'))
    # st.write(module.get_json(path='/tmp/commune/bro.json'))
    # # st.write(module.glob('/tmp/commune/**'))
    pass
