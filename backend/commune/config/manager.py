
import os, sys
sys.path.append(os.environ['PWD'])


import ray
from copy import deepcopy
from commune.process import BaseProcess
from commune.utils.misc import dict_fn, dict_get, dict_has, dict_put
import streamlit as st
from commune.config import ConfigLoader
from commune.process.launcher import Launcher

class ConfigManager(BaseProcess):

    default_cfg_path =  "config.manager"
    def __init__(self, cfg=None):
        BaseProcess.__init__(self, cfg=None)
        self.database = self.cfg.get('database', 'commune')
        self.collection = self.cfg.get('collection', 'config')


    # def process(self, module, tags={}):
    #     """
    #     kwargs = {
    #         'module': query
    #     }

    #     """
    #     self.object_dict['config'] =  self.find_modules(module=module, tags=tags)


    def load_template(self, path):
        '''
        loads template from local file using the config loader
        '''
        return self.config_loader.load(path=path, local_var_dict={}, override={}) 
    
    def save(data:dict,tag:str=None):
        
        if tag:
            data['tag'] = tag

        return self.client['mongo'].write(data=data, 
                                         collection=self.collection, database=self.database)


    def delete(module:str ,tag:str=None, query:dict={}):
        if module:
             query['module'] = module
        if tag:
            query['tag'] = tag

        return self.client['mongo'].delete(query=query, 
                                         collection=self.collection, database=self.database)


    def ls_templates(self, *args,**kwargs):
        return self.module_tree( *args,**kwargs)

    @staticmethod     
    def module_tree(root:str='/app/commune', tree:bool=True, linear:bool=True):


        out_dict = {}

        for local_root, dirs, files in os.walk(root, topdown=False):
            for name in files:
                file_path = os.path.join(local_root, name)
                if '.' in file_path and 'yaml' == file_path.split('.')[-1]:
                    config_path = file_path
                    module_path = config_path.replace('.yaml', '.py')
                    module_key_path = os.path.dirname(module_path).replace(root, '').lstrip('/').replace('/', '.')
                    
                    if linear and tree:
                        dict_put(out_dict,
                                    keys=module_key_path,
                                    value= {'config': config_path, 'module': module_path})
                    else:
                        out_dict[module_key_path] = {'config': config_path, 'module': module_path}
                   
        if tree:
            return out_dict
        else:
            return list(out_dict.values())

    @staticmethod
    def config2module(cfg_path):
        return BaseProcess.get_object(BaseProcess.load_config(cfg_path, recursive=False)['module'])

    def ls(self, module=None, tag=None, query={}, select=[]):
        documents = self.find(module=module, tag=tag, query=query, select=['module',*select])
        return documents

    def find(self, module=None, tag=None, query={}, select=[]):
        if module:
            query['module'] = module
        if tag:
            data['tag'] = tag

        cfg = self.client['mongo'].load(collection=self.collection,
                                        database=self.database, 
                                        query=query, 
                                        projection= {s:1 for s in select} if select else None,
                                       return_one=False, remove_id=True)
        # cfg = dict_fn(cfg,self.resolve_pipeline_cfg)
        return cfg 
            

if __name__ == "__main__":
    import plotly.graph_objects as go
    from commune.utils.misc import dict_fn
    import json
    import streamlit as st

    cm = ConfigManager.deploy(actor=False)
    st.write(cm.ls_templates())
    # cfg = process.run(module= 'data.regression.crypto.sushiswap.dataset')

        

'''

## DEPRACTED LAND ##


def resolve_pipeline_cfg(self, cfg):
    
    if dict_has(cfg, 'dag'):
        for process_key, process_cfg_template in cfg['dag'].items():

            if 'template' in process_cfg_template:
                continue
            

            process_cfg_template['write']['cfg']['params']['query'] = {'module': process_cfg_template['write']['cfg']['params']['query']['module']}
            process_cfg_template['write']['cfg']['params']['remove_id']=True
            process_cfg_list = self.client['mongo'].load(**process_cfg_template['write']['cfg']['params'])
            process_cfg_list = list(map(self.resolve_explan_ipfs, process_cfg_list))
            
            cfg['dag'][process_key] =  {
                'template':process_cfg_template,
                'clone': process_cfg_list
            }




def resolve_explan_ipfs(self, cfg):
    if dict_has(cfg, 'explain.write.explain.params'):
        ipfs_params = deepcopy(dict_get(cfg, 'explain.write.explain.params'))
        ipfs_params['return_hash'] =True
        cfg['explain']['hash'] = self.client['ipfs'].load(**ipfs_params)
    return  cfg

'''