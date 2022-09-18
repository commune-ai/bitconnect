


import os, sys
sys.path.append(os.environ['PWD'])
import gradio
from commune import BaseModule
from commune.utils import *
import shlex
import subprocess

class SubprocessModule(BaseModule):
    subprocess_map = {}
    default_config_path =  'subprocess.module'
    def __init__(self, config=None, **kwargs):
        BaseModule.__init__(self, config=config)
        if self.config.get('refresh') == True:
            self.subprocess_map = {}
            self.save_state()
        self.load_state()

        self.subprocess_map = {p:v for p,v in self.subprocess_map.items() if self.check_pid(v['pid']) }




    def __reduce__(self):
        deserializer = self.__class__
        serialized_data = (self.config,)
        return deserializer, serialized_data

    def submit(command):
        return self.run_command(command)
    

    def rm_subprocess(self, key, load=True, save=True):

        if load:
            self.load_state()
        subprocess_dict = self.subprocess_map[key]
        pid = subprocess_dict['pid']
        try:
            self.kill_pid(pid)
        except ProcessLookupError:
            pass
        del self.subprocess_map[key]
        if save:
            self.save_state()
        return pid

    rm = rm_subprocess

    def rm_all(self):
        self.load_state()
        rm_dict = {}
        for k in self.list_keys():
            rm_dict[k] = self.rm(key=k, load=False, save=False)

        self.save_state()
        return rm_dict

    def add_subprocess(self, command:str,key=None, cache=True):

        process = subprocess.Popen(shlex.split(command))
        process_state_dict = process.__dict__
        # process_state_dict.pop('_waitpid_lock')

        subprocess_dict = {k:v for k,v in process_state_dict.items() if k != '_waitpid_lock'}
        if cache == True:
            self.load_state()
            if key == None or key == 'pid':
                key= process.pid
            self.subprocess_map[key] =subprocess_dict
            self.save_state()
        # return process.__dict__
        # return process
        return subprocess_dict

    submit = add = add_subprocess  
    
    def ls(self):
        self.load_state()
        return list(self.subprocess_map.keys())

    ls_keys = list_keys = list = ls

    @property
    def tmp_dir(self):
        return f'/tmp/commune/{self.name}'
    @property
    def subprocess_map_path(self):
        return os.path.join(self.tmp_dir, 'subprocess_map.json')
    
        
    last_saved_timestamp=0
    @property
    def state_staleness(self):
        self.current_timestamp - self.last_saved_timestamp
    def save_state(self, staleness_period=100):
        data =  self.subprocess_map
        self.client.local.put_json(path=self.subprocess_map_path, data=data)

    def load_state(self):
        self.client.local.makedirs(os.path.dirname(self.subprocess_map_path), True)
        data = self.client.local.get_json(path=self.subprocess_map_path, handle_error=True)
        
        if data == None:
            data  = {}
        self.subprocess_map = data

    @property
    def portConnection( port : int, host='0.0.0.0'):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)       
        result = s.connect_ex((host, port))
        if result == 0: return True
        return False


if __name__ == "__main__":

    import streamlit as st
    module = SubprocessModule.deploy(actor={'refresh': False}, override={'refresh':False})
    st.write(module)
    import ray

    # # st.write(module.subprocess_map)


    st.write(ray.get(module.ls.remote()))
    st.write(ray.get(module.rm_all.remote()))
    # st.write(ray.get(module.add.remote(key='pid', command='python commune/gradio/api/module.py  --module="gradio.client.module.ClientModule"')))
    # st.write(module.ls()) 
    st.write(ray.get(module.ls.remote()))



