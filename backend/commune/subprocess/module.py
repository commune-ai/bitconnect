


import os, sys
sys.path.append(os.environ['PWD'])
import gradio
from commune import BaseModule
from commune.utils import *
import shlex
import subprocess

class SubprocessModule(BaseModule):
    process_map = {}
    default_config_path =  'subprocess.module'
    def __init__(self, config=None, **kwargs):
        BaseModule.__init__(self, config=config)
        if self.config.get('refresh') == True:
            self.process_map = {}
            self.save_state()
        self.load_state()

        self.process_map = {p:v for p,v in self.process_map.items() if self.check_pid(v['pid']) }




    def __reduce__(self):
        deserializer = self.__class__
        serialized_data = (self.config,)
        return deserializer, serialized_data

    def submit(command):
        return self.run_command(command)
    

    def kill_subprocess(self, pid):
        self.kill_pid(pid)
        self.process_map.pop(pid)
        self.save_state()

    def run_subprocess(self, command:str,cache=True):

        process = subprocess.Popen(shlex.split(command))
        process_state_dict = process.__dict__
        # process_state_dict.pop('_waitpid_lock')

        if cache == True:
            self.load_state()
            key= process.pid
            self.process_map[key] = {k:v for k,v in process_state_dict.items() if k != '_waitpid_lock'}
            self.save_state()
        return process
        # return process

    def ls(self):
        self.load_state()
        return self.process_map.keys()

    @property
    def tmp_dir(self):
        return f'/tmp/commune/{self.name}'
    @property
    def process_map_path(self):
        return os.path.join(self.tmp_dir, 'process_map.json')
    
        
    last_saved_timestamp=0
    @property
    def state_staleness(self):
        self.current_timestamp - self.last_saved_timestamp
    def save_state(self, staleness_period=100):
        data =  self.process_map
        self.client.local.put_json(path=self.process_map_path, data=data)

    def load_state(self):
        self.client.local.makedirs(os.path.dirname(self.process_map_path), True)
        data = self.client.local.get_json(path=self.process_map_path, handle_error=True)
        
        if data == None:
            data  = {}
        self.process_map = data


if __name__ == "__main__":

    import streamlit as st
    module = SubprocessModule.deploy(actor=False, override={'refresh':False})
    st.write(module)
    import ray

    # # st.write(module.process_map)
    for pid in deepcopy(list(module.process_map.keys())):
        # pid = int(list(module.process_map.keys())[-1])
        # st.write(module.check_pid(pid))
        module.kill_subprocess(pid)

    module.run_subprocess('python commune/gradio/api/module.py  --module="gradio.client.module.ClientModule"')
    
    # st.write(module.process_map)

    st.write(module.process_map)