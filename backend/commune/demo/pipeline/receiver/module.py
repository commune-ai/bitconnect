import os, sys
sys.path.append(os.environ['PWD'])
from commune import Module

class SenderModule(Module):
    def __init__(self, *args, **kwargs):
        Module.__init__(self, *args, **kwargs)
        st.write(self.list_actor_names())
        self.queue =  self.load_module('commune.asyncio.queue_server', actor={'refresh': False}, wrap=True  )
        st.write(self.queue)
    def put(self, key='bro', value={'bro': [1,3,4,5]}):
        return self.queue.put(key, value)
    def get(self, key='key'):
        
        return self.queue.get(key)

import asyncio

class StreamlitApp:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self.main())

    async def main(self):
        job = self.bro()

        self.loop.call_soon(job)
        self.loop.create_task(job)
        st.write('fam')

if __name__ == '__main__':
    import streamlit as st
    # st.write(Module)
    # Module.new_loop()
    # module = SenderModule.deploy(actor=True, wrap=True)
    # import ray
    # st.write(ray.get(module.queue.getattr('actor_name')))
    

    # stop = st.button('stop')
    # send_object = {'bro': [10]*10}
    
    # # st.write(module.put('key', send_object))
    # # if get_button:
    # key  = st.text_input('key','default')

    # response_list  = []
    # while not stop: 
    #     response_list.append(module.get(key))
    #     st.write(len(response_list))
    


    StreamlitApp()