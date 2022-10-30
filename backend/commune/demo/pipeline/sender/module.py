import os, sys
sys.path.append(os.environ['PWD'])
from commune import Module

class SenderModule(Module):
    def __init__(self, *args, **kwargs):
        Module.__init__(self, *args, **kwargs)
        st.write(self.list_actor_names())
        self.queue =  self.load_module('commune.asyncio.queue_server', actor={'refresh': False}, wrap=True  )
        # st.write(self.queue)
    def put(self, key='bro', value={'bro': [1,3,4,5]}):
        return self.queue.put(key, value)
    def get(self, key='key'):
        return self.queue.get(key)

if __name__ == '__main__':
    import streamlit as st
    # st.write(Module)
    # Module.new_loop()
    module = SenderModule.deploy(actor={'refresh': False}, wrap=True)

    put_button = st.button('Put')
    send_object = {'bro': [10]*10}

    key  = st.text_input('key','default')
    send_object = st.text_area('string')
    if put_button:
        st.write(module.put(key, send_object))
    # st.write(module.get('key'))