import streamlit as st
import bittensor
import plotly 
import _thread


class BitModule:
    networks = ['nakamoto', 'nobunaga', 'local']
    def __init__(self,
                 network='local',
                 block=None):
        self.sync_network(network=network, block=block)

    @property
    def block(self):
        return self.subtensor.block

    def n(self):
        return self.subtensor.max_n

    def sync_network(self,network='nakamoto',  block=None):
        # Fetch data from URL here, and then clean it up.
        
        @st.cache
        def get_network(block=block,network=network):
            subtensor = self.get_subtensor(network=network)
            graph = self.get_graph(subtensor=subtensor , block=block)
            return graph
        self.graph = get_network(network=network, block=block)



    @staticmethod
    def get_graph( subtensor=None, block=None ):
        # Fetch data from URL here, and then clean it up.
        return bittensor.metagraph(subtensor=subtensor).sync(block=block)

    @staticmethod
    def get_subtensor( network='nakamoto', **kwargs):
        '''
        The subtensor.network should likely be one of the following choices:
            -- local - (your locally running node)
            -- nobunaga - (staging)
            -- nakamoto - (main)
        '''
        subtensor = bittensor.subtensor(network=network)
        return subtensor
    
    '''
        streamlit functions
    '''

    @staticmethod
    def describe(key):
        st.markdown('# '+key)
        module = getattr(bittensor,key)
        for fn_key in dir(module):
            fn = getattr(module,fn_key)
            if callable(fn):
                st.markdown('#### '+fn_key)
                st.write(fn)
                st.write(type(fn))

            # with st.expander(fn_name):
            #     st.write(getattr(module,fn_name) )

    @classmethod
    def st_sidebar(cls):
        st.sidebar.slider('Block', 0, )
        st.sidebar.selectbox('Network', cls.networks)

manual =  st.expander('bro')
# with manual:
#     src='https://docs.streamlit.io/library/components/components-api'
#     st.components.v1.iframe(src, width=None, height=1000, scrolling=False)
BitModule.st_sidebar()
subtensor = None


# bt = ExplainBittensor()
# for fn_name in dir(subtensor):
#     with st.expander(fn_name):
#         st.write(getattr(subtensor,fn_name) )
subtensor =  bittensor.subtensor(network='local')
# st.write(dir(subtensor))
st.write(subtensor.neuron_for_pubkey)
# BitModule.describe('wallet')
# wallet = bittensor.wallet()
# st.write(dir(wallet))
st.write(bittensor.cli().create_new_coldkey())
# st.write(wallet.new_coldkey(overwrite=True))