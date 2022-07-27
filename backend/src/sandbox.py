import streamlit as st
import bittensor
import plotly 


class ExplainBittensor:
    def __init__(self, graph=None):
        self.graph  = bittensor.metagraph().sync()

    @classmethod
    def st_sidebar(cls):
        st.sidebar.slider('Block', 0, )
        st.sidebar.select('Network', ['Bro'])


ExplainBittensor.st_sidebar()


@st.cache
def get_graph(block=None):
    # Fetch data from URL here, and then clean it up.
    graph = bittensor.metagraph().sync()
    return graph


graph = get_graph()

st.write(graph.__dict__.keys())

st.write(graph.to_dataframe().iloc[:10])