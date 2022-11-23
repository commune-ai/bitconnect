import sys
import os
sys.path.append(os.environ['PWD'])
from commune import Module
import gradio as gr
import streamlit as st


class TestGradio(Module):
    default_config_path =  'gradio.test'

    @classmethod
    def gradio(cls):
        def test(s : str) -> str:
            return f"{s} :Information Reccived"

        return gr.Interface(fn=test, inputs="text", outputs="text")

    @classmethod
    def streamlit(cls):
        st.write("Hello World")
        st.write("This function is a inherited from Class Module which allows the API to check if this function is Streamlit streamable")


if __name__ == "__main__":
    TestGradio().run()