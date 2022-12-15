""" An interface for serializing and deserializing bittensor tensors"""

# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

import torch
import msgpack
import msgpack_numpy
from typing import Tuple, List, Union, Optional
import sys
import os
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())
sys.path.append(os.getenv('PWD'))
from commune.proto import DataBlock
import commune
import json
import streamlit as st


class SerializerModule:
    r""" Bittensor base serialization object for converting between DataBlock and their
    various python tensor equivalents. i.e. torch.Tensor or tensorflow.Tensor
    """

    @staticmethod
    def empty():
        """Returns an empty DataBlock message with the version"""
        return DataBlock()

    def serialize (self, data: object, metadata:dict={}, mode='dict') -> DataBlock:
        serializer = getattr(self, f'serialize_{mode}')
        data_bytes, metadata = serializer( data = data, metadata=metadata )
        metadata['mode'] =  mode
        metadata_bytes = self.dict2bytes(metadata)
        return DataBlock(data=data_bytes, metadata = metadata_bytes)

    def deserialize (self, proto: DataBlock) -> object:
        """Serializes a torch object to DataBlock wire format.
        """
        metadata = self.bytes2dict(proto.metadata)
        mode = metadata['mode']
        deserializer = getattr(self, f'deserialize_{mode}')
        
        return deserializer( data = proto.data , metadata= metadata)


    """
    ################ BIG DICT LAND ############################
    """

    def serialize_dict(self, data: dict, metadata:dict={}) -> DataBlock:
        data = self.dict2bytes(data=data)
        st.write(data)
        return  data,  metadata

    def deserialize_dict(self, data: bytes, metadata:dict={}) -> DataBlock:
        data = self.dict2bytes(data=data)
        return data

    @staticmethod
    def dict2bytes(data:dict={}) -> bytes:
        data_json_str = json.dumps(data)
        data_json_bytes = msgpack.packb(data_json_str)
        return data_json_bytes

    @staticmethod 
    def bytes2dict(self, data:bytes) -> dict:
        json_object_bytes = msgpack.unpackb(data)
        return json.loads(json_object_bytes)


    """
    ################ BIG TORCH LAND ############################
    """


    def serialize_torch(self, data: torch.Tensor, metadata:dict={}) -> DataBlock:

        metadata['dtype'] = torch_tensor.dtype
        metadata['shape'] = list(torch_tensor.shape)
        metadata['requires_grad'] = torch_tensor.requires_grad
        data = self.torch2bytes(data=data)

        return  data,  metadata

    def deserialize_torch(self, data: bytes, metadata: dict) -> torch.Tensor:
        dtype = metadata['dtype']
        shape = metadata['shape']
        requires_grad = metadata['requires_grad']
        data =  self.torch2bytes(data=data, shape=shape, dtype=dtype, requires_grad=requires_grad )
        return data
    @staticmethod
    def torch2bytes(data:torch.Tensor)-> bytes:
        torch_numpy = data.cpu().detach().numpy().copy()
        torch_object = msgpack.packb(torch_numpy, default=msgpack_numpy.encode)
        return data_buffer

    @staticmethod
    def bytes2torch(data:bytes, shape:list, dtype:str, requires_grad:bool=False) -> torch.Tensor:
        numpy_object = msgpack.unpackb(data, object_hook=msgpack_numpy.decode).copy()
        torch_object = torch.as_tensor(numpy_object).view(shape).requires_grad_(requires_grad)
        torch_object =  torch_object.type(dtype)
        return torch_object

    @property
    def get_str_type(data):
        return str(type(data)).split("'")[1]
if __name__ == "__main__":
    module = SerializerModule()
    data = {'bro': [0, 1, 'fam']}
    st.write(module.get_str_type(torch.tensor([0,4,5])))
