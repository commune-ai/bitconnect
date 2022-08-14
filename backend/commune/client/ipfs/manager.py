from copy import deepcopy
from re import M
from symbol import return_stmt
import os
import sys
import pickle
sys.path.append(os.environ['PWD'])
from commune.ray.actor import ActorBase
from commune.utils.misc import dict_fn, get_object
import tempfile
import json
import ipfshttpclient
import codecs
from commune.client.mongo.manager import MongoManager
import streamlit as st
import ast
class IPFSManager(ActorBase):
    default_cfg_path = f"{os.path.dirname(__file__)}/manager.yaml"
    def __init__(self, cfg):
        # self.client_kwargs = kwargs
        # self.client = Minio(**kwargs)
        self.client = ipfshttpclient.connect(**cfg['ipfs_client'])
        self.mongo_client = MongoManager(cfg['mongo_client'])
        self.supported_modes = ['json', 'pkl']
        self.cfg = cfg


    def write_object(self, data, mode='json', pin=True):

        input_dict = {'data' : '', 'mode' : mode}
        if mode == 'json':
            input_dict['data'] = json.dumps(data).replace("'", '"')
        elif mode == 'torch.state_dict':
            input_dict['data'] = codecs.encode(pickle.dumps(data), "base64").decode()
        elif mode == 'torch.model':
            raise Exception(f"Saving Torch Model is Not Implemented")
        elif mode == 'torch.onnx':
            raise Exception(f"Saving Torch ONNX is Not Implemented")
        else:
            raise Exception(f"only supports json: Input {mode}")


        cid = self.client.add_json(json.dumps(input_dict))

        if pin:
            self.client.pin.add(cid)

        return cid

    def read_object(self, cid, pin=False):
        object_dict = json.loads(self.client.get_json(cid))
        raw_data = object_dict.get('data')
        mode = object_dict.get('mode')
        if mode == 'json':
            data = json.loads(raw_data) 
        elif mode == 'torch.state_dict':
            data = pickle.loads(codecs.decode(raw_data.encode(), "base64"))
        elif mode == 'torch.model':
            raise Exception(f"{module} not impleneted")
        elif mode == 'torch.onnx':
            raise Exception(f"{module} not impleneted")
        elif mode == 'python':
            raise Exception(f"{module} not impleneted")
        elif mode == 'pickle':
            raise Exception(f"{module} not impleneted")
        elif mode == 'module':
            raise Exception(f"{module} not impleneted")
        else:
            raise Exception(f"{module} not impleneted")


        if pin:
            self.client.add.pin(cid)
    
        return data

    def load(self, 
            hash=None,
            meta=None,
            mode='json', 
            return_hash=True,
            index=None):
        """
        name: name of the object
        taxonomy: name of the taxonomy of the object
        client: mongod client for staching
        """
        
        if meta:
            documents = self.mongo_client.find(collection=self.cfg['collection'],
                                    database=self.cfg['database'], query=meta)
            outputs = []

            for i,document in enumerate(documents):
                if return_hash:
                    outputs.append(document['hash'])
                else:
                    outputs.append(self.read_object(document['hash'], mode=mode))

                if index == i:
                    return outputs[i]
        
            return documents
        else:
            return self.read_object(hash)
            

    def write(self,data, meta=None, workers=1):
        

        hash_key = self.write_object(data=data, mode=mode)

        if meta:
            document = deepcopy(meta)
            document['hash'] = hash_key

            
            updates = [{'filter': meta,
                        'update': {'$set': document},
                    'upsert': True}]

            self.mongo_client.update(collection=self.cfg['collection'],
                                        database=self.cfg['database'],
                                        updates=updates,
                                        workers=workers)


        return hash_key

    # this is a temp fix to get attributes from a given actor
    def get(self, item):
        if item is None:
            return self.__dict__
        else:
            return getattr(self,item)

    @classmethod
    def create_actor(cls,**kwargs):
        create_actor(cls, **kwargs)



if __name__ == '__main__':
    import torch
    from torch import nn
    import io
    from commune.process.base import BaseProcess


    class Model(torch.nn.Module, BaseProcess): 
        def __init__(self):
            super().__init__()
        
            self.layer1 = nn.Linear(10,10)




    client = IPFSManager.deploy(actor=False) 
    model = Model()

    # cid = client.write_object(data=model.state_dict(), mode='torch.state_dict', pin=True)
    # data = client.read_object(cid)




