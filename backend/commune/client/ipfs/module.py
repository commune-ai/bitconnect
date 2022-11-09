import io
import os
import time
import weakref
import copy
import asyncio
import aiohttp
from glob import has_magic
import json
from copy import deepcopy
import streamlit as st
import logging
from glob import glob
from typing import *
from ipfshttpclient.multipart import stream_directory, stream_files 


logger = logging.getLogger("ipfsspec")

def sync_wrapper(fn):
    def wrapper_fn(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return  wrapper_fn

IPFSHTTP_LOCAL_HOST = 'ipfs'
class IPFSModule:

    data_dir = '/tmp/ipfs_client'

    def __init__(self,
                ipfs_urls = {'get': f'http://{IPFSHTTP_LOCAL_HOST}:8080', 
                             'post': f'http://{IPFSHTTP_LOCAL_HOST}:5001'},
                loop=None,
                client_kwargs={}):

        self.ipfs_url = ipfs_urls
        self.path2hash = self.load_path2hash()
        self.loop = asyncio.set_event_loop(asyncio.new_event_loop())
        self.sync_the_async()


    def sync_the_async(self):
        for f in dir(self):
            if 'async_' in f:
                setattr(self, f.replace('async_',  ''), sync_wrapper(getattr(self, f)))

    async def async_api_post(self, 
                      endpoint:str, 
                      params:dict = {} ,
                      headers:dict={},
                      data={},
                      return_json:bool = True, 
                      content_type:str=None, 
                      chunk_size:int=1024, 
                      num_chunks:int=None,
                      **kwargs) -> 'aiohttp.Response':
        
        '''
        async api post

        Args:
            url (str):
                url of endpoint.
            return_json (bool): 
                Return repsonse as json.
            content_type (str):
                Content type of request.
            chunk_size (int):
                Chunk size of streaming endpoint.
            num_chunks (int):
                Number of chunks to stream.
        Returns (aiohttp.Response)
        '''


        url = os.path.join(self.ipfs_url['post'],'api/v0', endpoint)


        return_result = None
        # we need to  set the 
        timeout = aiohttp.ClientTimeout(sock_connect=10, sock_read=10)
        async with aiohttp.ClientSession( timeout=timeout) as session:
            async with session.post(url,params=params,headers=headers, data=data) as res:
                if return_json: 
                    return_result = await res.json(content_type=content_type)
                else:
                    return_result = res

                # if num_chunks != None
                if num_chunks:
                    return_result = b''
                    async for data in res.content.iter_chunked(chunk_size):
                        return_result += data
                        num_chunks-= 1
                        if num_chunks == 0:
                            break
        return return_result

    async def async_api_get(self, 
                      endpoint:str,
                     return_json:bool = True,
                     content_type:str=None, 
                     chunk_size:int=1024, 
                     num_chunks:int=1,
                     params: dict={},
                     headers: dict={},
                     **kwargs) -> 'aiohttp.Response':
        '''
        async api post

        Args:
            url (str):
                url of endpoint.
            return_json (bool): 
                Return repsonse as json.
            content_type (str):
                Content type of request.
            chunk_size (int):
                Chunk size of streaming endpoint.
            num_chunks (int):
                Number of chunks to stream.
        Returns (aiohttp.Response)
        '''

        url = os.path.join(self.ipfs_url['get'],'api/v0', endpoint)
    
        return_result = None
        async with aiohttp.ClientSession(loop=self.loop) as session:
            async with session.get(url,params=params,headers=headers) as res:
                if return_json: 
                    return_result = await res.json(content_type=content_type)
                else:
                    return_result = res

                if chunk_size:
                    return_result = b''
                    async for data in res.content.iter_chunked(chunk_size):
                        return_result += data
                        num_chunks-= 1
                        if num_chunks == 0:
                            break
        return return_result

    async def async_version(self, session):
        res = await self.async_api_get("version")
        return rest

    
    def resolve_absolute_path(self, path):
        if path[:len(os.getenv('PWD'))] != os.getenv('PWD'):
            path = os.getenv('PWD')+'/' + path
        
        return path

    async def async_cat(self, path, *args, **kwargs):
        res = await self.async_api_get(endpoint='cat', arg=path,  *args, **kwargs)
        return res

    async def async_pin(self, session, cid, recursive=False, progress=False, **kwargs):
        kwargs['params'] = kwargs.get('params', {})
        kwargs['params'] = dict(arg=cid, recursive= recursive,progress= progress)
        res = await self.async_api_post(endpoint='pin/add', arg=cid, recursive= recursive,  **kwargs)
        return bool(cid in pinned_cid_list)



    async def async_add(self,
            path,
            pin=True,
            chunker=262144 ):
        path = self.resolve_absolute_path(path)
        self.path2hash = await self.async_load_path2hash()
        file_paths=[]
        assert os.path.exists(path), f'{path} does not exist'
        if os.path.isdir(path):
            file_paths = glob(path+'/**', recursive=True)
        elif os.path.isfile(path):
            file_paths = [path]
  

        file_paths = list(filter(os.path.isfile, file_paths))

        assert len(file_paths) > 0
    
        jobs = asyncio.gather(*[self.async_add_file(path=fp, pin=pin, chunker=chunker) for fp in file_paths])
        responses = await jobs
        path2hash =  dict(zip(file_paths,responses))
        self.path2hash.update(path2hash)
        await self.async_save_path2hash()

        return dict(zip(file_paths,responses))


    async def async_rm(self, path):
        await self.async_load_path2hash()
        file_paths = await self.async_ls(path)      
        tasks = []
        for fp in file_paths:
            file_meta = self.path2hash[fp]
            tasks.append(self.async_pin_rm(cid=file_meta['Hash']))
        return_jobs = await asyncio.gather(*tasks)
        await self.async_gc()

        await self.async_save_path2hash()
        return return_jobs

    async def async_pin_ls(self,
        type_:str='all', # The type of pinned keys to list. Can be "direct", "indirect", "recursive", or "all"
        **kwargs,
    ):
        'List objects pinned to local storage.'    
        params = {}
        params['type'] = type_
        params.update(kwargs)
        return await self.async_api_post('pin/ls', params=params)

    async def async_gc(self):

        response = await self.async_api_post('repo/gc', return_json=False)
        return response

    async def async_pin_rm(self,
        cid:str, # Path to object(s) to be unpinned
        recursive:str='true', #  Recursively unpin the object linked to by the specified object(s)
        **kwargs,
    ):
        'List objects pinned to local storage.'    

        params = {}
        params['arg'] = cid
        params['recursive'] = recursive
        params.update(kwargs)

        response = await self.async_api_post('pin/rm', params=params)
        await self.async_load_path2hash()
        return response

    async def async_add_file(self,
        path,
        pin=False,
        chunker=262144, 
        wrap_with_directory=False,
    ):

        path = self.resolve_absolute_path(path)

        params = {}
        params['wrap-with-directory'] = 'true' if wrap_with_directory else 'false'
        params['chunker'] = f'size-{chunker}'
        params['pin'] = 'true' if pin else 'false'
        data, headers = stream_files(path, chunk_size=chunker)

        async def data_gen_wrapper(data):
            for d in data:
                yield d

        data = data_gen_wrapper(data=data)   
             
        res = await self.async_api_post(endpoint='add',  params=params, data=data, headers=headers)
        return res
        # return res
    

    async def async_dag_get(self,  **kwargs):
        kwargs['params'] = kwargs.get('params', {})
        kwargs['params'] = dict(arg=cid, recursive= recursive,progress= progress)
        res = await self.async_api_post(endpoint='dag/get', **kwargs)
        return bool(cid in pinned_cid_list)
    dag_get = sync_wrapper(async_dag_get)

    async def async_rm_json(self, path=None, recursive=True, **kwargs):
        path = os.path.join(self.data_dir, path)
        return os.remove(path)

    rm_json = sync_wrapper(async_rm_json)
    async def async_save_json(self, 
                        path:str,
                        obj:Union[dict, list],
                        include_root:bool=True) -> str:
        """ 
        Async save of json for storing text hashes

        Args:
            path (List[str]):
                Axon to serve.
            obj (bool):
                The object to save locally
            include_root (bool):
                Include self.data_dir as the prefix.
                    - if True, ths meants shortens the batch and 
                    specializes it to be with respect to the dataset's 
                    root path which is in ./bittensor/dataset
            
        Returns: 
            path (str)
                Path of the saved JSON
        """
        
        if include_root:
            path = os.path.join(self.data_dir, path)

        dir_path = os.path.dirname(path)

        # ensure the json is the prefix
        if path[-len('.json'):] != '.json':
            path += '.json'

        # ensure the directory exists, make otherwise
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        assert os.access( dir_path , os.W_OK ), f'dir_path:{dir_path} is not writable'
        with open(path, 'w') as outfile:
            json.dump(obj, outfile)

        return path


    save_json = sync_wrapper(async_save_json)
    async def async_load_json(self, path:str,include_root:bool=True, default:Union[list, dict]={}) -> Union[list, dict]:

        """ 
        Async save of json for storing text hashes
        Args:
            path (str):
                Path of the loaded json
            include_root (bool):
                Include self.data_dir as the prefix.
                    - if True, ths meants shortens the batch and 
                    specializes it to be with respect to the dataset's 
                    root path which is in ./bittensor/dataset
        Returns: 
            obj (str)
                Object of the saved JSON.
        """
        
        if include_root:
            path = os.path.join(self.data_dir, path)

        # Ensure extension.
        dir_path = os.path.dirname(path)
        if os.path.splitext(path)[-1] != '.json':
            path += '.json'

        # Ensure dictionary.
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        # Load default if file does not exist.
        try:
            with open(path, 'r') as f:
                obj = json.load(f)
        except FileNotFoundError:
            obj = default
        except json.JSONDecodeError:
            obj = default

        if isinstance(obj, str):
            obj = json.loads(obj)
        return obj

    load_json = sync_wrapper(async_load_json)

    async def async_ls(self, path=''):
        await self.async_load_path2hash()
        path = self.resolve_absolute_path(path)
        path_list = []
        for fp in self.path2hash.keys():
            if fp[:len(path)] == path:
                path_list += [fp]
        return path_list
    ls = sync_wrapper(async_ls)
    async def async_save_path2hash(self):
        pinned_cids = (await self.async_pin_ls()).get('Keys', {}).keys()
        path2hash = {}
        for path, file_meta in self.path2hash.items():
            if file_meta['Hash'] in pinned_cids:
                path2hash[path] = file_meta

        await self.async_save_json('path2hash', path2hash )

    save_path2hash = sync_wrapper(async_save_path2hash)
    
    async def async_load_path2hash(self):
        loaded_path2hash  = await self.async_load_json('path2hash')
        pinned_cids = (await self.async_pin_ls()).get('Keys', {}).keys()
        path2hash = {}
        for path, file_meta in loaded_path2hash.items():
            if file_meta['Hash'] in pinned_cids:
                path2hash[path] = file_meta
        self.path2hash = path2hash
        return path2hash
    load_path2hash = sync_wrapper(async_load_path2hash)

    
    @property
    def hash2path(self):
        path2hash = self.load_path2hash()
        return {file_meta['Hash']: path for path, file_meta in path2hash.items()}


    @classmethod
    def test_add_rm_file(cls):
        module = cls()
        test_path = 'commune/client/local/module.py'
        module.add(test_path)
        file_paths = module.ls(test_path)
        assert len(file_paths) > 0
        module.rm(test_path)
        file_paths = module.ls(test_path)
        assert len(file_paths) == 0

    @classmethod
    def test_add_rm_folder(cls):
        module = cls()
        test_path = 'commune/client/local'
        module.add(test_path)
        file_paths = module.ls(test_path)
        assert len(file_paths) > 0
        module.rm(test_path)
        file_paths = module.ls(test_path)
        assert len(file_paths) == 0

    @classmethod
    def test(cls):
        for f in dir(cls):
            if 'test_' in f:
                getattr(cls, f)()
                st.write(f'PASSED: {f}')


    ##############
    #   ASYNCIO
    ##############
    @staticmethod
    def reset_event_loop(set_loop:bool=True) -> 'asyncio.loop':
        '''
        Reset the event loop

        Args:
            set_loop (bool):
                Set event loop if true.

        Returns (asyncio.loop)
        '''
        loop = asyncio.new_event_loop()
        if set_loop:
            asyncio.set_event_loop(loop)
        return loop

    def set_event_loop(self, loop:'asyncio.loop'=None)-> 'asynco.loop':
        '''
        Set the event loop.

        Args:
            loop (asyncio.loop):
                Event loop.

        Returns (asyncio.loop)
        '''
        
        if loop == None:
            loop = asyncio.get_event_loop()
        self.loop = loop
        return self.loop
  
    @classmethod
    def test_load_save_json(cls):
        module = cls()
        obj = {'bro': [1]}
        module.save_json('fam',obj )
        assert obj == module.load_json('fam')

    async def async_put_json(self, path,input):
        save_path = await self.async_save_json(path,input)
        file_meta = await self.async_add(path=save_path)
        await self.rm_json(path)
        return file_meta
    

    async def async_cat(self, cid):
        return await self.async_api_get('cat', cid)

if __name__ == '__main__':
    IPFSClient.test()
    module = IPFSClient()
    st.write(glob('commune/**', recursive=True))
