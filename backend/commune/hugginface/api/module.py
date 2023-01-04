import requests
import os
from time import time
import asyncio
import aiohttp
from .storageManager import async_get_json, async_put_json
from enum import Enum
import sys
sys.path.append(os.environ['PWD'])
from .utils import statusStrategy, searchConfigForApiName
from functools import partial
from commune import Module

API_MODELS_URL = "https://huggingface.co/api/models"
API_SPACES_URL = "https://huggingface.co/api/spaces"
API_DATASET_URL = "https://huggingface.co/api/datasets"
API_URL_MODEL = "https://api-inference.huggingface.co/models/"
GRADIO_SPACES_API_URL = "https://$SPACEHOST.hf.space"
UNIX_TIME_HOUR = 3600


    
async def query_model(payload, model, api_key):
        url = os.path.join(API_URL_MODEL, model)
        print(url)
        return_result = None
        timeout = aiohttp.ClientTimeout(sock_connect=100, sock_read=100)
        try: 
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=api_key, json=payload) as res:
                    return_result = await res.json()
                    
                    if "error" in return_result: 
                        return return_result
                    return return_result
        except Exception as e:
            return e

async def does_space_exsit(api):
        return_result = None
        timeout = aiohttp.ClientTimeout(sock_connect=3, sock_read=3)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(api) as res:
                    return_result = res.ok
        except Exception as e:
            return (api, False)
        return (api.replace("/config", ""), return_result)

async def does_hf_space_api_exist(space):
        does_have_space_api = ["/api", "/config"]
        timeout = aiohttp.ClientTimeout(sock_connect=20, sock_read=20)
        for api in does_have_space_api:
            url = space + api
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as res:
                        if "/api" == api and not statusStrategy(res) == None:
                            return (space, True)
                        if "/config" == api:
                            response = await res.json()
                            return (space , searchConfigForApiName(response) == 200)
            except Exception as e:
                continue


class CallFunction(Enum):
    does_hf_space_api_exist = does_hf_space_api_exist
    does_space_exsit = partial(does_space_exsit)
    query_hugginface_model = partial(query_model)
    async def __call__(self, *args):
        return await self.value(*args)

class Colour:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Cacher(Module):
    default_config_path = "hugginface.api"

    def request_json_from_api(self, api : str) -> any:
        """ return json request from api call

        ### Example

        Example 1:
        >>> c = Cacher()
        >>> c.request_json_from_api(path="...", file="mode.json", api="...")
        >>> {...}

        """
        res = requests.get(api)
        return res.json()

    def read_write_api_data(self, file : str, api : str, dir : str=".caches", **kwargs) -> any:
        """Return the list of data either form the api or
        the current cache data stored form less then
        a hour ago.

        ### Examples
        
        Ex 1:
        >>> c = Cacher()
        >>> data = c.read_write_api_data(file="modes.json", api="...")
        ==File Write==
        >>> c.read_write_api_data(file="models.json", api="...") # within the same hour
        ==File Read==
        """
        return asyncio.run(self.__async_read_write_api_data(file=file, api=api, dir=dir, **kwargs))

    async def __async_read_write_api_data(self, file : str, api : str, dir : str=".caches", **kwargs) -> any:
        """Return the list of data either form the api or
        the current cache data stored form less then
        a hour ago.

        ### Examples
        
        Ex 1:
        >>> c = Cacher()
        >>> data = asyncio.run(c.__async_read_write_api_data(file="modes.json", api="..."))
        ==File Write==
        >>> asyncio.run(c.__async_read_write_api_data(file="models.json", api="...")) # within the same hour
        ==File Read==
        """
        assert not file == "" and not api == "", f"{Colour.BOLD}{Colour.FAIL}file or api param can not be empty{Colour.ENDC}"
        cwd_file_path = os.path.join(dir, file)
        full_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), cwd_file_path)
        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dir)
        
        # check if there exist a path exist
        if not os.path.exists(full_path) or "override" in kwargs and kwargs["override"] == True:
            print(f"=={Colour.OKGREEN}API Call, and Write{Colour.ENDC}==")
            data = self.request_json_from_api(api=api)
            await async_put_json(full_path, data)
            return data# await the return write withing the path and return of the API call
        else:
            modified_file_threshold = os.stat(full_path).st_mtime + UNIX_TIME_HOUR
            current_time = time()

            if (current_time > modified_file_threshold):
                print(f"=={Colour.OKGREEN}File Write{Colour.ENDC}==")
                data = self.request_json_from_api(api=api)
                await async_put_json(full_path, data)
                return await data
            else:
                print(f"=={Colour.OKGREEN}File Read{Colour.ENDC}==")
                return await async_get_json(full_path)


    def get_cache(self, dir : str="", file : str="") -> list:
        return asyncio.run(self.__async_get_cache(dir=dir, file=file))

    async def __async_get_cache(self, dir : str="", file : str="") -> list:
        assert file != "" and dir != "", f"{Colour.BOLD}{Colour.FAIL}file or Directory param can not be empty{Colour.ENDC}"
        cache = os.path.join(dir, file)
        full_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), cache)
        assert os.path.exists(full_path), f"{Colour.BOLD}{Colour.FAIL}file does not exist{Colour.ENDC}"
        dlist = await async_get_json(full_path)
        return dlist
    
    
    def put_cache(self, dir : str="", file : str="", data={}) -> None:
        return asyncio.run(self.__async_put_cache(dir=dir, file=file, data=data))


    async def __async_put_cache(self, dir : str="", file : str="", data={}) -> None:
        """return None and put data within cache

        ### Example
        
        Example 1:
        >>> c = Cacher()
        >>> c.put_cache(dir=".caches", file="mode.json", data={"Hello" : "World"})


        Example 2:
        >>> c.put_cache(data={})
        >>> "Assertion Error: file or Directory param can not be empty"
        
        """
        assert file != "" and dir != "", f"{Colour.BOLD}{Colour.FAIL}file or Directory param can not be empty{Colour.ENDC}"
        cache = os.path.join(dir, file)
        full_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), cache)
        await async_put_json(path=full_path, data=data)


    def rm_file(self, file : str, dir : str=".caches") -> bool:
        """Remove given file from the current caches folder and return boolean unless OSError then raise OSError

        :param str file: cache file stored within the directory (dir) prama
        :param str dir : cache directory within the ./cacher directory default=".caches"
        :rtype: bool
        :raise OSError if is a problem removing the file with the directory
        ### Examples
        
        ```python
        # current state of directory=======================
        #|__cacher.py
        #|__utils.py
        #|__aiofile.py
        #|__storeManager.py
        #|__/.caches
        #       |__ mode.json
        # =================================================
        ```

        >>> Cacher().rm_file(file="mode.json")
        True
        
        ```python
        # current state of directory =======================
        #|__cacher.py
        #|__utils.py
        #|__aiofile.py
        #|__storeManager.py
        #|__/.caches
        # =================================================
        ```

        >>> Cacher().rm_file(file="mode.json")
        ==File Not Found==
        False

        """
        file_path = os.path.join(dir, file)
        full_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), file_path)
        try :
            print(full_file_path)
            if os.path.exists(full_file_path):
                os.remove(full_file_path)
                return True
            else:
                print(f"=={Colour.WARNING}File Not Found{Colour.ENDC}==")
                return False
        except OSError as o:
            print(f"=={Colour.FAIL}OSError{Colour.ENDC}==")
            raise o

    async def async_call(self, fn : object, urls, *args, **kwargs):
        return await asyncio.gather(*[fn(url, *args, **kwargs) for url in urls])

    def store_api(self, fn : CallFunction, urls, dir='.caches', file="temp.json"):
        data = asyncio.run(self.async_call(fn, urls))
        self.put_cache(dir=dir, file=file, data=data)
        return True
    
if __name__ == "__main__":
    Cacher().run()