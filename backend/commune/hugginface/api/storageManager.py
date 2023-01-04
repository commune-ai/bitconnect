import os
from  .aiofile import async_read, async_write, sync_wrapper
import json
import os

async def async_get_json(path):
    data = json.loads(await async_read(path))
    return data

read_json = load_json = get_json = sync_wrapper(async_get_json)

async def async_put_json( path, data):
    # Directly from dictionary
    path = ensure_path(path)
    json_str = json.dumps(data)    
    return await async_write(path, json_str)

put_json = save_json = sync_wrapper(async_put_json)

def path_exists(path:str):
    return os.path.exists(path)

def ensure_path( path):
    """
    ensures a dir_path exists, otherwise, it will create it 
    """

    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    return path