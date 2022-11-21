
from .asyncio import async_read, async_write, sync_wrapper
import asyncio
import json

async def async_get_json(path, return_type='dict'):
    try:  
        
        data = json.loads(await async_read(path))
    except FileNotFoundError as e:
        if handle_error:
            return None
        else:
            raise e

    if return_type in ['dict', 'json']:
        data = data
    elif return_type in ['pandas', 'pd']:
        data = pd.DataFrame(data)
    elif return_type in ['torch']:
        torch.tensor
    return data

read_json = sync_wrapper(async_get_json)

async def async_put_json( path, data):
        # Directly from dictionary
    data_type = type(data)
    if data_type in [dict, list, tuple, set, float, str, int]:
        json_str = json.dumps(data)
    elif data_type in [pd.DataFrame]:
        json_str = json.dumps(data.to_dict())
    else:
        raise NotImplementedError(f"{data_type}, is not supported")
    
    return await async_write(path, json_str)

put_json = sync_wrapper(async_put_json)
