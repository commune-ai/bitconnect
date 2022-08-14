import io
import os
import json
import pickle


def str_is_dict(input_str):
    try:
        assert dict(json.loads(str))
    except Exception as e:
        print('ERROR: Not a dict buddy')
        raise e

def ensure_path(path):
    """
    ensures a dir_path exists, otherwise, it will create it 
    """
    if os.path.isfile(path):
        dir_path = os.path.dirname(path)
    elif os.path.isdir(path):
        dir_path = path
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def load_json(path, handle_error = True):
    try:
        with open(path) as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError as e:
        if handle_error:
            return None
        else:
            raise e

def write_json(path, data):
        # Directly from dictionary
    ensure_dir_path(os.path.dirname(path))
    if isinstance(data, dict):
        with open(path, 'w') as outfile:
            json.dump(data, outfile)
    
    elif isinstance(data, str):
        str_is_dict(data) 
        # Using a JSON string
        with open(path, 'w') as outfile:
            outfile.write(data)

def put_pickle(bucket_name, object_name, data, client): 
    bytes_file = pickle.dumps(data)
    client.put_object(
        bucket_name=bucket_name,
        object_name=object_name,
        data=io.BytesIO(bytes_file),
        length=len(bytes_file))


def get_pickle(bucket_name, object_name, client):
    return pickle.loads(
                    client.get_object(
                        bucket_name=bucket_name,
                        object_name=object_name).read()
                )

def get_json(bucket_name, object_name, client):
    """
    get stored json object from the bucket
    """
    data = client.get_object(bucket_name, object_name)
    return json.load(io.BytesIO(data.data))
