import io
import json
import pickle

def put_json(bucket_name, object_name, data, client):
    """
    jsonify a dict and write it as object to the bucket
    """
    # prepare data and corresponding data stream
    data = json.dumps(data).encode("utf-8")
    data_stream = io.BytesIO(data)
    data_stream.seek(0)

    # put data as object into the bucket
    client.put_object(
        bucket_name=bucket_name,
        object_name=object_name,
        data=data_stream, length=len(data),
        content_type="application/json"
    )
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