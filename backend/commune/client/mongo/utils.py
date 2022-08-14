import os
import re
import multiprocessing as mp
from pymongo import MongoClient, UpdateOne
from commune.utils.misc import chunk
from commune.ray.utils import create_actor
import ray

class MongoConnection:
    def __init__(self, port, host ):
        self.port = int(port)
        self.host =  host
        self.client = MongoClient(port=port, host=host)

    # without '__reduce__', the instance is unserializable.
    def __reduce__(self):
        deserializer = MongoClient
        serialized_data = (int(self.port), self.host)
        return deserializer, serialized_data




def generate_client(client_kwargs):
    client = MongoClient(**client_kwargs)
    return client


def update(collection,
           database,
           updates,
           write_type= "bulk_write",
           client = None, # pass the client or client_kwargs
           client_kwargs=None, # client kwargs to generate
           transient_client=False # destroy the client after the method is done
           ):
    """

    update list of documents
    """

    # generate client if it does not exist
    if client is None:
        client = generate_client(client_kwargs)

    upserts = [UpdateOne(**update) for update in updates]
    insert_fn = getattr(client[database][collection],write_type)
    insert_fn(upserts)

    # close connection/delete client if connection is transient
    if transient_client:
        client.close()
        del client

def find_query(database,
               collection,
               skip_n = 0, # no skip default
               limit_n = 100, # no limit default
               client=None,
               client_kwargs = None,
               query={},
               projection = {},
               transient_client=False):


    try:
        if client is None:
            client = generate_client(client_kwargs=client_kwargs)

        find_kwargs = {'filter': query}
        if projection:
            if isinstance(projection, list):
                find_kwargs['projection'] =  {p: 1 for p in projection}

            elif isinstance(projection,dict):
                find_kwargs['projection'] =  projection

        output = list(client[database][collection].find(**find_kwargs).skip(skip_n).limit(limit_n))

    except Exception as e:
        raise(e)
        # pass
    finally:
        if transient_client:
            client.close()
            del client

    return output






def insert(database,
        collection,
        documents,
        write_type="insert_many",
        client=None,
        client_kwargs=None,
        transient_client=False):

    # try:
    if client is None:
        client = generate_client(client_kwargs=client_kwargs)

    insert_fn = getattr(client[database][collection], write_type)

    return insert_fn(documents)


def delete(database,
        collection,
        query,
        write_type="delete_many",
        client=None,
        client_kwargs=None,
        transient_client=False):

    # try:
    if client is None:
        client = generate_client(client_kwargs=client_kwargs)

    delete_fn = getattr(client[database][collection], write_type)

    return delete_fn(query)

