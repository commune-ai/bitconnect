import os
import sys
sys.path.append(os.environ['PWD'])
import re
import multiprocessing as mp
from pymongo import MongoClient, UpdateOne
from commune.utils.misc import chunk
import itertools
from commune.streamlit.utils import describe


from commune.utils.misc import dict_put, dict_get
from commune.ray.actor import ActorBase
from commune.ray.utils import create_actor
from commune.client.mongo.utils import (MongoConnection,
                    generate_client,
                    update,
                    insert,
                    find_query, delete)
import ray


class MongoManager(ActorBase):
    default_cfg_path = f"client.mongo.manager"
    def __init__(self, cfg=None ):
        ActorBase.__init__(self, cfg)
        self.client_kwargs = self.cfg['client_kwargs']
        self.client_kwargs['port'] = int(self.client_kwargs['port'])

        self.client = MongoClient(**self.client_kwargs)

    def clearCache(self, database, collection):
        self.client[database].command(
    {
        "planCacheClear": collection
    }
)
    def show_db_collection(self):
        '''
        Shows the db and collection structure
        return Dict{db_name:str, List[collection_name]}
        '''

        show_dict = {}

        for db_name in self.con.client.list_database_names():
            db = self.client[db_name]

            show_dict[db_name] = db.list_collection_names()

        return show_dict

    def generate_client(self, client_kwargs):
        generate_client(client_kwargs=client_kwargs)

    def has_database(self,database):
        '''
        check if client exists
        '''
        assert isinstance(database, str)
        return bool(database in self.client.list_database_names())

    def execute(self,
                database,
                collection,
                command,
                full_command=False):
        """"""

        collection = self.client[database][collection]
        if not full_command :
            command = '.'.join(['collection', command])
        return eval(command)


    def has_collection(self,database, collection):
        '''
        Check if client has colleciton
        '''
        assert isinstance(database, str) and \
            isinstance(collection, str)

        db = self.client[database]

        return bool(collection in db.list_collection_names())


    def list_database(self, database:str=".*"):
        return [db for db in self.client.list_database_names() if re.match(database, db)]

    def list_collection(self,
                        database: str,
                        collection: str=""
                        ):
        """
        string to match database
        uses regex expressions to filter collections
        """
        db = self.client[database]
        return [c for c in db.list_collection_names() if re.match(collection, c)]

    def find(self,
               collection,
               database,
               query={},
               projection= {},
               workers=1
               ):
        """

        update list of documents
        """



        if workers == 1:
            # use the default client

             documents =find_query(collection=collection,
                       database=database,
                       query=query,
                       projection=projection,
                       skip_n=0,
                       limit_n=1000,
                       client = self.client,
                       client_kwargs=None,
                       transient_client=False)
        elif workers > 1:
            collection_size = self.client[database][collection].count()
            # get batch_size
            batch_size = max(round((collection_size / workers) + 0.5), 1)

            # get skips
            skips = range(0, workers * batch_size, batch_size)

            # initiate a new client per worker
            find_fn = ray.remote(find_query)
            running_jobs = [find_fn.remote(collection=collection,
                                           database=database,
                                           skip_n=skip,
                                           limit_n=batch_size,
                                           query=query,
                                           projection=projection,
                                           client = None,
                                           client_kwargs=self.client_kwargs,
                                           transient_client=True) for skip in skips]
            total_finished_jobs = []
            while running_jobs:
                finished_jobs, running_jobs = ray.wait(running_jobs)
                if len(finished_jobs) > 0:
                    total_finished_jobs.extend(finished_jobs)

            documents = list(itertools.chain(*ray.get(total_finished_jobs)))

        return documents


    # def __del__(self):
    #     self.client.close()


    def update(self,
               collection,
               database,
               updates,
               write_type= "bulk_write",
               workers=1
               ):
        """

        update list of documents
        """

        if isinstance(updates, dict):
            updates = [updates]

        if workers == 1:
            # use the default client

            update(collection=collection,
                   database=database,
                   updates=updates,
                   write_type= write_type,
                   client = self.client,
                   client_kwargs=None,
                   transient_client=False)
        elif workers > 1:

            # initiate a new client per worker
            chunked_updates = chunk(updates,max(len(updates) // workers,1))
            insert_fn = ray.remote(update)
            running_jobs = [insert_fn.remote(collection=collection,
                                           database=database,
                                           updates=updates_chunk,
                                           write_type= write_type,
                                           client = None,
                                           client_kwargs=self.client_kwargs,
                                           transient_client=True) for updates_chunk in chunked_updates]

            while running_jobs:
                finished_jobs, running_jobs = ray.wait(running_jobs)

            ray.get(finished_jobs)


    def save(self, collection:str, database:str, data:dict={}, query:dict={}, root_key='', workers:int=1, upsert:bool=True, add_query:bool=True):

        if root_key:
            root_data = {}
            dict_put(input_dict=root_data, keys=root_key, value=data)
            data = root_data
        if add_query:
            data.update(query)

        
        updates = [{'filter': query,
            'update': {'$set': data},
            'upsert': True}]

        if query is None:
            return 

        self.update(collection=collection,
                                    database=database,
                                    updates=updates,
                                    workers=workers)

    def load(self, collection, database, query={},projection={}, workers=1, root_key='', return_one=False, remove_query=False, remove_id=True):
        """

        Loads colleciton based on query filter
        note that this works for nested queries if remove_query_fields== False as (True) is not supported


        """

        documents = self.find(collection=collection, 
                        database=database,  
                        query=query,
                        projection=projection,
                        workers=workers)

        for i in range(len(documents)):
            if remove_id:
                if '_id' in documents[i]:
                    del documents[i]['_id']
            if remove_query:
                for k in query:
                    del documents[i][k]

            if root_key:
                documents[i] = dict_get(input_dict=documents[i], keys=root_key)

            if return_one:
                return documents[i]

    


        return documents


    def insert(self,
               collection,
               database,
               documents,
               write_type = "insert_many",
               workers=1
               ):
        """

        update list of documents
        """
        if workers == 1:
            # use the default client

            insert(collection=collection,
                   database=database,
                   documents=documents,
                   write_type= write_type,
                   client = self.client,
                   client_kwargs=None,
                   transient_client=False)
        elif workers > 1:

            # initiate a new client per worker
            chunked_documents = chunk(documents,len(documents) // workers)
            insert_fn = ray.remote(insert)
            running_jobs = [insert_fn.remote(collection=collection,
                                           database=database,
                                           documents=documents_chunk,
                                           write_type= write_type,
                                           client = None,
                                           client_kwargs=self.client_kwargs,
                                           transient_client=True) for documents_chunk in chunked_documents]

            while running_jobs:
                finished_jobs, running_jobs = ray.wait(running_jobs)

            ray.get(finished_jobs)


    # this is a temp fix to get attributes from a given actor
    def get(self, item):
        if item is None:
            return self.__dict__
        else:
            return getattr(self,item)



    @classmethod
    def create_actor(cls, **kwargs):
        create_actor(cls=cls, **kwargs)

    def read(self, *args, **kwargs):
        self.load(*args, **kwargs)

    def write(self, *args, **kwargs):
        self.save(*args, **kwargs)



    def delete_database(self,
                        database: str
                        ):

        '''
        drop client
        '''
        assert database in self.client.list_database_names()

        # dont delete this hit
        assert database not in ['admin', 'config', 'local']

        self.client.drop_database(database)

    def delete_collection(self,
                          database: str,
                          collection: str,
                          verbose=False):
        '''
        drop collection

        '''

        self.client[database][collection].drop()
        if verbose:
            print(f'Mongo Manager: Dropped {col_name}')


    def collection(self, database, collection):
        return self.client[database][collection]

    def database(self, database):
        return self.client[database]

    def delete(self,
               collection:str,
               database:str,
               query:dict,
               write_type:str = "delete_many",
               ):
        """

        update list of documents
        """

        return delete(collection=collection,
                database=database,
                query=query,
                write_type= write_type,
                client = self.client,
                client_kwargs=None,
                transient_client=False)

if __name__ == '__main__':
    import streamlit as st
    mongo = MongoManager()
    dir(mongo)

    st.write()
    doc = {'whadup': 'fam', 'tag': 'bro'}
    mongo.write(database='demo', collection='demo', data=doc , query=doc )
    # st.write(mongo.find(database='demo', collection='demo',query={'whadup': 'fa*'} ))
    
    # mongo.delete(collection='demo', database='demo', query=doc)
    
    st.write(list(mongo.client.demo.demo.find()))

