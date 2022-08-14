import psycopg2
import os
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from copy import deepcopy
from commune.utils.misc import chunk
import pandas as pd
from commune.ray import ActorBase
from .utils import insert_list_dict, insert_df

from sqlalchemy import create_engine
import ray

@ray.remote
def query_worker(query,con_kwargs):
    self.con = psycopg2.connect(**self.con_kwargs)
    self.con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    return pd.read_sql_query(sql=query, con=self.con)



class PostgresManager(ActorBase):
    default_cfg_path = f"{os.environ['PWD']}/commune/config/client/block/postgres.yaml"
    def __init__(self,cfg):
        '''

        Initialize the Postgres Mananger

            host='localhost',
            port=5432,
            user='mlflow_user',
            password='mlflow_password',
            dbname='mlflow'
        '''

        self.con_kwargs = cfg['con_kwargs']
        self.host = self.con_kwargs['host']
        self.username = self.con_kwargs['port']
        self.user = self.con_kwargs['user']
        self.password = self.con_kwargs['password']
        self.dbname = self.con_kwargs['dbname']
        self.port = self.con_kwargs['port']
        self.con = psycopg2.connect(**self.con_kwargs)
        self.con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        self.engine = self.create_alchemy_engine()
    # def connect(self, dbname=None):


    def create_alchemy_engine(self):
        return create_engine(f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}')
    def create_table_from_pandas(self, df, table_name, index=False):
        engine = self.create_alchemy_engine()
        df.to_sql(table_name, engine, index=index)
        # type2pgtype_map = {
        #     float: "real"
        #     int: "bigint"
        # }

        # create_table_script = f"""
        # CREATE TABLE {table_name}
        #         (
        #         eid integer NOT NULL,
        #         f_i INTEGER NULL,
        #         f_ia INTEGER[] NULL,
        #         f_iaa INTEGER[][] NULL,
        #         f_d DATE NULL,
        #         f_da DATE[] NULL,
        #         f_daa DATE[][] NULL,
        #         PRIMARY KEY(eid)
        #         );

        
        # """

    

    def query(self, query, output_pandas=False):
        if output_pandas:
            if isinstance(query,list):  
                return ray.get([query_worker(q,con_kwargs) for q in query])
        
            return pd.read_sql_query(sql=query, con=self.con) 
        else:
            return self.execute(query=query, fetch='all')

    

    def execute(self, query, fetch=None):

        cursor = self.con.cursor()
        try:
            cursor.execute(query)

            if fetch == 'all':
                return cursor.fetchall()
            elif fetch == 'one':
                return cursor.fetchOne()

            assert fetch in ['all', 'one', None]
            self.con.commit()

        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            self.con.rollback()
            cursor.close()
            return None



    def schema(self, table_name):

        return pd.read_sql_query(f"""
                                    SELECT 
                                       table_name, 
                                       column_name, 
                                       data_type 
                                    FROM 
                                       information_schema.columns
                                    WHERE 
                                        table_name = '{table_name}';
                                    """, self.engine)

    def columns(self, table_name):
        return list(pd.read_sql_query(f'''select *
        from {table_name}
        where false;''', con=self.create_alchemy_engine()).columns)


        # commit the query


    def insert_df(self, table, df, upsert=True):
        insert_df(conn=self.con,
                  df=df,
                  table=table,
                  upsert=upsert)

    def insert(self,table,
               documents,
               upsert_key=None,
               workers=1):

        if workers > len(documents):
            workers = 1

        if workers == 1:
            insert_list_dict(con=self.engine,
                             table=table,
                             input_list=documents,
                             upsert_key=upsert_key,
                             transient_con=False)
        elif workers > 1:
            chunked_documents = chunk(documents, len(documents) // workers)
            insert_fn = ray.remote(insert_list_dict)

            running_jobs = [insert_fn.remote(con=None,
                                             table=table,
                                             upsert_key=upsert_key,
                                             con_kwargs=self.con_kwargs,
                                             input_list=documents_chunk,
                                             transient_con=True) for documents_chunk in chunked_documents]

            while running_jobs:
                finished_jobs, running_jobs = ray.wait(running_jobs)

            ray.get(finished_jobs)

    def count(self, table):
        if self.table_exists(table):
            return self.query(f'SELECT COUNT(*) FROM {table}', output_pandas=False )[0][0]
        else:
            return 0  
    def table_exists(self, table):
        return bool(table in self.tables)

    @property
    def tables(self):
        tables =  pd.read_sql_query("""
                                SELECT DISTINCT(tablename)
                                FROM pg_catalog.pg_tables
                                WHERE schemaname != 'pg_catalog' AND 
                                    schemaname != 'information_schema';
                                """, con=self.engine)['tablename'].tolist()

        return tables


    @property
    def custom_types(self):
        # list the custom types
        return \
                pd.read_sql_query("""
                        SELECT      n.nspname as schema, t.typname as type 
                        FROM        pg_type t 
                        LEFT JOIN   pg_catalog.pg_namespace n ON n.oid = t.typnamespace 
                        WHERE       (t.typrelid = 0 OR (SELECT c.relkind = 'c' FROM pg_catalog.pg_class c WHERE c.oid = t.typrelid)) 
                        AND     NOT EXISTS(SELECT 1 FROM pg_catalog.pg_type el WHERE el.oid = t.typelem AND el.typarray = t.oid)
                        AND     n.nspname NOT IN ('pg_catalog', 'information_schema');
                        """, self.engine)

    def delete_table(self,table):
        self.execute(f'''DROP TABLE IF EXISTS {table} CASCADE ''', fetch=None)


    def create_database(self, database):
        self.execute(f'CREATE DATABASE {database}', fetch=None)


    def __del__(self):
        if hasattr(self, 'con'):
            self.con.close()
        if hasattr(self, 'engine'):
            del self.engine


    def write_pandas_table(self,
                            df,
                            table_name,
                            refresh_table,
                            primary_key):
        """
        params:
            df: pandas df
            table_name: name of pandas dataframe
            refresh_table: boolean of whether to rewrite table
            primary_key: stirng of primary key
        """

        if self.table_exists(table_name):
            if refresh_table:
                print("BEFORE DROP",table_name, self.table_exists(table_name))
                self.delete_table(table_name)

                print("AFTER DROP",table_name, self.table_exists(table_name))
                self.create_table_from_pandas(table_name=table_name, df=df)
                self.execute(f'ALTER TABLE {table_name} ADD PRIMARY KEY ("{primary_key}");')
            else:
                self.insert_df(table=table_name,df=df,upsert=True)
        else:
            self.create_table_from_pandas(table_name=table_name, df=df)
            self.execute(f'ALTER TABLE {table_name} ADD PRIMARY KEY ("{primary_key}");')



    def write_pandas_table_2(self,
                            df,
                            table_name,
                            primary_key,
                            refresh_table=False,):
        """
        params:
            df: pandas df
            table_name: name of pandas dataframe
            refresh_table: boolean of whether to rewrite table
            primary_key: stirng of primary key
        """
        engine = self.create_alchemy_engine()
        if refresh_table:
            df.to_sql(name=table_name, con=engine,if_exists="replace", index_label=primary_key)
        else:
            df.to_sql(name=table_name, con=engine,if_exists="append")