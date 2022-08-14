import streamlit as st
import psycopg2
import pandas as pd

conn_kwargs = dict(
    host='postgres',
    port=5432,
    user='mlflow_user',
    password='mlflow_password',
    dbname='mlflow'
)
connection = psycopg2.connect(**conn_kwargs)


describe_tables_query = """
SELECT *
FROM pg_catalog.pg_tables
WHERE schemaname != 'pg_catalog' AND 
    schemaname != 'information_schema';
"""

table_info = pd.read_sql_query(describe_tables_query,con=connection)

st.write("# Tables")
st.write(table_info)
st.write("Enter Query")

table = st.selectbox("Select a Table", table_info['tablename'].tolist(), 0)
default_query = f"""
SELECT * FROM {table}

"""
query = st.text_area("Query", default_query)

df = pd.read_sql_query(query,
                       con=connection)

st.write(df)
connection.close()

