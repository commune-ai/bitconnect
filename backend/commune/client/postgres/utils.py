import psycopg2
import itertools
import os
import psycopg2.extras as extras
from io import StringIO
from sqlalchemy import create_engine


def insert_df(conn,
              df,
              table,
              upsert=True):
    """
    Using cursor.mogrify() to build the bulk insert query
    then cursor.execute() to execute the query
    """
    # Create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    num_columns = len(df.columns)
    # Comma-separated dataframe columns
    cols = ','.join([f'"{col}"' for col in df.columns])
    # SQL quert to execute
    cursor = conn.cursor()

    values = ','.join([cursor.mogrify(f"({','.join(num_columns * ['%s'])})".encode('utf8'), tup).decode('utf8')
              for tup in tuples])

    query = f'INSERT INTO {table}({cols}) VALUES{values}'

    if upsert:
        upsert_action = 'UPDATE SET' + ','.join([f'"{col}" = EXCLUDED."{col}"' for col in df.columns])
        query += f' ON CONFLICT ON CONSTRAINT {table}_pkey DO {upsert_action}'


    try:
        cursor.execute(query)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    cursor.close()



def dictionary_insert_value_converter(input_dict):
    """
    converts dictionary to respective insert query
    """

    output_list = []

    for k,v in input_dict.items():

        if isinstance(v, dict):
            output_list.append(list(v.values()))
        else:
            output_list.append([v])

    output_list = list(itertools.chain(*output_list))

    return tuple(output_list)



def insert_list_dict(
                     input_list,
                     table,
                     con=None,
                     con_kwargs=None,
                     upsert_key=None,
                     transient_con=False
                     ):


    """
    Using cursor.mogrify() to build the bulk insert query
    then cursor.execute() to execute the query
    """

    if con is None:
        assert con_kwargs is not None
        con = psycopg2.connect(**con_kwargs)

    # Create a list of tupples from the dataframe values

    # SQL quert to execute
    cursor = con.cursor()

    query_prefix= "INSERT INTO {} ({}) VALUES".format(
        table,
        '"'+'", "'.join(input_list[0].keys())+'"'
    )

    query_value_format = "({})".format(', '.join([f"ROW({', '.join(['%s']*len(value))})"
                                       if isinstance(value, dict) else '%s'
                                       for value in input_list[0].values()]))

    query_value = ', '.join(str(cursor.mogrify(query_value_format, dictionary_insert_value_converter(v)), "utf-8") for v in input_list)

    query = " ".join([query_prefix,query_value])


    if upsert_key:
        upsert_query = 'ON CONFLICT ({}) DO UPDATE SET {};'.format(
            upsert_key,
            ", ".join([f'"{col}" = {table}."{col}"'for col in input_list[0].keys() if col != upsert_key])
        )

    query = " ".join([query,upsert_query])
    # cursor.execute(query)
    try:
        cursor.execute(query)
        con.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        con.rollback()
        cursor.close()

    cursor.close()

    if transient_con:
        con.close()


def execute_many(conn, df, table):
    """
    Using cursor.executemany() to insert the dataframe
    """
    # Create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    # SQL quert to execute
    query = "INSERT INTO %s(%s) VALUES(%%s,%%s,%%s)" % (table, cols)
    cursor = conn.cursor()
    try:
        cursor.executemany(query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("execute_many() done")
    cursor.close()


def execute_batch(conn, df, table, page_size=100):
    """
    Using psycopg2.extras.execute_batch() to insert the dataframe
    """
    # Create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    # SQL quert to execute
    query = "INSERT INTO %s(%s) VALUES(%%s,%%s,%%s)" % (table, cols)
    cursor = conn.cursor()
    try:
        extras.execute_batch(cursor, query, tuples, page_size)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("execute_batch() done")
    cursor.close()


def execute_values(conn, df, table):
    """
    Using psycopg2.extras.execute_values() to insert the dataframe
    """
    # Create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    # SQL quert to execute
    query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
    cursor = conn.cursor()
    try:
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("execute_values() done")
    cursor.close()


def execute_mogrify(conn, df, table):
    """
    Using cursor.mogrify() to build the bulk insert query
    then cursor.execute() to execute the query
    """
    # Create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    # SQL quert to execute
    cursor = conn.cursor()
    values = [cursor.mogrify("(%s,%s,%s)", tup).decode('utf8') for tup in tuples]
    query = "INSERT INTO %s(%s) VALUES " % (table, cols) + ",".join(values)

    try:
        cursor.execute(query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    cursor.close()


def copy_from_file(conn, df, table):
    """
    Here we are going save the dataframe on disk as
    a csv file, load the csv file
    and use copy_from() to copy it to the table
    """
    # Save the dataframe to disk
    tmp_df = "./tmp_dataframe.csv"
    df.to_csv(tmp_df, index_label='id', header=False)
    f = open(tmp_df, 'r')
    cursor = conn.cursor()
    try:
        cursor.copy_from(f, table, sep=",")
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        os.remove(tmp_df)
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    cursor.close()
    os.remove(tmp_df)


def copy_from_stringio(conn, df, table):
    """
    Here we are going save the dataframe in memory
    and use copy_from() to copy it to the table
    """
    # save dataframe to an in memory buffer
    buffer = StringIO()
    df.to_csv(buffer, index_label='id', header=False)
    buffer.seek(0)

    cursor = conn.cursor()
    try:
        cursor.copy_from(buffer, table, sep=",")
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("copy_from_stringio() done")
    cursor.close()


# ----------------------------------------------------------------
# SqlAlchemy Only
# ----------------------------------------------------------------


def to_alchemy(df,conn):
    """
    Using a dummy table to test this call library
    """
    engine = create_engine(conn)
    df.to_sql(
        'test_table',
        con=engine,
        index=False,
        if_exists='replace'
    )
    print("to_sql() done (sqlalchemy)")
