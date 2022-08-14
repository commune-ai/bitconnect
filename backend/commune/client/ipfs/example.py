#
import os, sys
sys.path[0] = os.getcwd()
from commune.client.ipfs import IPFSManager

if __name__ == "__main__":
    client = IPFSManager()
    import pandas as pd

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Contestant": ["Alex", "Alex", "Alex", "Jordan", "Jordan", "Jordan"],
        "Number Eaten": [2, 1, 3, 1, 3, 2],
    })

    # Plotly Express

    import plotly.express as px

    fig = px.bar(df, x="Fruit", y="Number Eaten", color="Contestant", barmode="group")
    fig.show()

    import plotly.graph_objects as go

    fig = go.Figure()
    for contestant, group in df.groupby("Contestant"):
        fig.add_trace(go.Bar(x=group["Fruit"], y=group["Number Eaten"], name=contestant,
                             hovertemplate="Contestant=%s<br>Fruit=%%{x}<br>Number Eaten=%%{y}<extra></extra>" % contestant))
    fig.update_layout(legend_title_text="Contestant")
    fig.update_xaxes(title_text="Fruit")
    fig.update_yaxes(title_text="Number Eaten")

    json_object = fig.to_json()

    # hasg = client.write_json(json_object)
    # client.read_json(hash)

    {'name', 'explainable '}

    hash = client.add_json(json_object)
    ipfs_return_dict = client.get_json(hash)
    print(type(ipfs_return_dict), ipfs_return_dict, hash)
