import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import math

def plot_bundle(subplots, subplot_titles=[]):
    if isinstance(subplots, dict):
        subplot_titles = list(subplots.keys())
        subplots = list(subplots.values())

    plot_count  = len(subplots)
    rows = int(math.ceil(plot_count ** 0.5))
    cols = int(math.ceil(plot_count / rows)) 
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    for i, sub_fig in enumerate(subplots):
        row =int(math.ceil((i+1)/cols))
        col = i%cols+1

        for trace in sub_fig.data:
            fig.add_trace(trace, row=row, col=col )

    fig.update_yaxes(nticks=5)
    fig.update_xaxes(nticks=5)
    fig.update_layout(
                        autosize=False,
                        width=1000,
                        height=1000)
    return fig

