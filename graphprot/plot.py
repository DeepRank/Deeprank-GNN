import plotly.offline as py
import plotly.graph_objs as go
from ResidueGraph import ResidueGraph

pdb = 'data/pdb/1ATN.pdb'
pssm = {'A':'./data/pssm/1ATN.A.pdb.pssm','B':'./data/pssm/1ATN.B.pdb.pssm'}
graph = ResidueGraph(pdb,pssm)

Xe,Ye,Ze = [],[],[]
for e in graph.edge_index:

	xyz1 = graph.xyz[e[0],:]
	xyz2 = graph.xyz[e[1],:]

	Xe += [xyz1[0],xyz2[0],None]
	Ye += [xyz1[1],xyz2[1],None]
	Ze += [xyz1[2],xyz2[2],None]

Xi, Yi, Zi = [], [] ,[]
for e in graph.internal_edge_index:
	xyz1 = graph.xyz[e[0],:]
	xyz2 = graph.xyz[e[1],:]

	Xi += [xyz1[0],xyz2[0],None]
	Yi += [xyz1[1],xyz2[1],None]
	Zi += [xyz1[2],xyz2[2],None]

Xn = [xyz[0] for xyz in graph.xyz]
Yn = [xyz[1] for xyz in graph.xyz]
Zn = [xyz[2] for xyz in graph.xyz]


labels, group = [], []
chain = {'A':0, 'B':1}
for ninfo in graph.node_info:
	group.append(chain[ninfo[0]])
	labels.append(' '.join(ninfo[1:]))


trace1=go.Scatter3d(x=Xe,
               y=Ye,
               z=Ze,
               mode='lines',
               line=dict(color='rgb(125,125,125)', width=1),
               hoverinfo='none'
               )

trace2=go.Scatter3d(x=Xi,
               y=Yi,
               z=Zi,
               mode='lines',
               line=dict(color='rgb(0,0,0)', width=2),
               hoverinfo='none'
               )

trace3=go.Scatter3d(x=Xn,
               y=Yn,
               z=Zn,
               mode='markers',
               name='actors',
               marker=dict(symbol='dot',
                             size=6,
                             color=group,
                             colorscale='Viridis',
                             line=dict(color='rgb(50,50,50)', width=0.5)
                             ),
               text=labels,
               hoverinfo='text'
               )

axis=dict(showbackground=False,
          showline=False,
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )

layout = go.Layout(
         title="Network of coappearances of characters in Victor Hugo's novel<br> Les Miserables (3D visualization)",
         width=1000,
         height=1000,
         showlegend=False,
         scene=dict(
             xaxis=dict(axis),
             yaxis=dict(axis),
             zaxis=dict(axis),
        ),
     margin=dict(
        t=100
    ),
    hovermode='closest',
    annotations=[
           dict(
           showarrow=False,
            text="Data source:",
            xref='paper',
            yref='paper',
            x=0,
            y=0.1,
            xanchor='left',
            yanchor='bottom',
            font=dict(
            size=14
            )
            )
        ],    )

data=[trace2, trace3]
fig=go.Figure(data=data, layout=layout)

py.plot(fig, filename='graph.html')