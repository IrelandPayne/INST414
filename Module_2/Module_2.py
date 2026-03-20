import pandas as pd 
import networkx as nx 

#reading in tsv
df = pd.read_csv("Reddit_Data.tsv", sep='\t') #tab seperated file

# directed graph
graph = nx.DiGraph()

#creating nodes

for index, row in df.iterrows():
   source = row["SOURCE_SUBREDDIT"]
   target = row["TARGET_SUBREDDIT"]
   sentiment = row["LINK_SENTIMENT"]

   if graph.has_edge(source, target):
        graph[source][target]["weight"] += 1
   else: 
        graph.add_edge(source, target, weight= 1, sentiment=sentiment)

#counting edges of nodes, sorting them to the top 5 based on incoming/outgoing
sorted_incoming_nodes = sorted(graph.in_degree(), key=lambda x:x[1], reverse=True)[:10]
sorted_outgoing_nodes = sorted(graph.out_degree(), key=lambda x:x[1], reverse=True)[:10]

in_degree_centrality = nx.in_degree_centrality(graph)

#not needed for this analysis, but possibly helpful for future project
#top_in_degree = sorted(in_degree_centrality.items(), key=lambda x: x[1], reverse=True)

out_degree_centrality = nx.out_degree_centrality(graph)

#not needed for this analysis, but possibly helpful for future project
#top_out_degree = sorted(out_degree_centrality.items(), key=lambda x: x[1], reverse=True)

betweenness = nx.betweenness_centrality(graph, k=100) #approximate betweenness
top_btwn = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]

eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
top_eigen = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:10]

#print statements; printing results of "important" nodes based on connections and centrality
print(f"\nSubreddits with the most incoming edges and their centrailty measures {'*' * 10} ")
rows = []

for node, edge in sorted_incoming_nodes:
    rows.append({
        "Subreddit": node,
        "Incoming Links": edge,
        "In-Degree Centrality": round(in_degree_centrality[node], 4),
        "Betweenness": round(betweenness[node], 4),
        "Eigenvector": round(eigenvector[node], 4)
    })

df_table = pd.DataFrame(rows)

print(df_table)

print(f"\nSubreddits with the most outgoing edges and their centrailty measures {'*' * 10} ")
out_rows = []

for node, edge in sorted_outgoing_nodes:
    out_rows.append({
        "Subreddit": node,
        "Outgoing Links": edge,
        "Out-Degree Centrality": round(out_degree_centrality[node], 4),
        "Betweenness": round(betweenness[node], 4),
        "Eigenvector": round(eigenvector[node], 4)
    })

out_df = pd.DataFrame(out_rows)

print(out_df)
# measures for top in/out degree centrality. repetitive to include but helpful to examine on its own
# print(f"\nTop in degree centrality nodes {'*' * 10}")
# for node, score in top_in_degree:
#     print(f"{node} with an in degree centrality of {score:.4f}")
    
# print(f"\nTop out degree centrality nodes {'*' * 10}")
# for node, score in top_out_degree:
#     print(f"{node} with an out degree centrality of {score:.4f}")

#eigenvector centrality
print(f"\nSubreddits with the highest eigenvector centrailty {'*' * 10} ")
eig_rows = []

for node, score in top_eigen:
    eig_rows.append({
        "Subreddit": node,
        "Eigenvector Centrality": round(score, 4)
    })

eig_df = pd.DataFrame(eig_rows)

print(eig_df)

#between centrality
print(f"\nSubreddits with the highest betweenness centrailty {'*' * 10} ")
btwn_rows = []

for node, score in top_btwn:
    btwn_rows.append({
        "Subreddit": node,
        "Betweenness Centrality": round(score, 4)
    })

btwn_df = pd.DataFrame(btwn_rows)

print(btwn_df)

#graph centered around askreddit node
#2337 nodes 37996 edges
ego_askreddit = nx.ego_graph(graph, 'askreddit', radius=1, undirected=True)

for node in ego_askreddit.nodes():
    ego_askreddit.nodes[node]['In_Centrality'] = float(in_degree_centrality.get(node, 0))
    ego_askreddit.nodes[node]['Out_Centrality'] = float(out_degree_centrality.get(node, 0))
    ego_askreddit.nodes[node]['Betweenness'] = float(betweenness.get(node, 0))
    ego_askreddit.nodes[node]['Elite_Score'] = float(eigenvector.get(node, 0))

nx.write_graphml(ego_askreddit, "ask_reddit.graphml")