import networkx as nx
import matplotlib.pyplot as plt
import json

config = json.load(open("config.json", "r"))

# construct DAG
dag = nx.DiGraph()

# add info
for i in range(config["num_stages"]):
    dag.add_node(config[str(i)]['stage_name'], size = 2000)
    for j in config[str(i)]['children']:
        dag.add_edge(config[str(i)]['stage_name'], config[str(j)]['stage_name'], weight = 3)
        
node_sizes = [data["size"] for _, data in dag.nodes(data=True)]
edge_widths = [data["weight"] for _, _, data in dag.edges(data=True)]

# plot
pos = nx.spring_layout(dag)
nx.draw_networkx(dag, pos=pos, with_labels=True, node_size=node_sizes, width=edge_widths)  # 绘制节点和边

plt.savefig("workflow.pdf", format="pdf")