import networkx as nx
import matplotlib.pyplot as plt

N = 100
K = 6

# Random
G_random = nx.gnm_random_graph(N, N*K)

# Scale free
G_sf = nx.barabasi_albert_graph(N, 3)

# Small world
G_sw = nx.watts_strogatz_graph(N, K, 0.02)

fig, axs = plt.subplots(1,3, figsize=(12,4))

nx.draw(G_random, node_size=20, ax=axs[0])
axs[0].set_title("Random Network")

nx.draw(G_sf, node_size=20, ax=axs[1])
axs[1].set_title("Scale-Free Network")

nx.draw(G_sw, node_size=20, ax=axs[2])
axs[2].set_title("Small-World Network")

plt.tight_layout()
plt.savefig("network_topologies.png", dpi=300)
