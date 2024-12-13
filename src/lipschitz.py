import numpy as np
import networkx as nx
import multiprocessing as mp

def sssplr(G, nodes):
    result = {}
    for n in nodes:
        result[n] = nx.single_source_dijkstra_path_length(G, n, cutoff=None, weight='haversine')
    return result

def lipschitz_node_embeddings1(G, nodes, k):
    # G_temp = G.reverse(copy=True)
    G_temp = G
    anchor_nodes = np.random.choice(nodes, size=k, replace=False)
    
    lips_dist_list = []
    for anchor_node in anchor_nodes:
        dist_dict = sssplr(G_temp, [anchor_node])
        lips_dist_list.append(dist_dict)

    lips_dist = {}
    for d in lips_dist_list:
        lips_dist.update(d)
    
    embeddings = np.zeros((len(nodes), k))
    for i, node_i in enumerate(anchor_nodes):
        sd = lips_dist[node_i]
        for j, node_j in enumerate(nodes):
            dist = sd.get(node_j, -1)
            if dist != -1:
                embeddings[node_j, i] = 1 / (dist + 1)
    
    embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)
    
    return embeddings

def lifshitz_embedding(edge_index, edge_weight, k):
    # Convert tensors to NumPy arrays
    edge_index = edge_index.cpu().numpy()
    edge_weight = edge_weight.cpu().numpy()
    
    # Create a graph
    # G = nx.DiGraph()
    G = nx.Graph()
    
    # Add nodes from edge_index
    num_nodes = edge_index.max() + 1
    G.add_nodes_from(range(num_nodes))
    
    # Add edges with weights from edge_weight
    for edge_idx, weight in zip(edge_index.T, edge_weight):
        i, j = edge_idx
        G.add_edge(i, j, haversine=weight)
    
    
    # Call lipschitz_node_embeddings for further processing
    nodes = list(range(num_nodes))
    embeddings = lipschitz_node_embeddings1(G, nodes, k)
    
    return embeddings

def lipschitz_node_embeddings(G, nodes, k):
    G_temp = G
    #G_temp = G.reverse(copy=True)
    anchor_nodes = np.random.choice(nodes, size=k, replace=False)
    num_workers = 16 if k > 16 else k
    results = []
    per_worker = k/num_workers
    pool = mp.Pool(processes=num_workers)
    for n in range(num_workers):
        start, end = int(per_worker*n), int(per_worker*(n+1))
        results.append(
                pool.apply_async(
                    sssplr, args=[
                        G_temp, anchor_nodes[start:end]]))
    lips_dist_list = [result.get() for result in results]
    pool.close()
    pool.join()
    lips_dist = {}
    for d in lips_dist_list:
        lips_dist.update(d)
    embeddings = np.zeros((len(nodes), k))
    for i, node_i in enumerate(anchor_nodes):
        sd = lips_dist[node_i]
        for j, node_j in enumerate(nodes):
            dist = sd.get(node_j, -1)
            if dist!=-1:
                embeddings[node_j, i] = 1/(dist+1)
    embeddings = (embeddings - embeddings.mean(axis=0))/embeddings.std(axis=0)
    return embeddings



































