import torch
import networkx as nx
import matplotlib.pyplot as plt
import os


def plot_graph(pos, edge_index, index):
    # Create a NetworkX graph from edge_index
    os.makedirs('plots', exist_ok = True)
    G = nx.Graph()
    edge_list = edge_index.T.tolist()
    G.add_edges_from(edge_list)

    # Plot the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos.numpy(), with_labels=True, node_size=300, node_color='skyblue', font_size=10, font_color='black')
    plt.title('Graph Plot')
    plt.savefig(f'plots/graph_plot_{index}.png')
    plt.close()

    # Get connected components
    components = list(nx.connected_components(G))

    # Plot nodes of different connected components
    plt.figure(figsize=(8, 8))
    colors = ['r', 'g', 'b', 'y', 'c', 'm']  # You can extend this list as needed
    for i, component in enumerate(components):
        component_pos = pos[list(component)].numpy()
        plt.scatter(component_pos[:, 0], component_pos[:, 1], color=colors[i % len(colors)], label=f"Component {i}")

    plt.title('Nodes of Different Connected Components')
    plt.legend()
    plt.savefig(f'plots/connected_components_{index}.png')
    plt.close()

    # Calculate the degree matrix
    degrees = torch.zeros(pos.size(0))
    for node in G.nodes():
        degrees[node] = G.degree[node]
    degree_matrix = torch.diag(degrees)

    # Print diagonal elements of the degree matrix
    print("Diagonal Elements of Degree Matrix:")
    print(torch.diag(degree_matrix))
    print('max degree', torch.max(torch.diag(degree_matrix)))
    print('min degree', torch.min(torch.diag(degree_matrix)))
    print('mean degree', torch.mean(torch.diag(degree_matrix)))
    print('edge_index.shape', edge_index.shape)

    # Plot histogram of degree distribution
    plt.figure(figsize=(8, 6))
    plt.hist(degrees.numpy(), bins=20, color='skyblue', edgecolor='black')
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.savefig(f'plots/degree_distribution_{index}.png')
    plt.close()

    return degree_matrix