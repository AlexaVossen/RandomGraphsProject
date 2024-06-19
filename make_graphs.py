<<<<<<< HEAD
import networkx as nx
import matplotlib.pyplot as plt
import pickle



def store_graphs(nr_graphs, avrg_deg, nr_nodes, filename):
    graphs = []
    for i in range(nr_graphs):
        G = nx.barabasi_albert_graph(nr_nodes, avrg_deg//2)
        graphs.append(nx.to_dict_of_dicts(G))

    with open(filename, 'wb') as f:
            pickle.dump(graphs, f)


def load_graphs(filename, create_using=nx.Graph):
    
    with open(filename, 'rb') as f:
        list_of_dicts = pickle.load(f)
    
    graphs = [create_using(graph) for graph in list_of_dicts]
    
    return graphs

degrees = [4, 10, 20, 30, 50, 100]
for d in degrees:
    store_graphs(100, d, 1000, f'graphs_{d}.pickle')


#graphs = load_graphs('graphs.pickle')


=======
import networkx as nx
import matplotlib.pyplot as plt
import pickle



def store_graphs(nr_graphs, avrg_deg, nr_nodes, filename):
    graphs = []
    for i in range(nr_graphs):
        G = nx.barabasi_albert_graph(nr_nodes, avrg_deg//2)
        graphs.append(nx.to_dict_of_dicts(G))

    with open(filename, 'wb') as f:
            pickle.dump(graphs, f)


def load_graphs(filename, create_using=nx.Graph):
    
    with open(filename, 'rb') as f:
        list_of_dicts = pickle.load(f)
    
    graphs = [create_using(graph) for graph in list_of_dicts]
    
    return graphs

degrees = [4, 10, 20, 30, 50, 100]
for d in degrees:
    store_graphs(100, d, 1000, f'graphs_{d}.pickle')


#graphs = load_graphs('graphs.pickle')


>>>>>>> 5e973388a52fb6c45ff92f33650ac77c38099c0e
