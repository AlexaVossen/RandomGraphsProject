import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import scipy
from scipy.stats import bernoulli
import pickle
from numpy import sqrt, std, mean
import time

t0 = time.time()

def create_underlying_network(d, D): 
    """Generate adjacency matrix of underlying scale-free network with d nodes and average degree D.
    """
    m = int(0.5*D)
    G = nx.barabasi_albert_graph(d, m)
    A = nx.to_numpy_array(G)
    return A

def matrix_to_graph(A): 
    """
    This function visualizes the associated graph given an adjaceny matrix.
    """
    G = nx.from_numpy_array(A)
    G.edges(data=True)
    return G

# graph1 = matrix_to_graph(create_underlying_network(10, 2))
# print(graphlisjt)

def bfs_new(graph, p):
    """Performs bfs. Returns a list of infected nodes per timestep, the total number of infected and time till death"""
    
    # list_yield = []
    # list_infected_over_time_total = []
    # time_till_death = []

    # for i in range(100):
    
    queue = []     #Initialize a queue
    visited = [] # List to keep track of visited nodes.
    list_infected_over_time = [1]
    spread = 1
    time_till_death_counter = 0
    diction = {0: 1}

    # nx.draw(graph, with_labels = True)
    # plt.show()

    nr_nodes = graph.number_of_nodes()
    node = np.random.randint(0, nr_nodes-1) #nog ff kijken
    # print('random node is', node)
    visited.append(node)
    queue.append((node, 0))

    while queue:
        s,d = queue.pop(0)
        # print (s, end = " ")

        if d+1 not in diction.keys():
            diction[d+1] = diction[d] 

        for neighbour in graph[s]:
            if neighbour not in visited:
                Y = bernoulli.rvs(p)
                diction[d+1] += Y
                visited.append(neighbour)
                if Y == 1:
                    queue.append((neighbour, d+1))
            list_infected_over_time.append(spread)
    # print('end')
    lijstje = [float(z)/1000 for z in list(diction.values())]
    time_till_death = list(diction.keys())[-1]
    return lijstje, lijstje[-1], time_till_death

# print(bfs_new(graph1, 1))

def load_graphs(filename, create_using=nx.Graph):
    
    with open(filename, 'rb') as f:
        list_of_dicts = pickle.load(f)
    
    graphs = [create_using(graph) for graph in list_of_dicts]
    
    return graphs

#lijst_allegraphs = load_graphs('graphs.pickle')
#lijst_allegraphs_deel = lijst_allegraphs[:10]
# graphje1 = lijst_allegraphs[0]

def run_bfs_onehunderedtimes(graphje, p):
    """Performs 100 bfs on each graph."""
    list_yield_100 = []
    list_infected_over_time_total_100 = []
    time_till_death_100 = []
    
    for i in range(100):
        one_result = bfs_new(graphje, p)
        list_yield_100.append(one_result[1])
        list_infected_over_time_total_100.append(one_result[0])
        time_till_death_100.append(one_result[2])
        i += 1

    return list_infected_over_time_total_100, list_yield_100, time_till_death_100

def bfs_experiment(lijst_graphs, p):
    """Runs bfs for all graphs in the lists with parameter p. 
    Returns the mean yield, time till death and ..."""
    experiment_list_yield = []
    experiment_list_ttd = []
    experiment_list_infectedlist = []

    for j in lijst_graphs:
        run = run_bfs_onehunderedtimes(j, p) 
        for k in run[1]:
            experiment_list_yield.append(k)
        for l in run[2]:
            experiment_list_ttd.append(l)
        for m in run[0]:
            experiment_list_infectedlist.append(m)

    return experiment_list_infectedlist, experiment_list_yield, experiment_list_ttd

def confidence_interval(lijst):
    """Returns CI"""
    n = len(lijst)
    lijst_avg = mean(lijst)
    Sn = std(lijst)
    halfwidth = 1.96*Sn/sqrt(n)
    return (lijst_avg - halfwidth, lijst_avg, lijst_avg + halfwidth)

def CI_over_time(listoflists):
    """
    Input: A list of lists containing the number of infected per time per graph. 
    Output: CI's for the number of infected per timestep.
    """
    average = []
    stds = []
    lengths = []
    dictionary = {}
    for singlelist in listoflists:
        for j in range(len(singlelist)):
            if j in dictionary:
                dictionary[j].append(singlelist[j])
            else:
                dictionary[j] = [singlelist[j]]
    for i, v in dictionary.items():
        average.append(mean(v))
        stds.append(std(v))
        lengths.append(len(v))
    list_of_CI = []
    for i in range(len(average)):
        halfwidth = 1.96*stds[i]/sqrt(lengths[i])
        lowerbound_CI = average[i]-halfwidth
        upperbound_CI = average[i]+halfwidth
        list_of_CI.append((lowerbound_CI, average[i], upperbound_CI))
    return list_of_CI

lijstps = [0.05, 0.10, 0.15, 0.20, 0.25, 0.5, 0.75]
for p in lijstps:
    lijst_allegraphs = load_graphs(f'graphs_20.pickle')
    experiment = bfs_experiment(lijst_allegraphs, p)
    f = open("outputLiekeps.txt", "a")
    print(f'Time till death list {experiment[2]}', file=f)
    print(f'CI yield for p={p}:', confidence_interval(experiment[1]), file=f)
    print(f'CI ttd for p={p}', confidence_interval(experiment[2]), file=f)
    deel3 = CI_over_time(experiment[0])
    print(f'CI per time on nr infected for p={p}', deel3, file=f)
    f.close()

    lower_list = []
    mean_list = []
    upper_list = []
    for i in deel3:
        lower_list.append(i[0])
        mean_list.append(i[1])
        upper_list.append(i[2])

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(lower_list, linestyle='None', marker='o', label = "lower bounds CI") 
    ax1.plot(mean_list, linestyle='None', marker='o', label = "mean CI") 
    ax1.plot(upper_list, linestyle='None', marker='o', label = "upper bounds CI") 
    ax1.set_title(f'Fraction of infected nodes per time step with p={p} and E[D]=20')
    ax1.set(ylabel="Fraction of infected nodes")
    ax2.set(xlabel="Time steps",ylabel='Number of occurrences')
    ax1.legend() 
    ax2.hist(experiment[2], bins= [i+1 for i in range(max(experiment[2]))], color='skyblue', edgecolor='black')
    ax2.set_title('Histogram for the time till death of each run')
    fig.tight_layout()
    fig.savefig(f'spambot with p {p}.png')

# t1 = time.time()
# print('Time to run:', t1-t0)

lijstdegrees = [4, 10, 20, 30, 50, 100]
for degree in lijstdegrees:
    lijst_allegraphs = load_graphs(f'graphs_{degree}.pickle')
    experiment = bfs_experiment(lijst_allegraphs, 0.25)
    f = open("outputLieke.txt", "a")
    #print(f'Time till death list {experiment[2]}', file=f)
    print(f'CI yield for p=0.25 and average degree {degree}:', confidence_interval(experiment[1]), file=f)
    print(f'CI ttd for p=0.25 and average degree {degree}', confidence_interval(experiment[2]), file=f)
    deel3 = CI_over_time(experiment[0])
    print(f'CI per time on nr infected for p=0.25 and average degree {degree}', deel3, file=f)
    f.close()

    lower_list = []
    mean_list = []
    upper_list = []
    for i in deel3:
        lower_list.append(i[0])
        mean_list.append(i[1])
        upper_list.append(i[2])

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(lower_list, linestyle='None', marker='o', label = "lower bounds CI") 
    ax1.plot(mean_list, linestyle='None', marker='o', label = "mean CI") 
    ax1.plot(upper_list, linestyle='None', marker='o', label = "upper bounds CI") 
    ax1.set_title(f'Fraction of infected nodes per time step with p=0.25 and E[D]={degree}')
    ax1.set(ylabel="Fraction of infected nodes")
    ax2.set(xlabel="Time steps",ylabel='Number of occurrences')
    ax1.legend() 
    ax2.hist(experiment[2], bins= [i+1 for i in range(max(experiment[2]))], color='skyblue', edgecolor='black')
    ax2.set_title('Histogram for the time till death of each run')
    fig.tight_layout()
    fig.savefig(f'spambot with average degree {degree}.png')

t1 = time.time()

print('Time to run:', t1-t0)