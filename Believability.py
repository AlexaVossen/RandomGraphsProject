<<<<<<< HEAD
import networkx as nx
import random
from make_graphs import load_graphs
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, std, mean

def TsmAlgo(G, MaxIt=1000):
    numb_of_nodes = len(list(G.nodes))
    TiTilde = [1]*numb_of_nodes
    TwTilde = [1]*numb_of_nodes
    for i in range(MaxIt):
        TiTilde_new = [0]*numb_of_nodes
        TwTilde_new = [0]*numb_of_nodes
        for node in G.nodes:
            for neighbor in G.neighbors(node):
                TiTilde_new[node] += 1/(1+ TwTilde[neighbor])**(0.5)
            for predecessor in list(G.predecessors(node)):
                TwTilde_new[node] += 1/(1+ TiTilde[predecessor])**(0.5) 
        TiTilde_Normalized = [TiTilde_new[j]/max(TiTilde_new) for j in range(numb_of_nodes)]
        TwTilde_Normalized = [TwTilde_new[j]/max(TwTilde_new) for j in range(numb_of_nodes)]
        TiTilde = TiTilde_Normalized
        TwTilde = TwTilde_Normalized
    
    return TiTilde, TwTilde

def believability(G, trustingness, trustworthiness):
    for edge in G.edges:
        #Try to loop through edges more efficiently
        fromN, toN =  edge
        #print(fromN, toN)
        belief = trustworthiness[fromN]* trustingness[toN]
        if belief == 1:
            print(belief)
        BeliefDict = {(fromN, toN):{'believability': belief}}
        nx.set_edge_attributes(G, BeliefDict)
    return G

def loop1Times(G): # spam 1 keer
    Time = 0
    V = list(G.nodes)
    spam_node = random.choice(V)
    suscept = V.copy()
    suscept.remove(spam_node)
    spam_nodes = [spam_node]
    spammed = 1
    spammed_over_time = [1/1000]
    while suscept and spam_nodes:
        Time += 1
        spam_nodes_after_round = []
        for node in spam_nodes:
            for neighbor in G[node]:
                if neighbor in suscept:
                    number = random.uniform(0, 1)
                    if number <= min(G.get_edge_data(node, neighbor)['believability'],1):
                        spammed += 1
                        spam_nodes_after_round.append(neighbor)
                        suscept.remove(neighbor)
        spam_nodes = spam_nodes_after_round  
        spammed_over_time.append(spammed/1000)
    
    return Time, spammed/1000, spammed_over_time 
# Here the function that loops over all over graphs
def RunOverAllGraphs(graphs, it):
  times = []
  spammed_list = []
  spammed_over_time_list =[]
  for graph in graphs:
    graph_directed = graph.to_directed()
    trustingness, trustworthiness = TsmAlgo(graph_directed, 10)
    beliefGraph = believability(graph_directed, trustingness, trustworthiness)
    for loop in range(it):
      time, spammed, spammed_over_time =   loop1Times(beliefGraph)

      times.append(time)
      spammed_list.append(spammed)
      spammed_over_time_list.append(spammed_over_time)
      
  return times, spammed_list, spammed_over_time_list

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
  



t0 = time.time()


degrees = [4,20,10,30,50,100]
for d in degrees:
    graphs = load_graphs(f'graphs_{d}.pickle')
    times, spammed_list, spammed_over_time_list = RunOverAllGraphs(graphs,100)
    ltotinf = confidence_interval(spammed_list)
    ldeathtime = confidence_interval(times)
    deel3 = CI_over_time(spammed_over_time_list)
    # print('CI per time on nr infected', deel3)
    f = open(f'data/bel_deg_{d}_max.txt', "w")
    f.write('CI yield:')
    f.write(','.join(str(i) for i in ltotinf))
    f.write('\n')
    f.write('CI ttd:')
    f.write(','.join(str(i) for i in ldeathtime))
    f.write('\n')
    f.write('List of death times')
    f.write('\n')
    for i in times:
        f.write(str(i))
        f.write('\n')
    f.write('CI per time on nr infected \n')
    for t in deel3:
        f.write(','.join(str(i) for i in t))
        f.write('\n')
    f.close()
    lower_list = []
    mean_list = []
    upper_list = []
    for i in deel3:
        lower_list.append(i[0])
        mean_list.append(i[1])
        upper_list.append(i[2])
    num = 1
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(lower_list, linestyle='None', marker='o', label = "lower bounds CI") 
    ax1.plot(mean_list, linestyle='None', marker='o', label = "mean CI") 
    ax1.plot(upper_list, linestyle='None', marker='o', label = "upper bounds CI") 
    ax1.set_title(f'Fraction of infected nodes per time step with E[D]={d}')
    ax1.set( ylabel="Fraction of infected nodes")
    ax2.set(xlabel="Time steps",ylabel='Number of occurrences')
    ax1.legend() 
    ax2.hist(times, bins = [i for i in range(max(times)+1)], color='skyblue', edgecolor='black')
    ax2.set_title('Histogram for the time till death of each run')
    fig.tight_layout()
    fig.savefig(f'plots/bel_deg_{d}_max.png')

t1 = time.time()
=======
import networkx as nx
import random
from make_graphs import load_graphs
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, std, mean

def TsmAlgo(G, MaxIt=1000):
    numb_of_nodes = len(list(G.nodes))
    TiTilde = [1]*numb_of_nodes
    TwTilde = [1]*numb_of_nodes
    for i in range(MaxIt):
        TiTilde_new = [0]*numb_of_nodes
        TwTilde_new = [0]*numb_of_nodes
        for node in G.nodes:
            for neighbor in G.neighbors(node):
                TiTilde_new[node] += 1/(1+ TwTilde[neighbor])**(0.5)
            for predecessor in list(G.predecessors(node)):
                TwTilde_new[node] += 1/(1+ TiTilde[predecessor])**(0.5) 
        TiTilde_Normalized = [TiTilde_new[j]/max(TiTilde_new) for j in range(numb_of_nodes)]
        TwTilde_Normalized = [TwTilde_new[j]/max(TwTilde_new) for j in range(numb_of_nodes)]
        TiTilde = TiTilde_Normalized
        TwTilde = TwTilde_Normalized
    
    return TiTilde, TwTilde

def believability(G, trustingness, trustworthiness):
    for edge in G.edges:
        #Try to loop through edges more efficiently
        fromN, toN =  edge
        #print(fromN, toN)
        belief = trustworthiness[fromN]* trustingness[toN]
        if belief == 1:
            print(belief)
        BeliefDict = {(fromN, toN):{'believability': belief}}
        nx.set_edge_attributes(G, BeliefDict)
    return G

def loop1Times(G): # spam 1 keer
    Time = 0
    V = list(G.nodes)
    spam_node = random.choice(V)
    suscept = V.copy()
    suscept.remove(spam_node)
    spam_nodes = [spam_node]
    spammed = 1
    spammed_over_time = [1/1000]
    while suscept and spam_nodes:
        Time += 1
        spam_nodes_after_round = []
        for node in spam_nodes:
            for neighbor in G[node]:
                if neighbor in suscept:
                    number = random.uniform(0, 1)
                    if number <= min(G.get_edge_data(node, neighbor)['believability'],1):
                        spammed += 1
                        spam_nodes_after_round.append(neighbor)
                        suscept.remove(neighbor)
        spam_nodes = spam_nodes_after_round  
        spammed_over_time.append(spammed/1000)
    
    return Time, spammed/1000, spammed_over_time 
# Here the function that loops over all over graphs
def RunOverAllGraphs(graphs, it):
  times = []
  spammed_list = []
  spammed_over_time_list =[]
  for graph in graphs:
    graph_directed = graph.to_directed()
    trustingness, trustworthiness = TsmAlgo(graph_directed, 10)
    beliefGraph = believability(graph_directed, trustingness, trustworthiness)
    for loop in range(it):
      time, spammed, spammed_over_time =   loop1Times(beliefGraph)

      times.append(time)
      spammed_list.append(spammed)
      spammed_over_time_list.append(spammed_over_time)
      
  return times, spammed_list, spammed_over_time_list

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
  



t0 = time.time()


degrees = [4,20,10,30,50,100]
for d in degrees:
    graphs = load_graphs(f'graphs_{d}.pickle')
    times, spammed_list, spammed_over_time_list = RunOverAllGraphs(graphs,100)
    ltotinf = confidence_interval(spammed_list)
    ldeathtime = confidence_interval(times)
    deel3 = CI_over_time(spammed_over_time_list)
    # print('CI per time on nr infected', deel3)
    f = open(f'data/bel_deg_{d}_max.txt', "w")
    f.write('CI yield:')
    f.write(','.join(str(i) for i in ltotinf))
    f.write('\n')
    f.write('CI ttd:')
    f.write(','.join(str(i) for i in ldeathtime))
    f.write('\n')
    f.write('List of death times')
    f.write('\n')
    for i in times:
        f.write(str(i))
        f.write('\n')
    f.write('CI per time on nr infected \n')
    for t in deel3:
        f.write(','.join(str(i) for i in t))
        f.write('\n')
    f.close()
    lower_list = []
    mean_list = []
    upper_list = []
    for i in deel3:
        lower_list.append(i[0])
        mean_list.append(i[1])
        upper_list.append(i[2])
    num = 1
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(lower_list, linestyle='None', marker='o', label = "lower bounds CI") 
    ax1.plot(mean_list, linestyle='None', marker='o', label = "mean CI") 
    ax1.plot(upper_list, linestyle='None', marker='o', label = "upper bounds CI") 
    ax1.set_title(f'Fraction of infected nodes per time step with E[D]={d}')
    ax1.set( ylabel="Fraction of infected nodes")
    ax2.set(xlabel="Time steps",ylabel='Number of occurrences')
    ax1.legend() 
    ax2.hist(times, bins = [i for i in range(max(times)+1)], color='skyblue', edgecolor='black')
    ax2.set_title('Histogram for the time till death of each run')
    fig.tight_layout()
    fig.savefig(f'plots/bel_deg_{d}_max.png')

t1 = time.time()
>>>>>>> 5e973388a52fb6c45ff92f33650ac77c38099c0e
print('time', t1-t0)