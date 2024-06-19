<<<<<<< HEAD
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, std, mean
import random
from scipy.stats import bernoulli
from make_graphs import load_graphs

def spread_rat(G, node, I_s, S_c):
    """
    Finds fraction of neighbours of a specific node which are in S_c
    """
    spreads = 0
    neighs = 0
    for neigh in G[node]:
        neighs += 1
        if neigh in I_s or neigh in S_c:
            spreads += 1
    return spreads/neighs


def one_it(G,p_1,p_2, p_3,I_s,S_c,S_t,S_p,E,E_im,I_m, tot_inf,tot_im, always_infected = True):
    """
    Based on the states of each node in the previous timestep computes their state in the next timestep
    Choose always_infected = False if you do not want every node of a node in I_s to be infected
    """
    I_s_new = I_s.copy()
    S_c_new = S_c.copy()
    S_t_new  = S_t.copy()
    S_p_new = S_p.copy()
    E_new = E.copy()
    E_im_new = E_im.copy()
    I_m_new = I_m.copy()
    for node in I_s:
        i = np.random.binomial(1, p_1)
        if i == 1:
            I_s_new.remove(node)
            S_t_new.append(node)
    for node in E:
        i = np.random.binomial(1, p_2)
        if i == 1:
            E_new.remove(node)
            I_s_new.append(node)
            tot_inf += 1
        else:
            E_new.remove(node)
            S_t_new.append(node)
    for node in S_p:
        rat = spread_rat(G, node, I_s, S_c)
        rat2 = spread_rat(G, node, I_m,I_m)
        if always_infected:
            if rat2 > 0:
                j = 1
                i=0
            elif rat > 0:
                i = 1
                j=0
            else: 
                i = 0
                j = 0
        if i == 1:
            S_p_new.remove(node)
            E_new.append(node)
        if j == 1:
            S_p_new.remove(node)
            E_im_new.append(node)
            tot_im += 1
    for node in E_im:
        i = np.random.binomial(1, p_3)
        if i == 1:
            I_m_new.append(node)
            E_im_new.remove(node)
        else:
            E_im_new.remove(node)
            S_t_new.append(node)
        
        
    
    return I_s_new,S_c_new,S_t_new,S_p_new,E_new, E_im_new, I_m_new, tot_inf, tot_im

def k_it(k,G,p_1,p_2, p_3,I_s,S_c,S_t,S_p,E, E_im,I_m):
    "Given a graph and nodes in each state, performs state transitions until stationary"
    tot_inf = 1
    tot_inf_list = [tot_inf/1000]
    tot_im = 1
    tot_im_list = [tot_im/1000]
    ratio_im_inf = [tot_inf/tot_im]
    cond = True
    i = 0
    while cond:
        if i > k:
            cond = False
        I_s,S_c,S_t,S_p,E,E_im,I_m,tot_inf,tot_im = one_it(G,p_1,p_2, p_3,I_s,S_c,S_t,S_p,E,E_im,I_m, tot_inf,tot_im)
        tot_inf_list.append(tot_inf/1000)
        tot_im_list.append(tot_im/1000)
        ratio_im_inf.append(tot_inf/tot_im)
        if not I_s and not E:
            seen = True
            for node in S_c:
                for neigh in G[node]:
                    if neigh in S_p:
                        seen = False
            if seen:
                cond = False
        i+=1
    return tot_inf/1000,tot_inf_list,tot_im/1000,tot_im_list,ratio_im_inf

p_1 = 1
p_2 = 0.25
p_3 = 0.25
def simulations(it, graphs, p_1,p_2,p_3):
    "given a number of iterations, graphs and values for the different transition probabilities, performs simulations for those values"
    k=500
    L_inf = []
    L_death_time = []
    L_tot_inf = []
    L_tot_im = []
    L_im = []
    L_rat = []
    for G in graphs:
        for i in range(it):
            V = list(G.nodes)
            S_p = []
            E = []
            I_s = []
            S_c = []
            S_t = []
            E_im = []
            I_m = []

            start_1 = random.choice(V)
            S_c.append(start_1)
            
            S_p = V.copy()
            S_p.remove(start_1)
            start_2 = random.choice(S_p)
            S_p.remove(start_2)
            I_m.append(start_2)
            tot_inf, inf, tot_im,im,ratio_im_inf = k_it(k,G,p_1,p_2, p_3,I_s,S_c,S_t,S_p,E, E_im,I_m)
            L_inf.append(inf)
            L_death_time.append(len(inf)-1)
            L_tot_inf.append(tot_inf)
            L_im.append(im)
            L_tot_im.append(tot_im)
            L_rat.append(ratio_im_inf)

    return L_inf, L_death_time ,L_tot_inf,L_im, L_tot_im,L_rat

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


##### RUN SIMULATIONS ######
p_2=0.25
# degrees = [4,10,30,50,100]
probabilities = [0.05,0.1,0.15,0.2,0.25,0.5,0.75]
degrees = [20]
# probabilities = [0.05]
for d in degrees:
    if d == 20:
        for p in probabilities:
            graphs = load_graphs(f'graphs_{d}.pickle')
            L_inf, L_death_time, L_tot_inf, L_im, L_tot_im,L_rat = simulations(100,graphs, p_1,p_2,p)
            # print('')
            # print('degree', d)
            # print('probability', p)
            # print('CI yield:', confidence_interval(L_tot_inf))
            # print('CI ttd', confidence_interval(L_death_time))
            ltotinf = confidence_interval(L_tot_inf)
            ldeathtime = confidence_interval(L_death_time)
            lim = confidence_interval(L_tot_im)
            deel3 = CI_over_time(L_inf)
            im = CI_over_time(L_im)
            # print('CI per time on nr infected', deel3)
            f = open(f'data/SEIR_block_deg_{d}_prob_{p}.txt', "w")
            f.write('CI yield:')
            f.write(','.join(str(i) for i in ltotinf))
            f.write('\n')
            f.write('CI ttd:')
            f.write(','.join(str(i) for i in ldeathtime))
            f.write('\n')
            f.write('CI im:')
            f.write(','.join(str(i) for i in lim))
            f.write('\n')
            f.write('List of death times')
            f.write('\n')
            for i in L_death_time:
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
            lower_listm = []
            mean_listm = []
            upper_listm = []
            for i in im:
                lower_listm.append(i[0])
                mean_listm.append(i[1])
                upper_listm.append(i[2])
            fig, (ax1, ax2) = plt.subplots(2, sharex=True)
            ax1.plot(lower_list, linestyle='None', marker='o', label = "LB CI inf") 
            ax1.plot(mean_list, linestyle='None', marker='o', label = "mean CI inf") 
            ax1.plot(upper_list, linestyle='None', marker='o', label = "UB CI inf") 
            ax1.plot(lower_listm, linestyle='None', marker='o', label = "LB CI im") 
            ax1.plot(mean_listm, linestyle='None', marker='o', label = "mean CI im") 
            ax1.plot(upper_listm, linestyle='None', marker='o', label = "UB CI im") 
            ax1.set_title(f'Infected/immunized percentage per time step with p_3={p}')
            ax1.set( ylabel="Infected/immunized fraction")
            ax2.set(xlabel="Time steps",ylabel='Number of occurrences')
            # ax1.legend() 
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax2.hist(L_death_time, bins = [i for i in range(max(L_death_time)+1)], color='skyblue', edgecolor='black')
            ax2.set_title('Histogram for the time till death of each run')
            fig.tight_layout()
            fig.savefig(f'plots/SEIR_block_deg_{d}_prob_{p}.png')
            #fig.close()
    else:
        graphs = load_graphs(f'graphs_{d}.pickle')
        L_inf, L_death_time, L_tot_inf, L_im, L_tot_im,L_rat = simulations(100,graphs, p_1,0.25,p_3)
        # print('')
        # print('degree', d)
        # print('probability', p)
        # print('CI yield:', confidence_interval(L_tot_inf))
        # print('CI ttd', confidence_interval(L_death_time))
        ltotinf = confidence_interval(L_tot_inf)
        ldeathtime = confidence_interval(L_death_time)
        deel3 = CI_over_time(L_inf)
        im = CI_over_time(L_im)
        # print('CI per time on nr infected', deel3)
        f = open(f'data/SEIR_deg_{d}_prob_0.25.txt', "w")
        f.write('CI yield:')
        f.write(','.join(str(i) for i in ltotinf))
        f.write('\n')
        f.write('CI ttd:')
        f.write(','.join(str(i) for i in ldeathtime))
        f.write('\n')
        f.write('List of death times')
        f.write('\n')
        for i in L_death_time:
            f.write(str(i))
            f.write('\n')
        f.write('CI per time on nr infected')
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
        lower_listm = []
        mean_listm = []
        upper_listm = []
        for i in im:
            lower_listm.append(i[0])
            mean_listm.append(i[1])
            upper_listm.append(i[2])
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(lower_list, linestyle='None', marker='o', label = "LB CI inf") 
        ax1.plot(mean_list, linestyle='None', marker='o', label = "mean CI inf") 
        ax1.plot(upper_list, linestyle='None', marker='o', label = "UB CI inf") 
        ax1.plot(lower_listm, linestyle='None', marker='o', label = "LB CI im") 
        ax1.plot(mean_listm, linestyle='None', marker='o', label = "mean CI im") 
        ax1.plot(upper_listm, linestyle='None', marker='o', label = "UB CI im") 
        ax1.set_title(f'Infected/immunized percentage per time step with p_2={p} and E[D]={d}')
        ax1.set( ylabel="Fraction of infected/immunized nodes")
        ax2.set(xlabel="Time steps",ylabel='Number of occurrences')
        ax1.legend() 
        ax2.hist(L_death_time, bins = [i for i in range(max(L_death_time)+1)], color='skyblue', edgecolor='black')
        ax2.set_title('Histogram for the time till death of each run')
        fig.tight_layout()
        fig.savefig(f'plots/SEIR_block_deg_{d}_prob_0.25.png')

=======
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, std, mean
import random
from scipy.stats import bernoulli
from make_graphs import load_graphs

def spread_rat(G, node, I_s, S_c):
    """
    Finds fraction of neighbours of a specific node which are in S_c
    """
    spreads = 0
    neighs = 0
    for neigh in G[node]:
        neighs += 1
        if neigh in I_s or neigh in S_c:
            spreads += 1
    return spreads/neighs


def one_it(G,p_1,p_2, p_3,I_s,S_c,S_t,S_p,E,E_im,I_m, tot_inf,tot_im, always_infected = True):
    """
    Based on the states of each node in the previous timestep computes their state in the next timestep
    Choose always_infected = False if you do not want every node of a node in I_s to be infected
    """
    I_s_new = I_s.copy()
    S_c_new = S_c.copy()
    S_t_new  = S_t.copy()
    S_p_new = S_p.copy()
    E_new = E.copy()
    E_im_new = E_im.copy()
    I_m_new = I_m.copy()
    for node in I_s:
        i = np.random.binomial(1, p_1)
        if i == 1:
            I_s_new.remove(node)
            S_t_new.append(node)
    for node in E:
        i = np.random.binomial(1, p_2)
        if i == 1:
            E_new.remove(node)
            I_s_new.append(node)
            tot_inf += 1
        else:
            E_new.remove(node)
            S_t_new.append(node)
    for node in S_p:
        rat = spread_rat(G, node, I_s, S_c)
        rat2 = spread_rat(G, node, I_m,I_m)
        if always_infected:
            if rat2 > 0:
                j = 1
                i=0
            elif rat > 0:
                i = 1
                j=0
            else: 
                i = 0
                j = 0
        if i == 1:
            S_p_new.remove(node)
            E_new.append(node)
        if j == 1:
            S_p_new.remove(node)
            E_im_new.append(node)
            tot_im += 1
    for node in E_im:
        i = np.random.binomial(1, p_3)
        if i == 1:
            I_m_new.append(node)
            E_im_new.remove(node)
        else:
            E_im_new.remove(node)
            S_t_new.append(node)
        
        
    
    return I_s_new,S_c_new,S_t_new,S_p_new,E_new, E_im_new, I_m_new, tot_inf, tot_im

def k_it(k,G,p_1,p_2, p_3,I_s,S_c,S_t,S_p,E, E_im,I_m):
    "Given a graph and nodes in each state, performs state transitions until stationary"
    tot_inf = 1
    tot_inf_list = [tot_inf/1000]
    tot_im = 1
    tot_im_list = [tot_im/1000]
    ratio_im_inf = [tot_inf/tot_im]
    cond = True
    i = 0
    while cond:
        if i > k:
            cond = False
        I_s,S_c,S_t,S_p,E,E_im,I_m,tot_inf,tot_im = one_it(G,p_1,p_2, p_3,I_s,S_c,S_t,S_p,E,E_im,I_m, tot_inf,tot_im)
        tot_inf_list.append(tot_inf/1000)
        tot_im_list.append(tot_im/1000)
        ratio_im_inf.append(tot_inf/tot_im)
        if not I_s and not E:
            seen = True
            for node in S_c:
                for neigh in G[node]:
                    if neigh in S_p:
                        seen = False
            if seen:
                cond = False
        i+=1
    return tot_inf/1000,tot_inf_list,tot_im/1000,tot_im_list,ratio_im_inf

p_1 = 1
p_2 = 0.25
p_3 = 0.25
def simulations(it, graphs, p_1,p_2,p_3):
    "given a number of iterations, graphs and values for the different transition probabilities, performs simulations for those values"
    k=500
    L_inf = []
    L_death_time = []
    L_tot_inf = []
    L_tot_im = []
    L_im = []
    L_rat = []
    for G in graphs:
        for i in range(it):
            V = list(G.nodes)
            S_p = []
            E = []
            I_s = []
            S_c = []
            S_t = []
            E_im = []
            I_m = []

            start_1 = random.choice(V)
            S_c.append(start_1)
            
            S_p = V.copy()
            S_p.remove(start_1)
            start_2 = random.choice(S_p)
            S_p.remove(start_2)
            I_m.append(start_2)
            tot_inf, inf, tot_im,im,ratio_im_inf = k_it(k,G,p_1,p_2, p_3,I_s,S_c,S_t,S_p,E, E_im,I_m)
            L_inf.append(inf)
            L_death_time.append(len(inf)-1)
            L_tot_inf.append(tot_inf)
            L_im.append(im)
            L_tot_im.append(tot_im)
            L_rat.append(ratio_im_inf)

    return L_inf, L_death_time ,L_tot_inf,L_im, L_tot_im,L_rat

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


##### RUN SIMULATIONS ######
p_2=0.25
# degrees = [4,10,30,50,100]
probabilities = [0.05,0.1,0.15,0.2,0.25,0.5,0.75]
degrees = [20]
# probabilities = [0.05]
for d in degrees:
    if d == 20:
        for p in probabilities:
            graphs = load_graphs(f'graphs_{d}.pickle')
            L_inf, L_death_time, L_tot_inf, L_im, L_tot_im,L_rat = simulations(100,graphs, p_1,p_2,p)
            # print('')
            # print('degree', d)
            # print('probability', p)
            # print('CI yield:', confidence_interval(L_tot_inf))
            # print('CI ttd', confidence_interval(L_death_time))
            ltotinf = confidence_interval(L_tot_inf)
            ldeathtime = confidence_interval(L_death_time)
            lim = confidence_interval(L_tot_im)
            deel3 = CI_over_time(L_inf)
            im = CI_over_time(L_im)
            # print('CI per time on nr infected', deel3)
            f = open(f'data/SEIR_block_deg_{d}_prob_{p}.txt', "w")
            f.write('CI yield:')
            f.write(','.join(str(i) for i in ltotinf))
            f.write('\n')
            f.write('CI ttd:')
            f.write(','.join(str(i) for i in ldeathtime))
            f.write('\n')
            f.write('CI im:')
            f.write(','.join(str(i) for i in lim))
            f.write('\n')
            f.write('List of death times')
            f.write('\n')
            for i in L_death_time:
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
            lower_listm = []
            mean_listm = []
            upper_listm = []
            for i in im:
                lower_listm.append(i[0])
                mean_listm.append(i[1])
                upper_listm.append(i[2])
            fig, (ax1, ax2) = plt.subplots(2, sharex=True)
            ax1.plot(lower_list, linestyle='None', marker='o', label = "LB CI inf") 
            ax1.plot(mean_list, linestyle='None', marker='o', label = "mean CI inf") 
            ax1.plot(upper_list, linestyle='None', marker='o', label = "UB CI inf") 
            ax1.plot(lower_listm, linestyle='None', marker='o', label = "LB CI im") 
            ax1.plot(mean_listm, linestyle='None', marker='o', label = "mean CI im") 
            ax1.plot(upper_listm, linestyle='None', marker='o', label = "UB CI im") 
            ax1.set_title(f'Infected/immunized percentage per time step with p_3={p}')
            ax1.set( ylabel="Infected/immunized fraction")
            ax2.set(xlabel="Time steps",ylabel='Number of occurrences')
            # ax1.legend() 
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax2.hist(L_death_time, bins = [i for i in range(max(L_death_time)+1)], color='skyblue', edgecolor='black')
            ax2.set_title('Histogram for the time till death of each run')
            fig.tight_layout()
            fig.savefig(f'plots/SEIR_block_deg_{d}_prob_{p}.png')
            #fig.close()
    else:
        graphs = load_graphs(f'graphs_{d}.pickle')
        L_inf, L_death_time, L_tot_inf, L_im, L_tot_im,L_rat = simulations(100,graphs, p_1,0.25,p_3)
        # print('')
        # print('degree', d)
        # print('probability', p)
        # print('CI yield:', confidence_interval(L_tot_inf))
        # print('CI ttd', confidence_interval(L_death_time))
        ltotinf = confidence_interval(L_tot_inf)
        ldeathtime = confidence_interval(L_death_time)
        deel3 = CI_over_time(L_inf)
        im = CI_over_time(L_im)
        # print('CI per time on nr infected', deel3)
        f = open(f'data/SEIR_deg_{d}_prob_0.25.txt', "w")
        f.write('CI yield:')
        f.write(','.join(str(i) for i in ltotinf))
        f.write('\n')
        f.write('CI ttd:')
        f.write(','.join(str(i) for i in ldeathtime))
        f.write('\n')
        f.write('List of death times')
        f.write('\n')
        for i in L_death_time:
            f.write(str(i))
            f.write('\n')
        f.write('CI per time on nr infected')
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
        lower_listm = []
        mean_listm = []
        upper_listm = []
        for i in im:
            lower_listm.append(i[0])
            mean_listm.append(i[1])
            upper_listm.append(i[2])
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(lower_list, linestyle='None', marker='o', label = "LB CI inf") 
        ax1.plot(mean_list, linestyle='None', marker='o', label = "mean CI inf") 
        ax1.plot(upper_list, linestyle='None', marker='o', label = "UB CI inf") 
        ax1.plot(lower_listm, linestyle='None', marker='o', label = "LB CI im") 
        ax1.plot(mean_listm, linestyle='None', marker='o', label = "mean CI im") 
        ax1.plot(upper_listm, linestyle='None', marker='o', label = "UB CI im") 
        ax1.set_title(f'Infected/immunized percentage per time step with p_2={p} and E[D]={d}')
        ax1.set( ylabel="Fraction of infected/immunized nodes")
        ax2.set(xlabel="Time steps",ylabel='Number of occurrences')
        ax1.legend() 
        ax2.hist(L_death_time, bins = [i for i in range(max(L_death_time)+1)], color='skyblue', edgecolor='black')
        ax2.set_title('Histogram for the time till death of each run')
        fig.tight_layout()
        fig.savefig(f'plots/SEIR_block_deg_{d}_prob_0.25.png')

>>>>>>> 5e973388a52fb6c45ff92f33650ac77c38099c0e
