from cmath import pi
from tkinter import W
import numpy as np
from random import randint,random
import numpy.linalg as LA
from random import random, randrange, seed, choice, uniform
import math
from tqdm import tqdm
import random
import torch
import pickle
import Levenshtein as lev
from operator import itemgetter
import dill

class MOEA:

    def __init__(self,input):
        #param = {'data': data, 'k': k, 'threshold': threshold, 'utility': getThresholdUtility(threshold),
            #'res_file':res_file, 'times':times,'ea':ea,'pc':pc ,'res_pkl':res_pkl}

        self.candidates=input['data'][0]
        self.seqs=input['data'][1]
        self.columnIndex=input['data'][2]
        self.columnWeights=input['data'][3]
        
        self.k = input['k']
        self.threshold=input['threshold']
        self.utility=input['utility']
        self.res_file=input['res_file']
        self.times=input['times']
        self.res_pkl = input['res_pkl']
        self.pc = input['pc']
        self.ea=input['ea']

        self.n = self.candidates.shape[0]

        self.p=[]
        self.alphas=[]
        beta=2
        sum = 0
        for i in range(1,int(self.n / 2)):
            sum += math.pow(i, -beta)
        for i in range(1,int(self.n/2)):
            self.alphas.append(i)
            self.p.append((1 / sum) * pow(i, -beta))
        self.p=np.array(self.p)
        
    def func(self, alpha, beta=2):
        sum = 0
        for i in range(int(self.n / 2)):
            sum += np.pow(i, -beta)
        return (1 / sum) * pow(alpha, -beta)

    def setEvaluateTime(self,time):
        self.evaluateTime=time

    def run_MOEA(self, cm):

        if self.ea=='PVD-NSGA-II-WR':
            self.crossover="one-point crossover"
            self.mutation="bit-wise mutation"
            self.select="uniform selection"
            self.doNSGA2_WR(cm)

        elif self.ea=='PVD-GSEMO-R':
            self.pc=0
            self.crossover=None
            self.mutation="bit-wise mutation"
            self.select = "uniform selection"
            self.doGSEMO_with_Repair(cm)
        
        elif self.ea=='PVD-GSEMO-WR':
            self.pc=0
            self.crossover=None
            self.mutation="bit-wise mutation"
            self.select = "uniform selection"
            self.doGSEMO_with_Warm_start_Repair(cm)
        
        elif self.ea=='PVD-GSEMO':
            self.pc=0
            self.crossover=None
            self.mutation="bit-wise mutation without repair"
            self.select = "uniform selection"
            self.doGSEMO(cm)


    def offspring_del_similar(self,offspring):
        selectable = np.ones(self.n)
        offspring_index = np.nonzero(offspring)
        size=len(offspring_index[1])
        
        for i in range(size):
            index=offspring_index[1][i]
            if selectable[index]==1:
                cur_set=[]
                cur_set.append(index)
                for j in range(i+1,size):
                    if lev.distance(self.seqs[index], self.seqs[offspring_index[1][j]]) <= self.threshold:
                        cur_set.append(offspring_index[1][j])

                rand_index=randint(1, len(cur_set)) - 1
                for q in range(len(cur_set)):
                    if cur_set[q]!=cur_set[rand_index]:
                        selectable[cur_set[q]] = 0
        return np.mat(np.array(offspring) * selectable)

    def modified_offspring_del_similar(self, parent, offspring):
        selectable = np.ones(self.n)
        offspring_index = np.nonzero(offspring)
        size=len(offspring_index[1])
        
        for i in range(size):
            index=offspring_index[1][i]
            if selectable[index]==1:
                cur_set=[]
                cur_set.append(index)
                for j in range(i+1,size):
                    if lev.distance(self.seqs[index], self.seqs[offspring_index[1][j]]) <= self.threshold:
                        cur_set.append(offspring_index[1][j])

                for ind in range(len(cur_set)):
                    if parent[0, cur_set[ind]] == 0:
                        rand_index = ind
                        break
                else:
                    rand_index=randint(1, len(cur_set)) - 1

                for q in range(len(cur_set)):
                    if cur_set[q]!=cur_set[rand_index]:
                        selectable[cur_set[q]] = 0
        return np.mat(np.array(offspring) * selectable)
    
    def s_is_legal(self, candidate_s):
        candidate_s_index = np.nonzero(candidate_s)
        size = len(candidate_s_index[1])
        for i in range(size):
            for j in range(i+1, size):
                if lev.distance(self.seqs[candidate_s_index[1][i]], self.seqs[candidate_s_index[1][j]]) <= self.threshold:
                    return False
        return True
    
    def mutation_function(self, s):
        if self.mutation=='bit-wise mutation':
            rand_rate = 1.0 / (self.n)
            change = np.random.binomial(1, rand_rate, self.n)            
            offspring=np.abs(s - change)   

            # return self.modified_offspring_del_similar(s, offspring)
            return self.offspring_del_similar(offspring)
        
        elif self.mutation == 'bit-wise mutation without repair':
            rand_rate = 1.0 / (self.n)
            change = np.random.binomial(1, rand_rate, self.n)            
            offspring=np.abs(s - change)
            return offspring

        elif self.mutation == 'effective mutation':
            if s[0, :].sum()<self.k:
                rand_rate = 1.0 / (self.n)
                change = np.random.binomial(1, rand_rate, self.n)
                return np.abs(s - change)
            else:
                ones_index = []
                zero_index=[]
                change = np.mat(np.zeros([1, self.n], 'int8'))
                for i in range(self.n):
                    if s[0,i]==1 :
                        ones_index.append(i)
                    else:
                        zero_index.append(i)
                one = choice(ones_index)
                zero = choice(zero_index)
                change[0,one] = 0
                change[0,zero] = 1
                return  np.abs(s - change)
        
        elif self.mutation == 'fast mutation':
            alpha = np.random.choice(self.alphas, p=self.p.ravel())
            rand_rate = alpha / (self.n)
            change = np.random.binomial(1, rand_rate, self.n)
            
            offspring= np.abs(s - change)
            return self.offspring_del_similar(offspring)
        
    def crossover_function(self, elem1, elem2):
        if self.crossover=='one-point crossover':
            point = randrange(1, self.n - 1)
            child_elem1 = np.mat(np.zeros([1, self.n], 'int8'))
            child_elem2 = np.mat(np.zeros([1, self.n], 'int8'))
            for i in range(self.n):
                if i < point:
                    child_elem1[0, i] = elem1.element[0, i]
                    child_elem2[0, i] = elem2.element[0, i]
                else:
                    child_elem1[0, i] = elem2.element[0, i]
                    child_elem2[0, i] = elem1.element[0, i]
            return child_elem1, child_elem2
        elif self.crossover=='uniform crossover':
            child_elem1 = []
            child_elem2 = []
            for i in range(self.n):
                if random()< 0.5:
                    row1=elem1.element[i]
                    row2=elem2.element[i]
                else:
                    row2=elem1.element[i]
                    row1=elem2.element[i]
                child_elem1.append(row1)
                child_elem2.append(row2)
            return child_elem1, child_elem2
        elif self.crossover=='central-biased crossover':
            child_elem1 = np.mat(np.zeros([1, self.n],'int8'))
            different_index=[]
            for i in range(self.n):
                if elem1[0,i]==elem2[0,1]:
                    child_elem1[0,i]=elem1[0,i]
                else:
                    different_index.append(i)
            for i in range(int(len(different_index)/2)):
                item=choice(different_index)
                child_elem1[0,item] = 1
                different_index.remove(item)
            for item in different_index:
                child_elem1[0,item] = 0
            return child_elem1

    def turn_to_subset(self,solution):
        index = np.nonzero(solution)
        subset = []
        for i in index[1]:
            subset.append(i)
        return np.array(subset)

    def updateDistribution_forward(self, newRow, distributions):
        probabilityOfMiss = torch.prod(newRow[self.columnIndex], dim = 1,dtype = torch.float64).reshape(1, -1, 1)
        probabilityOfMiss=probabilityOfMiss.round(decimals=5)

        probabilityOfHit=1-probabilityOfMiss

        shiftedMass = distributions * probabilityOfHit
        convolution = distributions * probabilityOfMiss

        convolution[:,:,1:] += shiftedMass[:,:,:-1]
        convolution[:,:,-1] += shiftedMass[:,:,-1]
        return convolution
    
    def updateDistribution_backward(self, delRow, distributions):

        probabilityOfMiss = torch.prod(delRow[self.columnIndex], dim = 1,dtype = torch.float64).reshape(1, -1, 1)
        probabilityOfMiss=probabilityOfMiss.round(decimals=5)

        probabilityOfHit=1-probabilityOfMiss

        convolution = distributions / probabilityOfMiss

        col=distributions.shape[2]
        for i in range(1,col):
            cur=(probabilityOfHit/probabilityOfMiss)*(convolution[:,:,i-1].unsqueeze(2))
            convolution[:,:,i]-=cur[:,:,0]
    
        convolution[:,:,-1] *= probabilityOfMiss[:,:,0]
        return convolution

    def evaluateObjective(self,offspring, parent,distributions):
        # GPU
        distributions=distributions.cuda()

        design=self.turn_to_subset(offspring)
        if len(design)==0:
            distributions = torch.zeros( (1, len(self.columnIndex), len(self.utility)) )
            distributions[:, :, 0] = 1
            return 0.0, distributions

        change=parent-offspring

        index = np.nonzero(np.abs(change))       
        index=index[1]

        change=np.array(change)
        scores = torch.sum( distributions * self.utility.view(1,1,-1), dim = 2 ).reshape(-1)
        fitness_value=torch.sum(scores * self.columnWeights)
        for i in index:
            if change[0][i]==-1:
                distributions = self.updateDistribution_forward(self.candidates[i], distributions)
            
            elif change[0][i]==1:
                distributions = self.updateDistribution_backward(self.candidates[i], distributions)
                
                            
        scores = torch.sum( distributions * self.utility.view(1,1,-1), dim = 2 ).reshape(-1)
        fitness_value=torch.sum(scores * self.columnWeights).cpu().numpy()
        distributions=distributions.cpu()

        return fitness_value, distributions
    
    def doGSEMO_with_Repair(self, cm):
        # GPU
        self.candidates=self.candidates.cuda()
        self.utility=self.utility.cuda()
        self.columnWeights=self.columnWeights.cuda()

        if cm == True:
            population = self.res_pkl[str(self.times)]['population']
            distributions = self.res_pkl[str(self.times)]['distributions']
            fitness = self.res_pkl[str(self.times)]['fitness']
            it_of_T = self.res_pkl[str(self.times)]['iteration_T']
            subset = self.res_pkl[str(self.times)]['subset']
            maxValue = self.res_pkl[str(self.times)]['value']
            self.current_best_subset_fitness = (subset, maxValue)
            popSize = np.shape(fitness)[0]
            iter = 0
            self.evaluateTime = self.evaluateTime - it_of_T

        else:
            # initiate the population
            population = np.mat(np.zeros([1, self.n], 'int8')) 

            distribution = torch.zeros( (1, len(self.columnIndex), len(self.utility)),dtype = torch.float64 )
            distribution[:, :, 0] = 1

            distributions=[]
            distributions.append(distribution)

            fitness = np.mat(np.zeros([1, 2]))
            fitness[0, 0] = 0.0
        
            self.current_best_subset_fitness = (self.turn_to_subset(population[0,:]), fitness[0, 0] )

            popSize = 1
            iter = 0
            it_of_T=0
        with tqdm(range(int(self.evaluateTime*self.k * self.n)), position = 0, leave = True) as pbar:# progress bar which represents the total number of evaluations
            for _ in pbar:
                if iter == int(self.k * self.n):
                    iter = 0
                    it_of_T+=1
                    subset = self.current_best_subset_fitness[0]
                    maxValue = self.current_best_subset_fitness[1]

                    plt_obj = {'population': population, 'distributions':distributions ,'fitness': fitness, 'iteration_T': it_of_T,'subset': subset, 'value': maxValue}
                    
                    if str(self.times) not in self.res_pkl:
                        self.res_pkl[str(self.times)]={}

                    self.res_pkl[str(self.times)] = plt_obj 

                    with open(self.res_file + '/time_'+ str(self.times)+ '.pkl', 'wb') as f:
                        pickle.dump(self.res_pkl, f, pickle.HIGHEST_PROTOCOL)

                    print("current best solution:", maxValue)
                    log = open(self.res_file + '/time_'+ str(self.times)+ '.txt', 'a')
                    log.write(str(maxValue))
                    log.write("\n")
                    for item in subset:
                        log.write(str(item))
                        log.write(" ")
                    log.write("\n")
                    log.close()

            

                if self.select=='uniform selection':# choose a individual from population randomly
                    select_index=randint(1, popSize) - 1

                parent=population[select_index, :]
                parent_distribution=distributions[select_index]
                offSpring=self.mutation_function(parent)
                offSpringFit = np.mat(np.zeros([1, 2]))
                offSpringFit[0, 1] = offSpring[0, :].sum()
                
                if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > self.k:
                    continue

                iter += 1

                offSpringFit[0, 0], offSpring_distribution = self.evaluateObjective(offSpring, parent,parent_distribution)           
                
                isDominate = False
                for i in range(0, popSize):
                    if (fitness[i, 0] > offSpringFit[0, 0] and fitness[i, 1] <= offSpringFit[0, 1]) or (
                        fitness[i, 0] >= offSpringFit[0, 0] and fitness[i, 1] < offSpringFit[0, 1]):
                        isDominate = True
                        break

                if isDominate == False:  # there is no better individual than offSpring
                    Q = []
                    for j in range(0, popSize):
                        if offSpringFit[0, 0] >= fitness[j, 0] and offSpringFit[0, 1] <= fitness[j, 1]:
                            continue
                        else:
                            Q.append(j)

                    #update distributions 
                    for index in range(len(distributions) - 1, -1, -1):
                        if index not in Q:
                            del distributions[index]
                    distributions.append(offSpring_distribution)

                    fitness = np.vstack((fitness[Q, :], offSpringFit))  # update fitness
                    population = np.vstack((population[Q, :], offSpring))  # update population
                   
                    if offSpringFit[0, 0]> self.current_best_subset_fitness[1]:
                        self.current_best_subset_fitness = (self.turn_to_subset(offSpring),offSpringFit[0, 0])

                popSize = np.shape(fitness)[0]
        # migrate back to cpu
        self.candidates=self.candidates.cpu()
        self.utility=self.utility.cpu()
        self.columnWeights=self.columnWeights.cpu()
        torch.cuda.empty_cache()

    def doNSGA2_WR(self, cm):
        # GPU
        self.candidates=self.candidates.cuda()
        self.utility=self.utility.cuda()
        self.columnWeights=self.columnWeights.cuda()

        crossover_rate = self.pc
        mutation_rate = 1.0
        num_objectives = 2

        if cm == True:
            # read the saved parameters and continue running
            P = self.res_pkl[str(self.times)]['P']
            Q = self.res_pkl[str(self.times)]['Q']
            query_times = self.res_pkl[str(self.times)]['query_times']
            T=self.evaluateTime
            it = self.res_pkl[str(self.times)]['it']
            subset = self.res_pkl[str(self.times)]['subset']
            maxValue = self.res_pkl[str(self.times)]['value']
            population_num = 2*self.k + 2
            greedy_time = (int)(self.k*self.n)

        else:
            # initialize the population
            P = []
            population_num = 2*self.k + 2
            for count in range(population_num):
                P.append(self.get_elem(np.mat(np.zeros([1, self.n], 'int8'))))
            for i in range(1, self.k):
            # randomly initialize two solutions of size i
                P[2*i].element, pfitness, P[2*i].distribution = self.random_k_solution(i)
                P[2*i].f1_value = pfitness[0, 0]
                P[2*i].f2_value = pfitness[0, 1]

                P[2*i+1].element, Pfitness, P[2*i+1].distribution = self.random_k_solution(i)
                P[2*i+1].f1_value = Pfitness[0, 0]
                P[2*i+1].f2_value = Pfitness[0, 1]
            P[2*self.k].element, kf, P[2*self.k].distribution = self.random_k_solution(self.k)
            P[2*self.k].f1_value = kf[0, 0]
            P[2*self.k].f2_value = kf[0, 1]

            # add greedy solution
            init_solution = np.mat(np.zeros([1, self.n], 'int8'))
            init_distribution = torch.zeros( (1, len(self.columnIndex), len(self.utility)),dtype = torch.float64 )
            init_distribution[:, :, 0] = 1

            with open('greedy_res/mhc1_credences_greedy_k_'+str(self.k)+'_thrd_'+str(int(self.k*0.25))+'.pkl', 'rb') as f:
                greedy_pkl = pickle.load(f)
            P[2*self.k+1].element[0, greedy_pkl['solution']] = 1
            P[2*self.k+1].f2_value = np.sum(P[2*self.k+1].element[0] == 1)
            P[2*self.k+1].f1_value, P[2*self.k+1].distribution = self.evaluateObjective(P[2*self.k+1].element, init_solution, init_distribution)

            print("greedy solution:", P[2*self.k+1].element, "is legal:", self.s_is_legal(P[2*self.k+1].element), "f1:", P[2*self.k+1].f1_value, "f2:", P[2*self.k+1].f2_value, "distribution:", P[2*self.k+1].distribution)
            print(greedy_pkl['value'])


            query_times = population_num
            # set T times the number of rounds of the greedy algorithm
            T=self.evaluateTime
            greedy_time = (int)(self.k*self.n)
            Q = []
            it = 1
        while query_times <= T * greedy_time:
            R = []
            R.extend(P)
            R.extend(Q)
            fronts = self.fast_nondominated_sort(R)
            del P[:]
            for front in fronts.values():
                if len(front) == 0:
                    break

                self.crowding_distance_assignment(front, num_objectives)
                P.extend(front)

                if len(P) >= population_num:
                    break
            self.sort_crowding(P)

            if len(P) > population_num:
                del P[population_num:]
            
            query_times += population_num
            Q = self.make_new_pop( P, crossover_rate, mutation_rate)
            

            # information storage
            if query_times / greedy_time >= it:
                print("save information!")
                it += 1
                resultIndex = -1
                maxValue = float("-inf")
                for i in range(0, len(P)):
                    p=P[i]
                    if p.f2_value <= self.k and p.f1_value > maxValue:
                        maxValue = p.f1_value
                        resultIndex = i

                subset = self.turn_to_subset(P[resultIndex].element)
                self.current_best_subset_fitness = (subset, maxValue)
                print("current best solution:", self.current_best_subset_fitness)
                plt_obj = {'P': P, 'Q': Q, 'query_times': query_times, 'subset': subset, 'value': maxValue, 'it': it}
                
                if str(self.times) not in self.res_pkl:
                    self.res_pkl[str(self.times)] = {}
                self.res_pkl[str(self.times)] = plt_obj
                
                with open(self.res_file + '/time_'+ str(self.times)+ '.pkl', 'wb') as f:
                    dill.dump(self.res_pkl, f)

                
                log = open(self.res_file + '/time_'+ str(self.times)+ '.txt', 'a')
                log.write(str(maxValue))
                log.write("\n")
                for item in subset:
                    log.write(str(item))
                    log.write(" ")
                log.write("\n")
                log.close()
        # migrate back to cpu
        self.candidates=self.candidates.cpu()
        self.utility=self.utility.cpu()
        self.columnWeights=self.columnWeights.cpu()
        torch.cuda.empty_cache()


    def turn_to_subset_NSGA2(self,solution):
        index = np.nonzero(solution.element)
        subset = []
        for i in index[0]:
            subset.append(i)
        return np.array(subset)

    def get_elem(self, elem, parent=None):
        class Elem(object):
            def __init__(self, f1_value, f2_value, element, distribution):
                super(Elem, self).__init__()
                self.f1_value = f1_value # f1
                self.f2_value = f2_value # f2
                self.element = element # 01 string
                self.distribution = distribution
        #f2
        f2=np.sum(elem)

        #f1
        if f2==0:
            f1 = 0.0
            elem_distribution = torch.zeros( (1, len(self.columnIndex), len(self.utility)),dtype = torch.float64 )
            elem_distribution[:, :, 0] = 1
        else:
            f1, elem_distribution= self.evaluateObjective(elem, parent.element, parent.distribution)

        if f2>self.k:
            f1=float("-inf")
        return Elem(f1, f2, elem, elem_distribution)

    def sort_objective(self,P, obj_idx):
        for i in range(len(P) - 1, -1, -1):
            for j in range(1, i + 1):
                s1 = P[j - 1]
                s2 = P[j]
                if obj_idx == 0:
                    if s1.f1_value > s2.f1_value:
                        P[j - 1] = s2
                        P[j] = s1
                else:
                    if s1.f2_value > s2.f2_value:
                        P[j - 1] = s2
                        P[j] = s1

    def sort_crowding(self,P):
        for i in range(len(P) - 1, -1, -1):
            for j in range(1, i + 1):
                s1 = P[j - 1]
                s2 = P[j]

                if self.crowded_comparison(s1, s2) < 0:
                    P[j - 1] = s2
                    P[j] = s1

    def make_new_pop(self, P, crossover_rate, mutation_rate):
        '''
        Make new population Q, offspring of P.
        '''
        Q = []
        while len(Q) != len(P):
            selected_solutions = [None, None]

            while selected_solutions[0] == selected_solutions[1]:
                for i in range(2):
                    s1 = choice(P)
                    while s1.f1_value==float("-inf"):
                        s1 = choice(P)
                    s2 = s1
                    while s1 == s2 or s1.f1_value==float("-inf"):
                        s2 = choice(P)

                    if self.crowded_comparison(s1, s2) > 0:
                        selected_solutions[i] = s1

                    else:
                        selected_solutions[i] = s2

            if random.random() < crossover_rate:
                child_solution_element1,child_solution_element2 = self.crossover_function(selected_solutions[0], selected_solutions[1])

                '''if random() < mutation_rate:
                    child_solution1 = self.mutation_function(child_solution_element1)
                    child_solution2 = self.mutation_function(child_solution_element2)

                else:
                    child_solution1 = child_solution_element1
                    child_solution2 = child_solution_element2'''
                child_solution1 = self.mutation_function(child_solution_element1)
                child_solution2 = self.mutation_function(child_solution_element2)
                if self.s_is_legal(child_solution1) == False or self.s_is_legal(child_solution2) == False:
                    print("create error!!!!")
                Q.append(self.get_elem(child_solution1, selected_solutions[0]))
                Q.append(self.get_elem(child_solution2, selected_solutions[0]))

        return Q

    def fast_nondominated_sort(self,P):
        '''
        Discover Pareto fronts in P, based on non-domination criterion.
        '''
        fronts = {}

        S = {}
        n = {}
        it=0
        for s in P:
            it+=1
            S[s] = []
            n[s] = 0

        fronts[1] = []

        for p in P:
            for q in P:
                if (np.array(p.element)[0]==np.array(q.element)[0]).all():
                    continue
                if ((p.f1_value > q.f1_value) and (p.f2_value >= q.f2_value)) or (
                        (p.f1_value >= q.f1_value) and (p.f2_value > q.f2_value)):
                    S[p].append(q)

                elif ((q.f1_value > p.f1_value) and (q.f2_value >= p.f2_value)) or (
                        (q.f1_value >= p.f1_value) and (q.f2_value > p.f2_value)):
                    n[p] += 1

            if n[p] == 0:
                p.rank = 1
                fronts[1].append(p)

        i = 1

        while len(fronts[i]) != 0:
            next_front = []

            for r in fronts[i]:
                for s in S[r]:
                    n[s] -= 1
                    if n[s] == 0:
                        s.rank = i + 1
                        next_front.append(s)

            i += 1
            fronts[i] = next_front

        return fronts

    def crowding_distance_assignment(self,front, num_objectives):
        '''
        Assign a crowding distance for each solution in the front.
        '''
        for p in front:
            p.distance = 0

        for obj_index in range(num_objectives):
            self.sort_objective(front, obj_index)

            front[0].distance = float('inf')
            front[len(front) - 1].distance = float('inf')

            for i in range(1, len(front) - 1):
                front[i].distance += (front[i + 1].distance - front[i - 1].distance)

    def crowded_comparison(self,s1, s2):
        '''
        Compare the two solutions based on crowded comparison.
        '''
        if s1.rank < s2.rank:
            return 1

        elif s1.rank > s2.rank:
            return -1

        elif s1.distance > s2.distance:
            return 1

        elif s1.distance < s2.distance:
            return -1

        else:
            return 0

    def random_k_solution(self, k):
        zero_solution = np.mat(np.zeros([1, self.n], 'int8'))
        zero_distribution = torch.zeros( (1, len(self.columnIndex), len(self.utility)),dtype = torch.float64 )
        zero_distribution[:, :, 0] = 1
        # solution of size k
        k_solution = np.mat(np.zeros([1, self.n], 'int8'))
        k_solution_fitness = np.mat(np.zeros([1, 2]))
        k_solution_fitness[0, 0] = 0.0
        # first flip 10*k bits
        filp_index = random.sample(range(self.n), 10*k)
        k_solution[0, filp_index] = 1
        k_solution = self.offspring_del_similar(k_solution)
        leaved_index = list(np.nonzero(k_solution[0])[1])
        # add/remove after constraint processing
        if len(leaved_index) > k:
            delete_index = random.sample(leaved_index, len(leaved_index)-k)
            k_solution[0, delete_index] = 0
        elif len(leaved_index) < k:
            for i in range(self.n):
                if k_solution[0, i] == 0:
                    cad = True
                    for ind in leaved_index:
                        if lev.distance(self.seqs[ind], self.seqs[i]) <= self.threshold:
                            cad = False
                            break
                    if cad == True:
                        k_solution[0, i] = 1
                        leaved_index.append(i)
                if np.sum(k_solution[0]) == k:
                    break

        k_solution_fitness[0, 1] = np.sum(k_solution[0])
        k_solution_fitness[0, 0], k_solution_distribution = self.evaluateObjective(k_solution, zero_solution, zero_distribution)
        if self.s_is_legal(k_solution):
            print("create legal solution:", k_solution_fitness[0, 1], 'fitness:',k_solution_fitness[0, 0])
        return k_solution, k_solution_fitness, k_solution_distribution
               
    def doGSEMO_with_Warm_start_Repair(self, cm):
        # GPU
        self.candidates=self.candidates.cuda()
        self.utility=self.utility.cuda()
        self.columnWeights=self.columnWeights.cuda()

        if cm == True:
            # read the saved parameters and continue running
            population = self.res_pkl[str(self.times)]['population']
            distributions = self.res_pkl[str(self.times)]['distributions']
            fitness = self.res_pkl[str(self.times)]['fitness']
            it_of_T = self.res_pkl[str(self.times)]['iteration_T']
            subset = self.res_pkl[str(self.times)]['subset']
            maxValue = self.res_pkl[str(self.times)]['value']
            self.current_best_subset_fitness = (subset, maxValue)
            popSize = np.shape(fitness)[0]
            iter = 0
            self.evaluateTime = self.evaluateTime - it_of_T

        else:
            # initialize the population
            init_solution = np.mat(np.zeros([1, self.n], 'int8'))
            init_distribution = torch.zeros( (1, len(self.columnIndex), len(self.utility)),dtype = torch.float64 )
            init_distribution[:, :, 0] = 1
            init_fitness = np.mat(np.zeros([1, 2]))
            init_fitness[0, 0] = 0.0

            # add greedy solution
            greedy_s = np.mat(np.zeros([1, self.n], 'int8'))  # initiate the population
            greedy_s_Fit = np.mat(np.zeros([1, 2]))
            greedy_s_Fit[0, 0] = 0.0

            with open('greedy_res/mhc1_credences_greedy_k_'+str(self.k)+'_thrd_'+str(int(self.k*0.25))+'.pkl', 'rb') as f:
                greedy_pkl = pickle.load(f)

            greedy_s[0, greedy_pkl['solution']] = 1
            greedy_s_Fit[0, 1] = len(greedy_pkl['solution'])
            greedy_s_Fit[0, 0], greedy_s_distribution = self.evaluateObjective(greedy_s, init_solution.copy(), init_distribution.clone())
            print(greedy_s_Fit[0, 0])

            fitness = greedy_s_Fit.copy()
            population = greedy_s.copy()
            self.current_best_subset_fitness = (self.turn_to_subset(population[0,:]), fitness[0, 0] ) 
            distributions=[]
            distributions.append(greedy_s_distribution.clone())
            print("add greedy solution to population", self.current_best_subset_fitness, greedy_pkl['value'])

            # fill a front
            fitness = np.vstack((fitness, init_fitness.copy()))
            population = np.vstack((population, init_solution.copy()))
            distributions.append(init_distribution.clone())
            for ik in range(1, self.k):
                k_solution, k_solution_fitness, k_solution_distribution = self.random_k_solution(ik)
                if self.s_is_legal(k_solution):
                    fitness = np.vstack((fitness, k_solution_fitness.copy()))
                    population = np.vstack((population, k_solution.copy()))
                    distributions.append(k_solution_distribution.clone())

            popSize = np.shape(fitness)[0]
            iter = 0
            it_of_T=0
            print("init population size:", popSize)
        with tqdm(range(int(self.evaluateTime*self.k * self.n)), position = 0, leave = True) as pbar:# progress bar which represents the total number of evaluations
            for _ in pbar:
                if iter == int(self.k * self.n):
                    iter = 0
                    it_of_T+=1
                    subset = self.current_best_subset_fitness[0]
                    maxValue = self.current_best_subset_fitness[1]

                    plt_obj = {'population': population, 'distributions':distributions ,'fitness': fitness, 'iteration_T': it_of_T,'subset': subset, 'value': maxValue}
                    
                    if str(self.times) not in self.res_pkl:
                        self.res_pkl[str(self.times)]={}

                    self.res_pkl[str(self.times)] = plt_obj 

                    with open(self.res_file + '/time_'+ str(self.times)+ '.pkl', 'wb') as f:
                        pickle.dump(self.res_pkl, f, pickle.HIGHEST_PROTOCOL)

                    log = open(self.res_file + '/time_'+ str(self.times)+ '.txt', 'a')
                    log.write(str(maxValue))
                    log.write("\n")
                    for item in subset:
                        log.write(str(item))
                        log.write(" ")
                    log.write("\n")
                    log.close()
                    print("current best solution:", self.current_best_subset_fitness)

            

                if self.select=='uniform selection': # choose a individual from population randomly 
                    select_index=randint(1, popSize) - 1 

                parent=population[select_index, :]
                parent_distribution=distributions[select_index]
                offSpring=self.mutation_function(parent)
                if not self.s_is_legal(offSpring):
                    print("illegal offspring!")
                offSpringFit = np.mat(np.zeros([1, 2]))
                offSpringFit[0, 1] = offSpring[0, :].sum()

                             
                if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > self.k:
                    continue

                iter += 1

                offSpringFit[0, 0], offSpring_distribution = self.evaluateObjective(offSpring, parent,parent_distribution)
              
                isDominate = False
                for i in range(0, popSize):
                    if (fitness[i, 0] > offSpringFit[0, 0] and fitness[i, 1] <= offSpringFit[0, 1]) or (
                        fitness[i, 0] >= offSpringFit[0, 0] and fitness[i, 1] < offSpringFit[0, 1]):
                        isDominate = True
                        break

                if isDominate == False:  # there is no better individual than offSpring
                    Q = [] # Q is the solution to stay
                    for j in range(0, popSize):
                        if offSpringFit[0, 0] >= fitness[j, 0] and offSpringFit[0, 1] <= fitness[j, 1]:
                            continue
                        else:
                            Q.append(j)

                    #update distributions 
                    for index in range(len(distributions) - 1, -1, -1):
                        if index not in Q:
                            del distributions[index]
                    distributions.append(offSpring_distribution)

                    fitness = np.vstack((fitness[Q, :], offSpringFit))  # update fitness
                    population = np.vstack((population[Q, :], offSpring))  # update population
                   
                    # update the current best solution
                    if offSpringFit[0, 0]> self.current_best_subset_fitness[1]:
                        self.current_best_subset_fitness = (self.turn_to_subset(offSpring),offSpringFit[0, 0])
                        print("best solution update:", offSpringFit[0, 0], np.nonzero(offSpring[0]))

                popSize = np.shape(fitness)[0]
        # migrate back to cpu
        self.candidates=self.candidates.cpu()
        self.utility=self.utility.cpu()
        self.columnWeights=self.columnWeights.cpu()
        torch.cuda.empty_cache()


    def doGSEMO(self, cm):
        # GPU
        self.candidates=self.candidates.cuda()
        self.utility=self.utility.cuda()
        self.columnWeights=self.columnWeights.cuda()

        if cm == True:
            # read the saved parameters and continue running
            population = self.res_pkl[str(self.times)]['population']
            distributions = self.res_pkl[str(self.times)]['distributions']
            fitness = self.res_pkl[str(self.times)]['fitness']
            it_of_T = self.res_pkl[str(self.times)]['iteration_T']
            subset = self.res_pkl[str(self.times)]['subset']
            maxValue = self.res_pkl[str(self.times)]['value']
            self.current_best_subset_fitness = (subset, maxValue)
            popSize = np.shape(fitness)[0]
            iter = 0
            self.evaluateTime = self.evaluateTime - it_of_T

        else:
            # initiate the population
            population = np.mat(np.zeros([1, self.n], 'int8'))

            distribution = torch.zeros( (1, len(self.columnIndex), len(self.utility)),dtype = torch.float64 )
            distribution[:, :, 0] = 1

            distributions=[]
            distributions.append(distribution)

            fitness = np.mat(np.zeros([1, 2]))
            fitness[0, 0] = 0.0#f_1
        
            self.current_best_subset_fitness = (self.turn_to_subset(population[0,:]), fitness[0, 0] ) 

            popSize = 1
            iter = 0
            it_of_T=0
        with tqdm(range(int(self.evaluateTime*self.k * self.n)), position = 0, leave = True) as pbar:# progress bar which represents the total number of evaluations
            for _ in pbar:
                if iter == int(self.k * self.n):
                    iter = 0
                    it_of_T+=1
                    subset = self.current_best_subset_fitness[0]
                    maxValue = self.current_best_subset_fitness[1]

                    plt_obj = {'population': population, 'distributions':distributions ,'fitness': fitness, 'iteration_T': it_of_T,'subset': subset, 'value': maxValue}
                    
                    if str(self.times) not in self.res_pkl:
                        self.res_pkl[str(self.times)]={}

                    self.res_pkl[str(self.times)] = plt_obj 

                    with open(self.res_file + '/time_'+ str(self.times)+ '.pkl', 'wb') as f:
                        pickle.dump(self.res_pkl, f, pickle.HIGHEST_PROTOCOL)

                    log = open(self.res_file + '/time_'+ str(self.times)+ '.txt', 'a')
                    log.write(str(maxValue))
                    log.write("\n")
                    for item in subset:
                        log.write(str(item))
                        log.write(" ")
                    log.write("\n")
                    log.close()
                    print("current best solution", self.current_best_subset_fitness)

            
                while True:
                    if self.select=='uniform selection':# choose a individual from population randomly
                        select_index=randint(1, popSize) - 1

                    parent=population[select_index, :]
                    parent_distribution=distributions[select_index]
                    offSpring=self.mutation_function(parent)
                    if self.s_is_legal(offSpring):
                        offSpringFit = np.mat(np.zeros([1, 2]))  # value, size
                        offSpringFit[0, 1] = offSpring[0, :].sum()
                        break

                if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > self.k:
                    continue
                
                iter += 1

                offSpringFit[0, 0], offSpring_distribution = self.evaluateObjective(offSpring, parent,parent_distribution)
              
                isDominate = False
                for i in range(0, popSize):
                    if (fitness[i, 0] > offSpringFit[0, 0] and fitness[i, 1] <= offSpringFit[0, 1]) or (
                        fitness[i, 0] >= offSpringFit[0, 0] and fitness[i, 1] < offSpringFit[0, 1]):
                        isDominate = True
                        break

                if isDominate == False:  # there is no better individual than offSpring
                    Q = []
                    for j in range(0, popSize):
                        if offSpringFit[0, 0] >= fitness[j, 0] and offSpringFit[0, 1] <= fitness[j, 1]:
                            continue
                        else:
                            Q.append(j)

                    #update distributions 
                    for index in range(len(distributions) - 1, -1, -1):
                        if index not in Q:
                            del distributions[index]
                    distributions.append(offSpring_distribution)

                    fitness = np.vstack((fitness[Q, :], offSpringFit))  # update fitness
                    population = np.vstack((population[Q, :], offSpring))  # update population
                   
                    # update the current best solution
                    if offSpringFit[0, 0]> self.current_best_subset_fitness[1]:
                        self.current_best_subset_fitness = (self.turn_to_subset(offSpring),offSpringFit[0, 0])

                popSize = np.shape(fitness)[0]
        # migrate back to cpu
        self.candidates=self.candidates.cpu()
        self.utility=self.utility.cpu()
        self.columnWeights=self.columnWeights.cpu()
        torch.cuda.empty_cache()    