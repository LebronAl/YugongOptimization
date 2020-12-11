
import pandas as pd

## load data
cluster_capacity_df = pd.read_csv('./data/tmp_cluster_20201029', names=['cluster_name', 'cluster_total_cpu', 'cluster_total_storage']).drop_duplicates()
table_cpu_cost_df = pd.read_csv('./data/tmp_table_cpu_cost_20201029', names=['project_name','table_name', 'table_cost_cpu']).drop_duplicates()
table_job_df = pd.read_csv('./data/tmp_table_input_relevence_20201029', names=['ndTable', 'stTable', 'relevence_size']).drop_duplicates()
table_updown_df = pd.read_csv('./data/tmp_table_output_relevence_20201029', names=['output', 'input', 'relevence_size']).drop_duplicates()

## clean data
cluster_capacity_df['cluster_name'] = cluster_capacity_df['cluster_name'].str.lower()
table_job_df['ndTable'] = table_job_df['ndTable'].str.lower()
table_job_df['stTable'] = table_job_df['stTable'].str.lower()
table_updown_df['output'] = table_updown_df['output'].str.lower()
table_updown_df['input'] = table_updown_df['input'].str.lower()
cluster_capacity_df.fillna(value={'cluster_total_cpu':0}, inplace=True)
table_cpu_cost_df.fillna(value={'table_cost_cpu':0}, inplace=True)
table_job_df.fillna(value={'relevence_size':0}, inplace=True)
table_updown_df.fillna(value={'relevence_size':0}, inplace=True)


## data info
print("--------------------")
print("ClusterCapacityInfo : size = {}".format(
    cluster_capacity_df.shape[0]))
print(cluster_capacity_df.dtypes)
print(cluster_capacity_df.head())
print("--------------------")
print("TableCpuInfo : size = {}".format(table_cpu_cost_df.shape[0]))
print(table_cpu_cost_df.dtypes)
print(table_cpu_cost_df.head())
print("--------------------")
print("TableJobInfo : size = {}".format(
    table_job_df.shape[0]))
print(table_job_df.dtypes)
print(table_job_df.head())
print("--------------------")
print("TableUpDownInfo : size = {}".format(
            table_updown_df.shape[0]))
print(table_updown_df.dtypes)
print(table_updown_df.head())
print("--------------------")



from tqdm import tqdm

table_size = table_cpu_cost_df.shape[0]
cluster_size = cluster_capacity_df.shape[0]
table_job_size = table_job_df.shape[0]
table_updown_size = table_updown_df.shape[0]

## get Cluster Probability during choosing cluster based on this total cpu
clusterRouletteProbability = cluster_capacity_df['cluster_total_cpu'] / cluster_capacity_df['cluster_total_cpu'].sum()
print('----clusterRouletteProbability----')
print(clusterRouletteProbability)
print('--------')


## inverted index from table to their index
table2index = {}
for index in tqdm(range(table_size), desc='Processing'):
    table2index[table_cpu_cost_df.iloc[index, 0] + '.' + table_cpu_cost_df.iloc[index, 1]] = index

## get total Flow
totalFlow = 0
for index in tqdm(range(table_job_size), desc='Processing'):
    totalFlow += table_job_df.iloc[index, 2]    
for index in tqdm(range(table_updown_size), desc='Processing'):
    totalFlow += table_updown_df.iloc[index, 2]

print('----totalFlow----')
print(totalFlow)
print('--------')



import random

## evalute roulette Random algorithm
def rouletteRandom(probability):
    sum = 0
    ran = random.random()
    for num, r in zip(range(len(probability)), probability):
        sum += r
        if ran < sum :
            break
    return num

## For each individual, we calculate it's assessment, which is innerFlow and innerFlow.
## Addition, we stores the usedCPUMap(key=cluster_index,value=totalCPUForThisPlancement) and clusterTableMap(key=cluster_index,value=list(table_index))
class Individual:
    
    def __init__(self, entity, usedCPUMap, clusterTableMap, execute):
        self.entity = entity
        self.innerFlow = 0
        self.crossFlow = 0
        self.usedCPUMap = usedCPUMap
        self.clusterTableMap = clusterTableMap
        if execute:
            self.figureAssessment()
        
    def figureAssessment(self):
        self.figureTableJob()
        self.figureTableUpdown()
        self.printInfo()
        
    def printInfo(self):
        print("----IndividualInfo----")
        print("InnerFlow: " + str(self.innerFlow))
        print("CrossFlow: " + str(self.crossFlow))
        print("CPUUsage: ")
        for index in range(len(self.usedCPUMap)):
            print(str(index) + ":" + str(self.usedCPUMap[index] / cluster_capacity_df.iloc[index, 1]))
        print("--------")
        
    def figureTableJob(self):
        for index in tqdm(range(table_job_size), desc='Processing'):
            if table_job_df.iloc[index, 0] in table2index.keys():
                left = table2index[table_job_df.iloc[index, 0]]
            else:
                continue
            if table_job_df.iloc[index, 1] in table2index.keys():
                right = table2index[table_job_df.iloc[index, 1]]
            else:
                continue
            if self.entity[left] != self.entity[right]:
                self.crossFlow += table_job_df.iloc[index, 2]
            else:
                self.innerFlow += table_job_df.iloc[index, 2]
        
    def figureTableUpdown(self):
        for index in tqdm(range(table_updown_size), desc='Processing'):
            if table_updown_df.iloc[index, 0] in table2index.keys():
                left = table2index[table_updown_df.iloc[index, 0]]
            else:
                continue
            if table_updown_df.iloc[index, 1] in table2index.keys():
                right = table2index[table_updown_df.iloc[index, 1]]
            else:
                continue
            if self.entity[left] != self.entity[right]:
                self.crossFlow += table_updown_df.iloc[index, 2]
            else:
                self.innerFlow += table_updown_df.iloc[index, 2]
    
## random generates a valid ndividual along with it's usedCPUMap and clusterTableMap
def figureInitPopulation():
    usedCPUMap = np.zeros([cluster_size], dtype = np.int64)
    clusterTableMap = {}
    entity = np.zeros([table_size], dtype = np.int8)
    for index in tqdm(range(table_size), desc='Processing'):
        while True:
            choosedCluster = rouletteRandom(clusterRouletteProbability)
            if usedCPUMap[choosedCluster] + table_cpu_cost_df.iloc[index, 2] <= cluster_capacity_df.iloc[choosedCluster, 1]:
                usedCPUMap[choosedCluster] += table_cpu_cost_df.iloc[index, 2]
                if choosedCluster not in clusterTableMap.keys():
                    clusterTableMap[choosedCluster] = []
                clusterTableMap[choosedCluster].append(index)
                entity[index] = choosedCluster
                break
    return Individual(entity, usedCPUMap, clusterTableMap, True)

## init cpu_num - 1 individuals for first population using alomost all cpu process to accelerate 
import numpy as np
import multiprocessing

population = []
population_size = multiprocessing.cpu_count() - 1

results = []
pool = multiprocessing.Pool(population_size)

for i in range(population_size):
    results.append(pool.apply_async(func=figureInitPopulation))

pool.close()
pool.join()

for result in results:
    population.append(result.get())



## select two individuals for population based on their assessment and probability
def selection(population):
    sum = 0
    populationProbability = []
    for individual in population:
        sum += individual.innerFlow
    for individual in population:
        populationProbability.append(individual.innerFlow / sum)
    individual_1 = rouletteRandom(populationProbability)
    individual_2 = rouletteRandom(populationProbability)
    while individual_1 == individual_2:
            individual_2 = rouletteRandom(populationProbability)
    return individual_1, individual_2

## generate one individuals base on their parents to store good gene
def crossover(individual_1, individual_2):
    stage = 1000
    entity = np.zeros([table_size], dtype = np.int8)
    usedCPUMap = np.zeros([cluster_size], dtype = np.int64)
    clusterTableMap = {}
    for index in range(table_size):
        if index % stage == 0:
            if index + stage <= table_size:
                end = index + stage
            else:
                end = table_size
            if random.randint(0,1) == 0:
                for i in range(index, end):
                    entity[i] = population[individual_1].entity[i]
            else:
                for i in range(index, end):
                    entity[i] = population[individual_2].entity[i]
        usedCPUMap[entity[index]] += table_cpu_cost_df.iloc[index, 2]
        if entity[index] not in clusterTableMap.keys():
            clusterTableMap[entity[index]] = []
        clusterTableMap[entity[index]].append(index)
    return Individual(entity, usedCPUMap, clusterTableMap, False)

## mutate this individuals to extent search space
def mutate(individual):
    mutateProbability = 0.001
    num = int(table_size * mutateProbability)
    for _ in range(num):
        chooseTable = random.randint(0, table_size)
        newCluster = rouletteRandom(clusterRouletteProbability)
        if newCluster != individual.entity[chooseTable]:
            individual.usedCPUMap[individual.entity[chooseTable]] -= table_cpu_cost_df.iloc[chooseTable, 2]
            individual.usedCPUMap[newCluster] += table_cpu_cost_df.iloc[chooseTable, 2]
            individual.clusterTableMap[individual.entity[chooseTable]].remove(chooseTable)
            individual.clusterTableMap[newCluster].append(chooseTable)
            individual.entity[chooseTable] = newCluster
            

## repair this individual to make it feasible after crossover and mutate
def repair(individual):
    while True:
        valid = True
        for index in range(cluster_size):
            while individual.usedCPUMap[index] > cluster_capacity_df.iloc[index, 1]:
                valid = False
                chooseTable = random.choice(individual.clusterTableMap[index])
                newCluster = rouletteRandom(clusterRouletteProbability)
                if newCluster != individual.entity[chooseTable]:
                    individual.usedCPUMap[individual.entity[chooseTable]] -= table_cpu_cost_df.iloc[chooseTable, 2]
                    individual.usedCPUMap[newCluster] += table_cpu_cost_df.iloc[chooseTable, 2]
                    individual.clusterTableMap[individual.entity[chooseTable]].remove(chooseTable)
                    individual.clusterTableMap[newCluster].append(chooseTable)
                    individual.entity[chooseTable] = newCluster
        if valid:
            break

## generate a feasible individual base on parent population
def run():
    individual_1, individual_2 = selection(population)
    individual = crossover(individual_1,individual_2)
    mutate(individual)
    repair(individual)
    individual.figureAssessment()
    return individual


import pickle

breeding_rate = multiprocessing.cpu_count() - 1
# one for 7 minutes
iterations = 500
minCrossFlows = []

## perform <iterations> iterations
## generation <breeding_rate> individuals for each iteration using <breeding_rate> process to accelerate 
    
import os

for times in tqdm(range(iterations), desc='Processing'):
    results = []
    pool = multiprocessing.Pool(breeding_rate)

    for i in range(breeding_rate):
        results.append(pool.apply_async(func=run))

    pool.close()
    pool.join()

    for result in results:
        population.append(result.get())
        
    minInnerFlow = population[0].innerFlow
    minInnerFlowIndex = 0
    minCrossFlow = population[0].crossFlow
    while len(population) > population_size:
        minInnerFlow = population[0].innerFlow
        minInnerFlowIndex = 0
        for index in range(len(population)):
            if population[index].innerFlow < minInnerFlow:
                minInnerFlowIndex = index
            if population[index].crossFlow < minCrossFlow:
                minCrossFlow = population[index].crossFlow
        del population[minInnerFlowIndex]
    
    minCrossFlows.append(minCrossFlow)
    print("----" + str(times + 1) + " iteration----")
    print("minCrossFlow: " +  str(minCrossFlow))
    print("CrossFlows: ")
    for i in population:
        print(i.crossFlow)
    print("--------")    
    isExists= os.path.exists("./placement")
    if not isExists:
        os.mkdir("./placement")
    for index in range(len(population)):
        pickle.dump(population[index], open("./placement/data" + str(index), 'wb'))

    pickle.dump(minCrossFlows,open("./placement/flows", 'wb'))
    
print("----itertions for minCrossFlow----")
print(minCrossFlows)
print("--------")