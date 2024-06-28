
import pandas as pd
import gurobipy as gp
import numpy as np
from gurobipy import GRB

m = 10
n = 200
base = int(20)
Num_of_Scenarios = 1000

Facility = pd.read_csv('Facility.txt', sep = '\t', header = None)

Cost = pd.read_csv('TransCost.txt', sep = '\t', header = None)
Cost = Cost.T
Demand = pd.read_csv('DemandRand.txt', sep = '\t', header = None)


FacCost = Facility[:][4]

FacPro = Facility[:][3]

Saved_Roots = []
Saved_All_Scenarios = []
Hash_Roots = []
Count_Roots =[0]*Num_of_Scenarios


def Create_Scenarios() :
    return np.random.normal(loc = Demand[:][3], scale = Demand[:][4])

def Solve_For_Each_Scenarios() :
    model =gp.Model('Facility problems')

    Need = Create_Scenarios()

    Choose = model.addVars(m, vtype = GRB.BINARY)

    Product = model.addVars(m, n, lb = 0, vtype = GRB.INTEGER)

    for j in range(n) :
        model.addConstr(gp.quicksum(Product[i, j] for i in range(m)) >= Need[j])

    for i in range(m) :

        model.addConstr(gp.quicksum(Product[i, j] for j in range(n)) <= FacPro[i]*Choose[i])

    model.setObjective(gp.quicksum(Choose[i]*FacCost[i] for i in range(m)) + gp.quicksum(Product[i, j]*Cost[i][j] for i in range(m) for j in range(n)), GRB.MINIMIZE)
    model.optimize()

    CurrentRoot = []
    CurrentPro = []


    for i in range(m) :
        if Choose[i].x > 0 : 
            CurrentRoot.append(i)
            for j in range(n) :
                if(Product[i, j].x > 0):
                    CurrentPro.append([i, j, int(Product[i, j].x)])
    


    Hash = int(0)
    for x in CurrentRoot :
        Hash = Hash + (x + 1)**base
    try :
        Count_Roots[Hash_Roots.index(Hash)] += 1
    except ValueError :
        Hash_Roots.append(Hash)
        Saved_Roots.append(CurrentRoot)
        Count_Roots[len(Hash_Roots) - 1] +=1
    Saved_All_Scenarios.append([model.objVal, CurrentRoot, CurrentPro, Hash])
    


for i in range(Num_of_Scenarios) :
    Solve_For_Each_Scenarios()

print(len(Hash_Roots))
for i in range(len(Hash_Roots)) : print(Count_Roots[i], end = ' ')

print()

for i in range(len(Hash_Roots)) :
    print(Saved_Roots[i])

def SortScenarios(array) :
    return (array[0])

Max = 0
MaxHash = 0
for i in range(len(Hash_Roots)) :
    if Count_Roots[i] > Max:
        Max = Hash_Roots
        MaxHash = Hash_Roots[i]



Saved_All_Scenarios = [x for x in Saved_All_Scenarios if x[3] == MaxHash]

Saved_All_Scenarios = sorted(Saved_All_Scenarios, key = SortScenarios)

Mean_IDX = len(Saved_All_Scenarios)//2

print(Saved_All_Scenarios[Mean_IDX][0])
print(Saved_All_Scenarios[Mean_IDX][1])
for x in Saved_All_Scenarios[Mean_IDX][2] :
    print(x[0]+1, ' ', x[1] + 1, ' ', x[2])
