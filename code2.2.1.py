
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

m = 10
n = 200

Facility = pd.read_csv('Facility.txt', sep = '\t', header = None)
Cost = pd.read_csv('TransCost.txt', sep = '\t', header = None)
Demand = pd.read_csv('Demand.txt', sep = '\t', header = None)

Cost = Cost.T

Need = Demand[:][3]

FacCost = Facility[:][4]

FacPro = Facility[:][3]

model =gp.Model('Facility problems')

Choose = model.addVars(m, vtype = GRB.BINARY)

Product = model.addVars(m, n, lb = 0, vtype = GRB.INTEGER)

for j in range(n) :
    model.addConstr(gp.quicksum(Product[i, j] for i in range(m)) == Need[j])

for i in range(m) :
    model.addConstr(gp.quicksum(Product[i, j] for j in range(n)) <= FacPro[i]*Choose[i])


model.setObjective(gp.quicksum(Choose[i]*FacCost[i] for i in range(m)) + gp.quicksum(Product[i, j]*Cost[i][j] for i in range(m) for j in range(n)), GRB.MINIMIZE)
model.optimize()

for i in range(m) :
    if(Choose[i].x > 0) :
        print(i+1, " ", FacCost[i], " ", gp.quicksum(Product[i, j].x for j in range(n)))

for i  in range(m) : 
    for j in range(n) :
        if Product[i, j].x > 0 : print(i+1, ' ' , j+1, ' ', Product[i, j].x)

print(model.ObjVal)


