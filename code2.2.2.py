import pandas as pd
import gurobipy as gp
from gurobipy import GRB

m = 10
n = 200

Facility = pd.read_csv('FacilityMultiple.txt', sep = '\t', header = None)
Cost = pd.read_csv('TransCost.txt', sep = '\t', header = None)
Demand = pd.read_csv('Demand.txt', sep = '\t', header = None)

Cost = Cost.T

Need = Demand[:][3]

FacCost = pd.DataFrame.from_records([[None]*10]*3)
FacPro = pd.DataFrame.from_records([[None]*10]*3)

for i in range(30) :
    FacPro[i%m][i//m]= Facility[3][i]

for i in range(30) :
    FacCost[i%m][i//m]= Facility[4][i]

print(FacCost)
print(FacPro)

model = gp.Model('Facility problems')

Choose = model.addVars(m, 3, vtype = GRB.BINARY)

Product = model.addVars(m, n, lb = 0, vtype = GRB.INTEGER)


for j in range(n) :
    model.addConstr(gp.quicksum(Product[i, j] for i in range(m)) == Need[j])

for i in range(m) :
    model.addConstr(gp.quicksum(Choose[i, t] for t in range(3)) <= 1)

for i in range(m) :
    model.addConstr(gp.quicksum(Product[i, j] for j in range(n)) <= gp.quicksum(FacPro[i][t]*Choose[i, t] for t in range(3)))


model.setObjective(gp.quicksum(Choose[i, t]*FacCost[i][t] for i in range(m) for t in range(3)) + gp.quicksum(Product[i, j]*Cost[i][j] for i in range(m) for j in range(n)), GRB.MINIMIZE)
model.optimize()

print(model.getObjective())



for i in range(m) :
    for t in range(3) :
      if(Choose[i, t].x > 0) :
          print(i + 1, " ", FacCost[i][t], " ", gp.quicksum(Product[i, j].x for j in range(n)))


for i  in range(m) : 
    for j in range(n) : 
        if Product[i, j].x > 0 : print(i+1, ' ', j+1, ' ', Product[i, j].x)
print(model.ObjVal)
