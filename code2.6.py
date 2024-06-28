import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np

def distance_point(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def k_means_with_weights(data, weights, k, max_iterations=300):
    centroids_idx = np.random.choice(len(data), k, replace=False)
    centroids_idx = [ 99, 131, 164, 157 , 14]
    print(centroids_idx)
    centroids = data[centroids_idx]
    
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        cluster_weights = np.zeros(k)
        
        for i, point in enumerate(data):
            min_distance = float('inf')
            closest_centroid = None
            for j, centroid in enumerate(centroids):
                distance = distance_point(point, centroid)
                if distance < min_distance and cluster_weights[j] + weights[i] <= 1000 :
                    min_distance = distance
                    closest_centroid = j
            if closest_centroid is not None:
              clusters[closest_centroid].append(i)
              cluster_weights[closest_centroid] += weights[i]
            else :
              assigned = 0
              for j in range(k) :
                if assigned == 1 : break
                if cluster_weights[j] + weights[i] > 1000 :
                  for t in clusters[j] :
                    if cluster_weights[j] - weights[t] + weights[i] <= 1000 and distance_point(data[t], centroids[j]) < distance_point(data[i], centroids[j]) :
                      cluster_weights[j] -= weights[t]
                      clusters[j].remove(t)
                      cluster_weights[j] += weights[i]
                      clusters[j].append(i)
                      assigned = 1
                      break
        
        new_centroids = np.zeros_like(centroids)
        for j, cluster in enumerate(clusters):
            if len(cluster) > 0:
                cluster_sum = np.zeros_like(data[0])
                weight_sum = 0
                for idx in cluster:
                    cluster_sum += weights[idx] * data[idx]
                    weight_sum += weights[idx]
                new_centroids[j] = cluster_sum / weight_sum
            else:
                new_centroids[j] = centroids[j]
        
        if np.allclose(new_centroids, centroids):
            break
        
        centroids = new_centroids
    
    return clusters, centroids

def plot_clusters(data, clusters, centroids=None):
    colors = ['r', 'g', 'b', 'y', 'c', 'm'] 
    plt.figure(figsize=(8, 6))
    
    for i, cluster in enumerate(clusters):
        cluster_points = data[cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'VÙNG {i+1}')
    
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='k', label='TÂM')
    
    plt.title('PHÂN VÙNG DỮ LIỆU')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

data = np.loadtxt('Demand.txt', usecols=(1, 2, 3))  

weights = data[:, 2]

data = data[:, :2]

k = 5



clusters, centroids = k_means_with_weights(data, weights, k)

for j in range(k) :
    cluster_points = clusters[j]
    sorted_indices = sorted(cluster_points, key=lambda x: weights[x])
    print(sorted_indices[0])

cnt = [0]*200

ClustersSum = [0]*5

for cl in clusters :
  sum = 0
  for id in cl :
    sum += weights[id]
    cnt[id]+=1
  ClustersSum[clusters.index(cl)] = sum
  print(sum)
for i in range(200) : 
  if cnt[i] == 0 : print(i)
plot_clusters(data, clusters, centroids)



model = gp.Model()

k = 5
F = 608

X_f = model.addVars(k, lb=1, ub=9, vtype=GRB.CONTINUOUS)
Y_f = model.addVars(k, lb=0, ub=9, vtype=GRB.CONTINUOUS)

rect = [
    [[3.5, 5.2], [4.8, 6.2]],
    [[2.7, 3.3], [3.0, 3.8]],
    [[3.2, 3.0], [3.8, 3.4]],
    [[4.7, 4.2], [5.0, 4.5]],
    [[5.9, 2.9], [6.2, 3.8]]
]

for i in range(k):
    for j in clusters[i]:
        model.addConstr((X_f[i] - data[j][0])**2 + (Y_f[i] - data[j][1])**2 >= 0.04)

for i in range(k):
    model.addConstr((X_f[i] - 6.25)**2 + (Y_f[i] - 7.45)**2 >= (0.75)**2 + 0.1)

for i in range(k):
    model.addConstr((X_f[i] - 4.15)**2/(0.65**2) + (Y_f[i] - 5.7)**2/(0.5**2) >= 1.4)

for i in range(k):
    model.addConstr((X_f[i] - 2.85)**2/(0.15**2) + (Y_f[i] - 3.55)**2/(0.25**2) >= 1.4)

for i in range(k):
    model.addConstr((X_f[i] - 3.5)**2/(0.3**2) + (Y_f[i] - 3.2)**2/(0.2**2) >= 1.4)

for i in range(k):
    model.addConstr((X_f[i] - 4.85)**2/(0.15**2) + (Y_f[i] - 4.35)**2/(0.15**2) >= 1.4)
for i in range(k):
    model.addConstr((X_f[i] - 6.05)**2/(0.15**2) + (Y_f[i] - 3.35)**2/(0.45**2) >= 1.4)


DistSqr = model.addVars(k, 200, vtype=GRB.CONTINUOUS)

for i in range(k):
    for j in range(200):
        model.addConstr(DistSqr[i, j] == (X_f[i] - data[j][0])**2 + (Y_f[i] - data[j][1])**2)


M = model.addVars(k, lb=0, vtype=GRB.CONTINUOUS)

for i in range(k):
    model.addConstr(M[i] == gp.quicksum(DistSqr[i, j] * weights[j] for j in clusters[i]))

model.setObjective(gp.quicksum(M[i] for i in range(k)), GRB.MINIMIZE)

model.optimize()

X_f_values = [X_f[i].X for i in range(k)]
Y_f_values = [Y_f[i].X for i in range(k)]

Sum_Cost = float(0)

for i in range(len(X_f_values)) :
   print(X_f_values[i], ' ', Y_f_values[i])
   for j in clusters[i] :
      Sum_Cost += (np.sqrt((X_f_values[i] - data[j][0])**2 + (Y_f_values[i] - data[j][1])**2)*weights[j])

Sum_Cost += k*F

print(Sum_Cost)

for i in range(k) :
   print(M[i].x , ' ', X_f[i].x , ' ', Y_f[i].x)

fig, ax = plt.subplots()

for rect_coords_min, rect_coords_max in rect:
    x_min, y_min = rect_coords_min
    x_max, y_max = rect_coords_max
    rect_patch = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect_patch)

circle = plt.Circle((6.25, 7.45), 0.75, color='b', fill=False)
ax.add_patch(circle)

ax.scatter(X_f_values, Y_f_values, color='g')

ax.scatter(data[:, 0], data[:, 1], color='black', marker='.')

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Visualization of Constraints and Points')
plt.show()
