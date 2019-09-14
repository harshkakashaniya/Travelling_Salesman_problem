import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools

def import_file(file_name):
    file =[]
    with open(file_name) as tsvfile:
      reader = csv.reader(tsvfile, delimiter=' ')
      for row in reader:
        file.append(row)
    # delete first 6 lines for data and last line
    del file[0:6]
    del file[-1]
    file_mat=np.zeros((np.shape(file)[0],np.shape(file)[1]))
    # Convert list to matrix
    for a in range(0,np.shape(file)[0]):
        for b in range(0,np.shape(file)[1]):
            file_mat[a,b]=file[a][b]
    return file_mat

def distance_formula(X1,Y1,X2,Y2):
    distance=math.pow(math.pow(X1-X2,2)+math.pow(Y1-Y2,2),0.5)
    #distance=abs(X1-X2)+abs(Y1-Y2)
    return distance

def distance_mat(file):
    # calculate distance
    distance=np.zeros((np.shape(file)[0],np.shape(file)[0]))
    for i in range(np.shape(distance)[0]):
        for j in range(np.shape(distance)[1]):
            distance[i,j]=distance_formula(file[i,1],file[i,2],file[j,1],file[j,2])
            if i==j:
                distance[i,j]=np.inf
    return distance

def set_key(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = value
    elif type(dictionary[key]) == list:
        dictionary[key].append(value)
    else:
        dictionary[key] = [dictionary[key], value]

def plot_mst(file,dict_final):
    plt.scatter(file[:,1], file[:,2])
    for i, txt in enumerate(file[:,0]):
        plt.annotate(txt, (file[:,1][i], file[:,2][i]))
    x= []
    y= []
    for k, v in dict_final.iteritems():
        if isinstance(v, list):
            x.extend(list(itertools.repeat(k, len(v))))
            y.extend(v)
        else:
            v=[v]
            x.extend(list(itertools.repeat(k, len(v))))
            y.extend(v)

    for i in range(len(x)):
        XX=[(file[x[i]-1,1]),(file[y[i]-1,1])]
        YY=[(file[x[i]-1,2]),(file[y[i]-1,2])]
        plt.plot(XX,YY)
    plt.axis([0,80,0,80])
    plt.show()

def Cost_route(tour):
    Cost=0
    for i in range(len(tour)-1):
        Cost=Cost+distance_formula(file[tour[i]-1,1],file[tour[i]-1,2],file[tour[i+1]-1,1],file[tour[i+1]-1,2])
    return Cost

def dfs(dict_final,first_element):
    tour_stack=[]
    tour=[]
    tour_stack.append(first_element)
    while tour_stack!=[]:
        elememt=tour_stack.pop()
        tour.append(elememt)
        if elememt in dict_final:
            if tour_stack!=[]:
                tour_stack=tour_stack+dict_final[elememt]
            else:
                tour_stack=dict_final[elememt]
    tour.append(first_element)
    print(tour,'tour')
    return tour

#####################################
file=import_file('eil51.tsp')
distance_mat=distance_mat(file)
print(file)
n_robots=3
start_point=1
tours=[]
cost=[]
for i in range(n_robots):
    min=np.argmin(distance_mat[start_point-1,:])
    cost.append(distance_mat[start_point-1,min])
    distance_mat[:,min]=np.inf
    distance_mat[:,start_point-1]=np.inf
    tours.append({start_point:min+1})

while np.min(distance_mat)!=np.inf:
    robot_num=np.argmin(cost)
    edge=start_point
    dict_edge=tours[robot_num]
    while edge in dict_edge:
        edge=dict_edge[edge]
    min=np.argmin(distance_mat[edge-1,:])
    cost[robot_num]= cost[robot_num]+ distance_mat[edge-1,min]
    distance_mat[:,min]=np.inf
    set_key(tours[robot_num], edge, min+1)

print(tours)

for i in range(len(tours)):
    plot_mst(file,tours[i])
    robot=dfs(tours[i],start_point)
    Cost=Cost_route(robot)
    print(Cost)
