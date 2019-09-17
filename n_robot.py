import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from MST_code import TSP_solution 

def distance_formula(X1,Y1,X2,Y2):
    distance=math.pow(math.pow(X1-X2,2)+math.pow(Y1-Y2,2),0.5)
    #distance=abs(X1-X2)+abs(Y1-Y2)
    return distance

def set_key(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = value
    elif type(dictionary[key]) == list:
        dictionary[key].append(value)
    else:
        dictionary[key] = [dictionary[key], value]

def plot_mst(file,dict_final,number):
    plt.plot(file[:,1], file[:,2],'*')
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

        if number == 0:
            colour ='r'
        elif number == 1:
            colour ='b'
        elif number == 2:
            colour ='c'
        else :
            colour ='g'
        plt.plot(XX,YY,colour)
    plt.axis([0,80,0,80])
    

def Cost_route(tour):
    Cost=0
    for i in range(len(tour)-1):
        Cost=Cost+distance_formula(file[tour[i]-1,1],file[tour[i]-1,2],file[tour[i+1]-1,1],file[tour[i+1]-1,2])
    return Cost

def plot_clusters(tour,file):
    for j in range(len(tour)):
        number=j%4
        XX=[]
        YY=[]
        for i in range(len(tour[j])):
            XX.append(int(file[tour[j][i]-1,1]))
            YY.append(int(file[tour[j][i]-1,2]))

        if number == 0:
            colour ='r'
        elif number == 1:
            colour ='b'
        elif number == 2:
            colour ='c'
        else :
            colour ='g'
        
        plt.plot(XX,YY,colour)
    plt.axis([0,80,0,80])
    # plt.show()
    plt.pause(0.01)
    plt.clf()
    
        
    


def k_means(n_robots,file,start_point):
    Master_list=[]
    node_centres=[]
    for i in range(n_robots):
        node_centres.append(np.random.randint(1,51))
    # node_centres=[1,1,1]

    for k in range(10):
        Robot_lists=[]
        for i in range(n_robots):
            Robot_lists.append([])
        
        for j in range(len(file)):
            min_distance=distance_formula(file[j,1],file[j,2],file[node_centres[0]-1,1],file[node_centres[0]-1,2])
            index=0
            for i in range(1,n_robots):
                if distance_formula(file[j,1],file[j,2],file[node_centres[i]-1,1],file[node_centres[i]-1,2])<min_distance:
                    index=i
            Robot_lists[index].append(file[j,0])
        #plot_clusters(Robot_lists,file)
        for robot_no in range(len(Robot_lists)):
            Opt_dist=np.inf
            Opt_center=0
            for index in Robot_lists[robot_no]:
                dist=0
                for element in Robot_lists[robot_no]:
                    X1=file[index-1,1]
                    Y1=file[index-1,2]
                    X2=file[element-1,1]
                    Y2=file[element-1,2]
                    dist=dist+distance_formula(X1,Y1,X2,Y2)

                if dist<Opt_dist:
                    Opt_center=index
                    Opt_dist=dist
            
            node_centres[robot_no]=Opt_center
            Master_list=Robot_lists

    
    return Master_list

def optimization_lists(tour,file,start_point):    
    for i in range(np.shape(file)[0]):
        cost=[]
        for j in range(len(tour)):
            print(tour[j])
            cost_=Cost_route(tour[j])
            cost.append(cost_)
        max_cost_index=np.argmax(cost)
        Opt_dist=np.inf
        Opt_center=0
        Opt_robot=0
        for robot_no in range(len(tour)):
            if robot_no!=max_cost_index:
                for index in range(len(tour[max_cost_index])):
                    for element in tour[robot_no]:
                        X1=file[tour[max_cost_index][index]-1,1]
                        Y1=file[tour[max_cost_index][index]-1,2]
                        X2=file[element-1,1]
                        Y2=file[element-1,2]
                        dist=distance_formula(X1,Y1,X2,Y2)

                        if dist<Opt_dist:
                            Opt_center=tour[max_cost_index][index]
                            Opt_dist=dist
                            Opt_robot=robot_no
        
        
        tour[Opt_robot].append(Opt_center)
        tour[max_cost_index].remove(Opt_center)
        plot_clusters(tour,file)
        
    
    return tour

def optimization_equality(tour,file,start_point):
    used=[]
    for i in range(np.shape(file)[0]/2):
        cost=[]
        for j in range(len(tour)):
            print(tour[j])
            cost_=Cost_route(tour[j])
            cost.append(cost_)
        max_cost_index=np.argmax(cost)
        min_cost_index=np.argmin(cost)
        
        Opt_dist=0
        Opt_center=np.inf
        for index in Robot_lists[max_cost_index]:
            dist=0
            for element in Robot_lists[max_cost_index]:
                X1=file[index-1,1]
                Y1=file[index-1,2]
                X2=file[element-1,1]
                Y2=file[element-1,2]
                dist=dist+distance_formula(X1,Y1,X2,Y2)

            if dist>Opt_dist and index not in used:
                Opt_center=index
                Opt_dist=dist
        if Opt_center!=np.inf:
            tour[min_cost_index].append(Opt_center)
            tour[max_cost_index].remove(Opt_center)
            used.append(Opt_center)
        plot_clusters(tour,file)

    return tour

def subset(n_robots,tour,file):
    for i in range(n_robots):
            tour[i].append(start_point)
    Sub_sets=[]
    for robot_no in range(len(tour)):
        Sub_sets=Sub_sets+[np.zeros((len(tour[robot_no]),3))]
        for index in range(len(tour[robot_no])):
            Sub_sets[robot_no][index,0]=tour[robot_no][index]
            Sub_sets[robot_no][index,1]=file[Sub_sets[robot_no][index,0]-1,1]
            Sub_sets[robot_no][index,2]=file[Sub_sets[robot_no][index,0]-1,2]
    return Sub_sets
    
def plot_multi_tour(tour,file):
    plt.scatter(file[:,1], file[:,2])
    for i, txt in enumerate(file[:,0]):
        plt.annotate(txt, (file[:,1][i], file[:,2][i]))

    for j in range(len(tour)):
        number=j%4
        for i in range(len(tour[j])-1):
            XX=[(file[tour[j][i]-1,1]),(file[tour[j][i+1]-1,1])]
            YY=[(file[tour[j][i]-1,2]),(file[tour[j][i+1]-1,2])]
            if number == 0:
                colour ='r'
            elif number == 1:
                colour ='b'
            elif number == 2:
                colour ='c'
            else :
                colour ='g'
            plt.plot(XX,YY,colour)
    plt.axis([0,80,0,80])
    plt.show()
    

#####################################
mst_solver=TSP_solution()
file=mst_solver.import_file('eil101.tsp')
distance_mat=mst_solver.distance_mat(file)
print(file)
n_robots=3
start_point=1
tours=[]
cost=[]
tour_list=[]
Robot_lists=k_means(n_robots,file,start_point)
print(Robot_lists)

Robot_lists=optimization_lists(Robot_lists,file,start_point)
Robot_lists=optimization_equality(Robot_lists,file,start_point)
Robot_lists=subset(n_robots,Robot_lists,file)
for i in range(n_robots):
    tour=mst_solver.pipeline(Robot_lists[i])
    temp=[]
    for j in range(len(tour)):
        temp.append(Robot_lists[i][tour[j]-1,0])
    tour_list.append(temp)
    print(Robot_lists[i])
    print(tour)
print(tour_list)
plot_multi_tour(tour_list,file)
print('##################')
for i in range(n_robots):
    print(Cost_route(tour_list[i]))
print('##################')

'''
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
    plot_mst(file,tours[i],i%4)
    robot=dfs(tours[i],start_point)
    Cost=Cost_route(robot)
    print(Cost)
plt.show()
'''
