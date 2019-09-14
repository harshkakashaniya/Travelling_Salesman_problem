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
        dictionary[key] = [value]
    elif type(dictionary[key]) == list:
        dictionary[key].append(value)
    else:
        dictionary[key] = [dictionary[key], value]

def MST(distance_mat):
    MST_dict={}
    visited=[]
    index=np.argmin(distance_mat)
    row_no=int(np.floor(index/np.shape(distance_mat)[0]))
    first_element=row_no+1
    col_no=index%np.shape(distance_mat)[0]
    set_key(MST_dict,row_no+1,col_no+1)
    Cost=distance_mat[row_no,col_no]
    distance_mat[:,row_no]=np.inf
    distance_mat[:,col_no]=np.inf
    visited.append(row_no)
    visited.append(col_no)

    for i in range(np.shape(distance_mat)[0]-2):
        minimum=np.inf
        for value in visited:
            index=np.argmin(distance_mat[value,:])
            if minimum>distance_mat[value,index]:
                minimum=distance_mat[value,index]
                master_index=index
                master_value=value
        visited.append(master_index)
        Cost=Cost+distance_mat[master_value,master_index]
        distance_mat[:,master_index]=np.inf
        set_key(MST_dict,master_value+1,master_index+1)
    return MST_dict,Cost,first_element

def plot_mst(dict_final):
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
        plt.plot(XX,YY,'r')
    plt.axis([0,80,0,80])
    # plt.show()
    plt.pause(1)
    plt.clf()
    # plt.close()

def Cost_route(tour):
    Cost=0
    for i in range(len(tour)-1):
        Cost=Cost+distance_formula(file[tour[i]-1,1],file[tour[i]-1,2],file[tour[i+1]-1,1],file[tour[i+1]-1,2])
    return Cost

def plot_tour(tour,i,j):
    Cost=0
    plt.scatter(file[:,1], file[:,2])
    for i, txt in enumerate(file[:,0]):
        plt.annotate(txt, (file[:,1][i], file[:,2][i]))

    for i in range(len(tour)-1):
        Cost=Cost+distance_formula(file[tour[i]-1,1],file[tour[i]-1,2],file[tour[i+1]-1,1],file[tour[i+1]-1,2])
        XX=[(file[tour[i]-1,1]),(file[tour[i+1]-1,1])]
        YY=[(file[tour[i]-1,2]),(file[tour[i+1]-1,2])]
        plt.plot(XX,YY,'g')
    # changes Connection
    #XX=[(file[tour[i]-1,1]),(file[tour[j]-1,1])]
    #YY=[(file[tour[i]-1,2]),(file[tour[j]-1,2])]
    plt.axis([0,80,0,80])
    #plt.plot(XX,YY,'y')
    # plt.show()
    plt.pause(0.01)
    plt.clf()

def plot_final(tour):
    Cost=0
    plt.scatter(file[:,1], file[:,2])
    for i, txt in enumerate(file[:,0]):
        plt.annotate(txt, (file[:,1][i], file[:,2][i]))

    for i in range(len(tour)-1):
        Cost=Cost+distance_formula(file[tour[i]-1,1],file[tour[i]-1,2],file[tour[i+1]-1,1],file[tour[i+1]-1,2])
        XX=[(file[tour[i]-1,1]),(file[tour[i+1]-1,1])]
        YY=[(file[tour[i]-1,2]),(file[tour[i+1]-1,2])]
        plt.plot(XX,YY,'b')
    plt.axis([0,80,0,80])
    plt.show()



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

def switcher(tour,index,n):
    for i in range(int(n/2)):
        tour[index+i], tour[index+n-i] = tour[index+n-i], tour[index+i]
    return tour

def optimize(tour):

    for k in range(int(len(tour)/2)):
        Cost=Cost_route(tour)
        for j in range(len(tour)-1):
            for i in range(1,len(tour)-1-j):
                tour=switcher(tour,i,j)
                if Cost_route(tour)<Cost:
                    print(Cost)
                    print(j)
                    plot_tour(tour,i,j)
                    Cost=Cost_route(tour)
                else:
                    tour=switcher(tour,i,j)

    for k in range(3):
        Cost=Cost_route(tour)
        for j in range(len(tour)-2):
            for i in range(1,len(tour)-1-j):
                tour[i],tour[i+j]=tour[i+j],tour[i]
                if Cost_route(tour)<Cost:
                    Cost=Cost_route(tour)
                    plot_tour(tour,i,j)
                else:
                    tour[i],tour[i+j]=tour[i+j],tour[i]
    return tour

################## Main ######################

file=import_file('eil51.tsp')
distance_mat=distance_mat(file)
dict_final,Cost,first_element=MST(distance_mat)
plot_mst(dict_final)
print(dict_final,'Dictionary')
print(Cost,'Cost of MST')
tour=dfs(dict_final,first_element)
Cost=Cost_route(tour)
print(Cost)
tour_opt=optimize(tour)
Cost_opt=Cost_route(tour_opt)
print(Cost_opt)
plot_final(tour_opt)
################## End ########################
