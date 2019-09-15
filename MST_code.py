import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import time
import random

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

def random_mat(number):
    mat=np.ones((number,3))
    for i in range(number):
        mat[i,0]=i+1
        mat[i,1]= np.random.randint(1,80)
        mat[i,2]= np.random.randint(1,80) 
    return mat

def output_file(tour,input_file,cost):
    output_file=input_file[:-4]+'.out.tour'
    output_file_name='eil76.out.tour'
    f= open(output_file,"w+")
    f.write("NAME : %s \n"% output_file)
    f.write("COMMENT : Optimal tour for %s (" % input_file)
    f.write("%s)\n" % np.round(cost,2))
    f.write("TYPE : TOUR \n")
    if len(output_file)==14:
        f.write("DIMENSION : %s\n" % output_file[3:5])
    else :
        f.write("DIMENSION : %s\n" % output_file[3:6])
    f.write("TOUR_SECTION\n")
    for item in tour:
        f.write("%s\n" % item)
    f.write("-1 \n")
    f.write("EOF \n")

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
    plt.show()
    plt.pause(1)
    plt.clf()
    # plt.close()

def Cost_route(tour):
    Cost=0
    for i in range(len(tour)-1):
        Cost=Cost+distance_formula(file[tour[i]-1,1],file[tour[i]-1,2],file[tour[i+1]-1,1],file[tour[i+1]-1,2])
    return Cost

def plot_tour(tour):
    Cost=0
    plt.scatter(file[:,1], file[:,2])
    for i, txt in enumerate(file[:,0]):
        plt.annotate(txt, (file[:,1][i], file[:,2][i]))

    for i in range(len(tour)-1):
        Cost=Cost+distance_formula(file[tour[i]-1,1],file[tour[i]-1,2],file[tour[i+1]-1,1],file[tour[i+1]-1,2])
        XX=[(file[tour[i]-1,1]),(file[tour[i+1]-1,1])]
        YY=[(file[tour[i]-1,2]),(file[tour[i+1]-1,2])]
        plt.plot(XX,YY,'g')
    plt.axis([0,80,0,80])
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

    #Algorithm 2
    for k in range(5):
        Cost=Cost_route(tour)
        for j in range(len(tour)-1):
            for i in range(1,len(tour)-1-j):
                tour=switcher(tour,i,j)
                if Cost_route(tour)<Cost:
                    #print(Cost)
                    #print(j)
                    #plot_tour(tour)
                    Cost=Cost_route(tour)
                else:
                    tour=switcher(tour,i,j)
    
    #Algorithm 1
    for k in range(2):
        Cost=Cost_route(tour)
        for j in range(len(tour)-2):
            for i in range(1,len(tour)-1-j):
                tour[i],tour[i+j]=tour[i+j],tour[i]
                if Cost_route(tour)<Cost:
                    Cost=Cost_route(tour)
                    #plot_tour(tour)
                else:
                    tour[i],tour[i+j]=tour[i+j],tour[i]

    return tour

################## Main ######################
master_file=[]
for i in range(10):
    input_file='eil51.tsp'
    #file=import_file(input_file)
    file=random_mat(300)
    distance_matrix=distance_mat(file)
    dict_final,MST_Cost,first_element=MST(distance_matrix)
    # plot_mst(dict_final)
    print(dict_final,'Dictionary')
    print(MST_Cost,'Cost of MST')
    tour=dfs(dict_final,first_element)
    # plot_final(tour)
    Cost_Tour=Cost_route(tour)
    print(Cost_Tour)
    start=time.time()
    tour_opt=optimize(tour)
    time_taken=time.time()-start
    print('tour optimization time : ', time.time()-start)

    Cost_opt=Cost_route(tour_opt)
    print(Cost_opt)
    # plot_final(tour_opt)
    master_file.append([i+1,np.round(MST_Cost,1),np.round(Cost_Tour,1),np.round(Cost_opt,1),np.round(time_taken,1)])
print(master_file)
#output_file(tour_opt,input_file,Cost_opt)
################## End ########################
