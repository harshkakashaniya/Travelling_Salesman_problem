#  BSD 3-Clause License
#  Copyright (c) 2019, Harsh Kakashaniya
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#  * Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 

#  *  @file    Question_1.py
#  *  @author  Harsh Kakashaniya 
#  *  @date    09/15/2019
#  *  @version 1.0
#  *  @copyright BSD 3-Clause
#  *
#  *  @brief Solving TSP using MST 
#  *
#  *  @section Using MST and 2 approximation to solve TSP instance
#  *
#  *  

# All libraries needed in the code.
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import time
import random
import argparse

"""This is a class for TSP solution
"""
class TSP_solution():
    """Constructor of the class.
    """
    def __init__(self):
        self.file='eil51.tsp'
        return
    """This function is used to convert given raw .tsp file to understandable matrix format.

    Returns:
        [matrix] -- [matrix with data of given file name]
    """
    def import_file(self,file_name):
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

    """This function is used to create random node matrix matrix for testing purpose.

    Returns:
        [matrix] -- [matrix with data of given random points]
    """ 
    def random_mat(self,number):
        mat=np.ones((number,3))
        for i in range(number):
            mat[i,0]=i+1
            mat[i,1]= np.random.randint(1,80)
            mat[i,2]= np.random.randint(1,80) 
        return mat

    """Function to make desired output file.

    Returns:
        [None] -- [Saves tour file in directory of program]
    """
    def output_file(self,tour,input_file,cost):
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
    """Function to calculate euclidean distance 

    Returns:
        [double] -- [Returns distance]
    """
    def distance_formula(self,X1,Y1,X2,Y2):
        distance=math.pow(math.pow(X1-X2,2)+math.pow(Y1-Y2,2),0.5)
        #distance=abs(X1-X2)+abs(Y1-Y2)
        return distance

    """Function for getting distances of full list.

    Returns:
        [Matrix] -- [Distance matrix from a node to all other nodes ]
    """
    def distance_mat(self,file):
        # calculate distance
        distance=np.zeros((np.shape(file)[0],np.shape(file)[0]))
        for i in range(np.shape(distance)[0]):
            for j in range(np.shape(distance)[1]):
                distance[i,j]=self.distance_formula(file[i,1],file[i,2],file[j,1],file[j,2])
                if i==j:
                    distance[i,j]=np.inf
        return distance
    """Set key value

    Returns:
        [dictionary] -- [Sets given value to key]
    """
    def set_key(self,dictionary, key, value):
        if key not in dictionary:
            dictionary[key] = [value]
        elif type(dictionary[key]) == list:
            dictionary[key].append(value)
        else:
            dictionary[key] = [dictionary[key], value]

    """Function to calculate MST 

    Returns:
        [Dictionary] -- [Gives dictionary of minimum spanning tree]
    """
    def MST(self,distance_mat):
        MST_dict={}
        visited=[]
        index=np.argmin(distance_mat)
        row_no=int(np.floor(index/np.shape(distance_mat)[0]))
        first_element=row_no+1
        col_no=index%np.shape(distance_mat)[0]
        self.set_key(MST_dict,row_no+1,col_no+1)
        Cost=distance_mat[row_no,col_no]
        distance_mat[:,row_no]=np.inf
        distance_mat[:,col_no]=np.inf
        visited.append(row_no)
        visited.append(col_no)

        for i in range(np.shape(distance_mat)[0]-2):
            minimum=np.inf
            master_index=0
            master_value=0
            for value in visited:
                index=np.argmin(distance_mat[value,:])
                if minimum>distance_mat[value,index]:
                    minimum=distance_mat[value,index]
                    master_index=index
                    master_value=value
            visited.append(master_index)
            Cost=Cost+distance_mat[master_value,master_index]
            distance_mat[:,master_index]=np.inf
            self.set_key(MST_dict,master_value+1,master_index+1)
        return MST_dict,Cost,first_element

    """Plotting of MST when dictionary is provided. 

    Returns:
        [None] -- [Shows MST in graph]
    """
    def plot_mst(self,dict_final,file):
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
        #plt.pause(0.1)
        #plt.clf()
        #plt.close()
    """Cost function for total tour

    Returns:
        [Cost] -- [Returns total cost of given nodes]
    """
    def Cost_route(self,tour,file):
        Cost=0
        for i in range(len(tour)-1):
            Cost=Cost+self.distance_formula(file[tour[i]-1,1],file[tour[i]-1,2],file[tour[i+1]-1,1],file[tour[i+1]-1,2])
        return Cost

    """ Plots given tour

    Returns:
        [None] -- [Plots the given list of points for animations]
    """

    def plot_tour(self,tour,file):
        Cost=0
        plt.scatter(file[:,1], file[:,2])
        for i, txt in enumerate(file[:,0]):
            plt.annotate(txt, (file[:,1][i], file[:,2][i]))

        for i in range(len(tour)-1):
            Cost=Cost+self.distance_formula(file[tour[i]-1,1],file[tour[i]-1,2],file[tour[i+1]-1,1],file[tour[i+1]-1,2])
            XX=[(file[tour[i]-1,1]),(file[tour[i+1]-1,1])]
            YY=[(file[tour[i]-1,2]),(file[tour[i+1]-1,2])]
            plt.plot(XX,YY,'g')
        plt.axis([0,80,0,80])
        plt.pause(0.01)
        plt.clf()
        
    """ Plots given tour

    Returns:
        [None] -- [Plots the given list of points for final tour]
    """
    def plot_final(self,tour,file):
        Cost=0
        plt.scatter(file[:,1], file[:,2])
        for i, txt in enumerate(file[:,0]):
            plt.annotate(txt, (file[:,1][i], file[:,2][i]))

        for i in range(len(tour)-1):
            Cost=Cost+self.distance_formula(file[tour[i]-1,1],file[tour[i]-1,2],file[tour[i+1]-1,1],file[tour[i+1]-1,2])
            XX=[(file[tour[i]-1,1]),(file[tour[i+1]-1,1])]
            YY=[(file[tour[i]-1,2]),(file[tour[i+1]-1,2])]
            plt.plot(XX,YY,'b')
        plt.axis([0,80,0,80])
        plt.show()

    """ Implementation of DFS

    Returns:
        [list] -- [Gives list of tour from giving graph with DFS]
    """

    def dfs(self,dict_final,first_element):
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

    """Swapping series of numbers in given index

    Returns:
        [list] -- [Swipes 2 numbers from given index to n numbers]
    """

    def switcher(self,tour,index,n):
        for i in range(int(n/2)):
            tour[index+i], tour[index+n-i] = tour[index+n-i], tour[index+i]
        return tour

    """ Function for optimization of the tour

    Returns:
        [list] -- [Uses 2 algorithms to optimize the tour length]
    """

    def optimize(self,tour,file):

        #Algorithm 2
        for k in range(5):
            Cost=self.Cost_route(tour,file)
            for j in range(len(tour)-1):
                for i in range(1,len(tour)-1-j):
                    tour=self.switcher(tour,i,j)
                    if self.Cost_route(tour,file)<Cost:
                        #print(Cost)
                        #print(j)
                        self.plot_tour(tour,file)
                        Cost=self.Cost_route(tour,file)
                    else:
                        tour=self.switcher(tour,i,j)
        
        #Algorithm 1
        for k in range(2):
            Cost=self.Cost_route(tour,file)
            for j in range(len(tour)-2):
                for i in range(1,len(tour)-1-j):
                    tour[i],tour[i+j]=tour[i+j],tour[i]
                    if self.Cost_route(tour,file)<Cost:
                        Cost=self.Cost_route(tour,file)
                        self.plot_tour(tour,file)
                    else:
                        tour[i],tour[i+j]=tour[i+j],tour[i]

        return tour
    def pipeline(self,file):
        
        #file=random_mat(300) # random co-ordinates in  file
        distance_matrix=self.distance_mat(file) # distance matrix for all elements in file
        dict_final,MST_Cost,first_element=self.MST(distance_matrix) #  MST for given nodes
        print('Close tour graph to continue the program')
        self.plot_mst(dict_final,file) # plotting the MST
        print(dict_final,'Dictionary') # printing of dictionary
        print(MST_Cost,'Cost of MST') # printing of MST cost
        tour=self.dfs(dict_final,first_element) # Tour with DFS
        self.plot_tour(tour,file) # plot tour
        Cost_Tour=self.Cost_route(tour,file) # cost of tour
        print(Cost_Tour) # print cost
        #start=time.time()
        tour_opt=self.optimize(tour,file) # Function to optimize tour
        #time_taken=time.time()-start # time taken
        #print('tour optimization time : ', time.time()-start) # print time taken for optimization.
        Cost_opt=self.Cost_route(tour_opt,file) # Cost of optimal route
        print('Close tour graph to End the program')
        self.plot_final(tour_opt,file) # plot final tour
        print('$$$$$$$$$$$$$$$$$$$$$$$$$')
        print("Final tour :",tour_opt) # print final tour
        print("Final cost :",Cost_opt) # print final cost
        print('$$$$$$$$$$$$$$$$$$$$$$$$$')
        return tour_opt,Cost_opt


################## Main ######################

if __name__ == "__main__":
    mst =TSP_solution()
    parser = argparse.ArgumentParser() # parser from command line
    parser.add_argument("-input", "--file", required=False,help="first operand")
    args = vars(parser.parse_args())
    try:
        input_file=str(args['file']) # parse data file name
        file=mst.import_file(input_file) # matrix of co-ordinates in file
    except:
        input_file='eil51.tsp' # given data file name
        file=mst.import_file(input_file) # matrix of co-ordinates in file
    tour,Cost_opt=mst.pipeline(file)
    mst.output_file(tour,input_file,Cost_opt) # To save output in .out.tour file

################## End ########################
