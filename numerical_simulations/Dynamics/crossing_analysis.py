#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:31:07 2019

@author: davide
"""
import numpy as np
import math
#import random
import os



def main():

    SimulationName="crossing_study"
    nl=30
    N=nl*nl
    m=1
    gammas=np.linspace(0.05,1,20)
    L=10.0
    f=0.05
    dir_flag=0 # 0= ellipse elognated along x, 1= elliple elongated along y
    order=15
    power=0
    xi=25

    #CREATE SIMULATION FOLDER
    if not os.path.exists(SimulationName):
        os.makedirs(SimulationName)
        
     #SAVE PARAMETERS:
    outfile=open(SimulationName+"/parameters.txt","w")
    outfile.write("N="+str(N)+"\n")
    outfile.write("L="+str(L)+"\n")
    outfile.write("f="+str(f)+"\n")
    outfile.close()
    
    np.save(SimulationName+"/gammas",gammas)
    
    for i in range(len(gammas)):   
        subfolder=SimulationName+"/gamma_"+str(i+1)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        
        print("Starting dynamics for gamma="+str(gammas[i]))
        grid=RegularPfc(N,L,m) # defines environment
        np.save(subfolder+"/pfc",grid)
        J=BuildJ(N,grid,L,gammas[i],order,power,xi) # Builds connectivity
        #V=np.random.uniform(0,1,N)
        starting_point1=[L/2,0] # starting on bottom
        starting_point2=[0,L/2.0] # starting on left
        V=correlate_activity(grid[0],starting_point2,L)
        V=V/np.mean(V)
        Vvec=dynamics(f,V,N,J)
        np.save(subfolder+"/Vdynamics",Vvec)
        print("Dynamics terminated, result saved")
        
    print("END")
    return

# FUNCTIONS

def KAx(x1,x2,L,order,power,xi):
    
        dx=x1[0]-x2[0]
        dy=x1[1]-x2[1]
        if dx>float(L)/2.0:
            dx=dx-L
        elif dx<-float(L)/2.0:
            dx=dx+L
        if dy>float(L)/2.0:
            dy=dy-L
        elif dy<-float(L)/2.0:
            dy=dy+L
    
        #cosine power kernel
        r=np.sqrt(pow(dx,2)+pow(dy,2))
        theta=np.arctan2(dy,dx)
        
        k_out= pow(np.cos(theta),order)
        k_out=k_out*pow(r,power)*np.exp(-pow(r,2)/xi)
        
        
        return k_out
    
def KAy(x1,x2,L,order,power,xi):
        
        dx=x1[0]-x2[0]
        dy=x1[1]-x2[1]
        if dx>float(L)/2.0:
            dx=dx-L
        elif dx<-float(L)/2.0:
            dx=dx+L
        if dy>float(L)/2.0:
            dy=dy-L
        elif dy<-float(L)/2.0:
            dy=dy+L
    
        #cosine power kernel
        r=np.sqrt(pow(dx,2)+pow(dy,2))
        theta=np.arctan2(dy,dx)+(math.pi/2.0)
        
        k_out= pow(np.cos(theta),order)
        k_out=k_out*pow(r,power)*np.exp(-pow(r,2)/xi)
        
        
        return k_out

def KS(x1,x2,L):
        dx=x1[0]-x2[0]
        dy=x1[1]-x2[1]
        if dx>float(L)/2.0:
            dx=dx-L
        elif dx<-float(L)/2.0:
            dx=dx+L
        if dy>float(L)/2.0:
            dy=dy-L
        elif dy<-float(L)/2.0:
            dy=dy+L
        d=np.sqrt(pow(dx,2)+pow(dy,2))
        return np.exp(-abs(d))
    
def KSellipse(x1,x2,L,a1,a2):
        dx=x1[0]-x2[0]
        dy=x1[1]-x2[1]
        if dx>float(L)/2.0:
            dx=dx-L
        elif dx<-float(L)/2.0:
            dx=dx+L
        if dy>float(L)/2.0:
            dy=dy-L
        elif dy<-float(L)/2.0:
            dy=dy+L
        d=np.sqrt(a1*pow(dx,2)+a2*pow(dy,2))
        return np.exp(-abs(d)) 


def transfer(h):
        if h>0:
            return h
        else:
            return 0

def RegularPfc(N,L,m):
        Nl=int(np.sqrt(N))
        grid=np.zeros((m,N,2))
        tempgrid=np.zeros((N,2))
        for i in range(Nl):
            for j in range(Nl):
                tempgrid[i+Nl*j][0]=i*float(L)/float(Nl)
                tempgrid[i+Nl*j][1]=j*float(L)/float(Nl)

        for j in range(m):
            labels=np.random.permutation(N)
            for k in range(N):
                grid[j][:]=tempgrid

        return grid

def BuildJ(N,grid,L,gamma,order,power,xi):
    width=L/10
    J=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            for k in range(len(grid)):
                x1=grid[k][i]
                x2=grid[k][j]
                if i!=j:
                    J[i][j]=J[i][j]+KS(x1,x2,L)
                if ((L/2.0-width)<x1[0]<(L/2.0+width)) and ((L/2.0-width)<x2[0]<(L/2.0+width)):
                    J[i][j]=J[i][j]+gamma*KAy(x1,x2,L,order,power,xi)
                
                if ((L/2.0-width)<x1[1]<(L/2.0+width)) and ((L/2.0-width)<x2[1]<(L/2.0+width)):
                    J[i][j]=J[i][j]+gamma*KAx(x1,x2,L,order,power,xi)
    
    return J

def Sparsify(V,f):
        vout=V
        th=np.percentile(V,(1.0-f)*100)
        for i in range(len(V)):
            if vout[i]<th:
                vout[i]=0
            else:
                vout[i]=vout[i]-th
        return vout

def dynamics(f,V,N,J):
        maxsteps=200
        Vvec=np.zeros((maxsteps,N))
        for step in range(maxsteps):
            h=np.dot(J,V)
            V=np.asarray(list(map(lambda h: transfer(h),h)))
            V=Sparsify(V,f)
            V=V/np.mean(V)
            Vvec[step][:]=V
            #print("Dynamic step: "+str(step)+" done, mean: "+str(np.mean(V))+" sparsity: "+str(pow(np.mean(V),2)/np.mean(pow(V,2))))
        return Vvec

def correlate_activity(pos,center,L):
	V=np.zeros(len(pos))
	for i in range(len(V)):
		V[i]=KS(pos[i],center,L)
	return V

def correlate_activity_ellipse(pos,L,flag):
    if flag==0:
        a1=6
        a2=1
    
    if flag==1:
        a1=1
        a2=6
    
    V=np.zeros(len(pos))
    center=np.array([L/2, L/2])
    for i in range(len(V)):
        V[i]=KSellipse(pos[i],center,L,a1,a2)
    return V

if __name__ == "__main__":
    main()
