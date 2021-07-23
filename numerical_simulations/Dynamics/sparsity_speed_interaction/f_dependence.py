#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:31:07 2019

@author: davide
"""
import numpy as np
import random
import os



def main():

    SimulationName="f_dependence"
    N=1000
    m=1
    gammas=np.linspace(0.1,2,10)
    L=10.0
    fs=np.linspace(0.001,0.7,20)

    #CREATE SIMULATION FOLDER
    if not os.path.exists(SimulationName):
        os.makedirs(SimulationName)

    #SAVE PARAMETERS
    of = open(SimulationName+"/parameters.txt", "w")
    of.write("N="+str(N)+"\n")
    of.write("L="+str(L)+"\n")
    of.close()
    np.save(SimulationName+"/fs",fs)
    np.save(SimulationName+"/gammas",gammas)


    print("Initializing network")

    grid=RegularPfc(N,L,m) # defines environment
    np.save(SimulationName+"/pfc",grid)
   

    print("Starting dynamics")
    Vvec=np.zeros((len(gammas),len(fs),200,N))
    for i in range(len(gammas)):
        J=BuildJ(N,grid,L,gammas[i]) # Builds connectivity
        np.save(SimulationName+"/J"+str(i+1),J)
        for j in range(len(fs)):
            V=correlate_activity(grid[0],L)
            V=V/np.mean(V)
            Vvec[i][j][:]=dynamics(fs[j],V,N,J)
    
    np.save(SimulationName+"/Vdynamics",Vvec)


    print("Dynamics terminated, result saved")
    return

# FUNCTIONS

def K(x1,x2,L,gamma):
        d=x1-x2
        if d>float(L)/2.0:
            d=d-L
        elif d<-float(L)/2.0:
            d=d+L
        return np.exp(-abs(d))+gamma*np.sign(d)*np.exp(-abs(d))

def KS(x1,x2,L):
        d=x1-x2
        if d>float(L)/2.0:
            d=d-L
        elif d<-float(L)/2.0:
            d=d+L
        return np.exp(-abs(d))

def transfer(h):
        if h>0:
            return h
        else:
            return 0

def RegularPfc(N,L,m):
        grid=np.zeros((m,N))
        tempgrid=np.zeros(N)
        for i in range(N):
            tempgrid[i]=i*float(L)/float(N)
        for j in range(m):
            #random.shuffle(tempgrid)
            grid[j][:]=tempgrid
        return grid

def BuildJ(N,grid,L,gamma):
    J=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            for k in range(len(grid)):
                x1=grid[k][i]
                x2=grid[k][j]
                if i!=j:
                    J[i][j]=J[i][j]+K(x1,x2,L,gamma)
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

def correlate_activity(pos,L):
    V=np.zeros(len(pos))
    center=L/2
    for i in range(len(V)):
        V[i]=KS(pos[i],center,L)
    return V

if __name__ == "__main__":
    main()
