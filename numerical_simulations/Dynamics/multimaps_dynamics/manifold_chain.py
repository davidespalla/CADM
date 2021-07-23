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
    
    SimulationName="manifold_chain"
    N=1000
    m=5
    gamma=0.5
    L=10.0
    f=0.3
    #Parameters for heteroassociation
    D=2
    
    #CREATE SIMULATION FOLDER
    if not os.path.exists(SimulationName):
        os.makedirs(SimulationName)
    
    #SAVE PARAMETERS:
    outfile=open(SimulationName+"/parameters.txt","w")
    outfile.write("N="+str(N)+"\n")
    outfile.write("m="+str(m)+"\n")
    outfile.write("gamma="+str(gamma)+"\n")
    outfile.write("L="+str(L)+"\n")
    outfile.write("f="+str(f)+"\n")
    outfile.write("D="+str(D)+"\n")
    print("Initializing...")
    
    grid=RegularPfc(N,L,m) # defines environment
    np.save(SimulationName+"/pfc",grid)
    J0=BuildJ0(N,grid,L,gamma) # Builds autoassociative connectivity
    np.save(SimulationName+"/J0",J0)
    G=BuildTransitionMatrix(m,D)
    np.save(SimulationName+"/G",G)
    JH=BuildJH(G,N,grid,L,gamma)  # Builds heteroassociative connectivity
    np.save(SimulationName+"/JH",JH)
    J=J0+JH/float(D)
    #V=np.random.uniform(0,1,N)
    print("Starting dynamics")
    V=correlate_activity(grid[0],L)
    V=V/np.mean(V)
    Vvec,overlaps=dynamicsOvelraps(f,V,N,J,grid,L)
    
    np.save(SimulationName+"/Vdynamics",Vvec)
    np.save(SimulationName+"/Overlaps",overlaps)
        
    print("Dynamics terminated, result saved")
    return

# FUNCTIONS
    
def K(x1,x2,L,gamma):
        d=x1-x2
        return np.exp(-abs(d))+gamma*np.sign(d)*np.exp(-abs(d))
    
def KS(x1,x2,L):
        d=x1-x2
        return np.exp(-abs(d))
    
def overlap(V,pos,L):
    m=0
    for i in range(len(V)):
        for j in range(i):
            m=m+V[i]*V[j]*KS(pos[i],pos[j],L)
    M=len(V)*(len(V)-1)/2.0
    m=m/M
    return m

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
            random.shuffle(tempgrid)
            grid[j][:]=tempgrid
        return grid
    
def BuildJ0(N,grid,L,gamma):
    J=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            for k in range(len(grid)):
                x1=grid[k][i]
                x2=grid[k][j]
                if i!=j:
                    J[i][j]=J[i][j]+K(x1,x2,L,gamma)
    return J
# Builds heteroassociative part of the connectivity
def BuildJH(G,N,grid,L,gamma):
    J=np.zeros((N,N))
    for k in range(len(grid)):
        for l in range(len(grid)):
            if G[k][l]==1:
                for i in range(N):
                    for j in range(N):
                        x1=grid[k][i]
                        x2=grid[l][j]
                        x1h=x1
                        x2h=x2+L
                        J[i][j]=J[i][j]+K(x1h,x2h,L,gamma)
                        J[j][i]=J[j][i]+K(x2h,x1h,L,gamma)
    return J

def BuildTransitionMatrix(m,D):
    G=np.zeros((m,m))
    for i in range(m):
        targets=list(range(m))
        targets.remove(i)
        selected=np.random.choice(targets,D,replace=False)
        for j in selected:
            G[i][j]=1
    return G

# Builds transition matrix with same number of 1s in rows and columns 
def BuildTransitionMatrixMatched(m,D):
    G=np.zeros((m,m))
    for i in range(m):
        for j in range(1,D+1):
            G[i][(i+D*j)%m]=1
    return G
        
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
    
def dynamicsOvelraps(f,V,N,J,grid,L): 
        maxsteps=200
        Vvec=np.zeros((maxsteps,N))
        overlaps=np.zeros((maxsteps,len(grid)))
        for step in range(maxsteps):
            h=np.dot(J,V)
            V=np.asarray(list(map(lambda h: transfer(h),h)))
            V=Sparsify(V,f)
            V=V/np.mean(V)
            Vvec[step][:]=V
            for i in range(len(grid)):
                overlaps[step][i]=overlap(V,grid[i],L)
            if step %10 ==0:
                print("Dynamic step: "+str(step)+" done, mean: "+str(np.mean(V))+" sparsity: "+str(pow(np.mean(V),2)/np.mean(pow(V,2))))
        return Vvec,overlaps

def correlate_activity(pos,L):
    V=np.zeros(len(pos))
    center=L/2
    for i in range(len(V)):
        V[i]=KS(pos[i],center,L)
    return V
    
if __name__ == "__main__":
    main()