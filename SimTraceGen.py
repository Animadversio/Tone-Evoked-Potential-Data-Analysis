# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 18:43:47 2017

@author: Compute

Simulated Voltage trace 

"""
import numpy as np
times
#from PSPFitting import alpha
def alpha(t,t0,A,tau1,tau2):
    '''Basic alpha function '''
    Msk=t<t0
    Rsp=A*np.exp(-(t-t0)/tau1)*(1-np.exp(-(t-t0)/tau2))
    Rsp[Msk]=0
    return Rsp

def SpikeTrainGen(Lambda):
    