# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:38:52 2017

@author: Compute

Non-linear fitting to extract


Documents on function see 
os.system("start winword \"D:\\NYUReyesLab\\Tone Evoked Response\\TEP Proc Report 1 EPSP extraction.docx\"")
os.system("start \"C:\\Program Files (x86)\\Adobe\\Acrobat 11.0\\Acrobat\\acrord32.exe\" \"D:\\NYUReyesLab\\Tone Evoked Response\\Tone Evoked Potential Processing Notes.pdf\"")
os.system("start chrome \"www.bing.com\"")
"""
import numpy as np
from numpy import inf
import scipy.optimize as optm
import matplotlib.pyplot as plt
from time import time
from scipy.optimize import curve_fit

from ReReadin import ReReadin_Data

def alpha(t,t0,A,tau1,tau2):
    '''Basic alpha function '''
    Msk=t<t0
    Rsp=A*np.exp(-(t-t0)/tau1)*(1-np.exp(-(t-t0)/tau2))
    Rsp[Msk]=0
    return Rsp

def alphabl(t,t0,A,tau1,tau2,bl):
    '''alpha function with baseline parameter'''
    Msk=t<t0
    Rsp=A*(np.exp(-(t-t0)/tau1)-np.exp(-(t-t0)/tau2))+bl
    Rsp[Msk]=bl
    return Rsp

#def alphabl5(t,t0_1,A_1,tau1_1,tau2_1,t0_2,A_2,tau1_2,tau2_2,t0_3,A_3,tau1_3,tau2_3,t0_4,A_4,tau1_4,tau2_4,t0_5,A_5,tau1_5,tau2_5,bl):
#    Rsp=alpha(t,t0_1,A_1,tau1_1,tau2_1)+alpha(t,t0_2,A_2,tau1_2,tau2_2)+alpha(t,t0_3,A_3,tau1_3,tau2_3)+alpha(t,t0_4,A_4,tau1_4,tau2_4)+alpha(t,t0_5,A_5,tau1_5,tau2_5)+bl
#    return Rsp

def multialpha(t,t0,A,tau1,tau2,bl):
    '''composite alpha function with baseline 
    each parameter t0, A, tau1, tau2 can be a list of these parameters length n 
    '''
    if np.isscalar(t0):
        return alpha(t,t0,A,tau1,tau2)+bl
    else:    
        n=len(t0)
        Rsp=alpha(t,t0[0],A[0],tau1[0],tau2[0])
        for i in range(1,n):
            Rsp=Rsp+alpha(t,t0[i],A[i],tau1[i],tau2[i])
        return Rsp+bl

def varalpha(t,*param):
    '''Sum of variable number (n) alpha functions
       parameter can be input in the form of param list or 4*n+1 parameters
       used in the optimization algorithm . Efficiency may be improved. 
    '''
    L=len(param)
    if L==1:
        param=param[0]
        L=len(param)
    n=int(L//4)
    t0=param[0:-1:4];
    A=param[1:-1:4];
    tau1=param[2:-1:4];
    tau2=param[3:-1:4];
    bl=param[-1]
    Rsp=alpha(t,t0[0],A[0],tau1[0],tau2[0])
    for i in range(1,n):
        Rsp=Rsp+alpha(t,t0[i],A[i],tau1[i],tau2[i])
    return Rsp+bl

# def varf(t,*)


#plt.figure()
#plt.plot(times, multialpha(times,[10,15,20],[4,-5,2],[5,40,50],[10,10,15],0))
##print(popt)
##print(pcov)
#plt.plot(times, varalpha(times,(5,1,30,5,10,-3,100,5,0)))

'''Fitting Functions'''
def fitting(ydata):
    '''Basic single alpha function fitting'''
    global times
    t_start=time()
    popt, pcov= curve_fit(alphabl,times,ydata, bounds=([min(times)-50,-inf,0,0,min(ydata)], [max(times),inf,50,20,max(ydata)]))
    t_end=time()
    print("Fitting time : ", t_end-t_start,"s")
    plt.figure()
    plt.plot(times,ydata)
    yfit=alphabl(times,popt[0],popt[1],popt[2],popt[3],popt[4])
    plt.plot(times,yfit,ls='--')
    Err=np.linalg.norm(ydata-yfit)
    print("Fitting Error: ",Err)
    print("Fitting Param: Amp=",popt[0]," Amp=",popt[1]," tau_d=",popt[2]," tau_r",popt[3])
    return popt

def Multifitting(times,ydata,n=5):
    '''Fit multiple but fixed number functions'''
    tbeg=min(times)
    tend=max(times)
    Vmin=min(ydata)
    Vmax=max(ydata)
    #init=[(tbeg+tend)/2,0,50,5,(Vmin+Vmax)/2]
    #init=[[(tbeg+tend)/2]*2,[0]*2,[50]*2,[5]*2,[(Vmin+Vmax)/2]*2]
    init=[0,0,50,5]*n+[(Vmin+Vmax)/2]
    for i in range(n):
        init[i*4] = ((n-i)*tbeg + i*tend)/n
    
    t_start=time()
    #popt, pcov= curve_fit(alphabl5,times,ydata, bounds=([min(times)-50,-inf,0,0]*5 +[min(ydata)], [max(times),inf,100,15]*5 +[max(ydata)]))
    #popt, pcov= curve_fit(multialpha,times,ydata, p0=init, bounds=([tbeg-50,-inf,0,0] +[Vmin], [tend,inf,100,15] +[Vmax]))
    popt, pcov= curve_fit(varalpha,times,ydata, p0=init, bounds=([tbeg-50,-inf,0,0]*n +[Vmin], [tend,inf,100,15]*n +[Vmax]))
    t_end=time()
    print("Fitting time : ", t_end-t_start,"s")
    plt.figure()
    plt.plot(times,ydata,lw=1.5)
    # yfit=alphabl5(times, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11], popt[12], popt[13], popt[14], popt[15], popt[16], popt[17], popt[18], popt[19], popt[20])
    # yfit=multialpha(times, popt[0], popt[1], popt[2], popt[3], popt[4])
    yfit=varalpha(times,popt)
    plt.plot(times,yfit,ls='--',lw=2)
    Err=np.linalg.norm(ydata-yfit)
    ratio=1 - Err**2/np.linalg.norm(ydata-np.average(ydata))
    print("Fitting Error: ",Err)
    print("Variance account: ",ratio)
    #print("Fitting Param: Amp=",popt[0]," Amp=",popt[1]," tau_d=",popt[2]," tau_r",popt[3])
    return popt

'''Signal Processing and Peak Finding'''
from scipy import signal
def PeakFinding(times,ydata,plot=True):
    '''Peak Finding Algorithms using cwt in scipy routine
    plot control if it will draw the scatter peak points in a figure
    '''
    pkind_max=signal.find_peaks_cwt(ydata,np.arange(7,20))
    pkind_min=signal.find_peaks_cwt(-ydata,np.arange(7,20))
    if plot:
        plt.figure(figsize=[25,7])
        plt.plot(times,ydata,lw=1.5,alpha=0.7)
        plt.scatter(times[pkind_min],ydata[pkind_min],marker='o',color='black',s=48)
        plt.scatter(times[pkind_max],ydata[pkind_max],marker='o',color='red',s=48)
        plt.xlim([min(times),max(times)])
    return pkind_max,pkind_min

#pkind_max,pkind_min=PeakFinding(ydata)

def Multifitting_peak(times,ydata):
    '''Peak inspired fitting'''
    tbeg=min(times)
    tend=max(times)
    Vmin=min(ydata)
    Vmax=max(ydata)
    #init=[(tbeg+tend)/2,0,50,5,(Vmin+Vmax)/2]
    #init=[[(tbeg+tend)/2]*2,[0]*2,[50]*2,[5]*2,[(Vmin+Vmax)/2]*2]
    
    pkind_max,pkind_min=PeakFinding(times,ydata,plot=False)
    nmax=len(pkind_max)
    nmin=len(pkind_min)
    n=len(pkind_max+pkind_min)
    init=[0,0,50,5]*n+[(Vmin+Vmax)/2]
    for i in range(nmax):
        init[i*4] = times[pkind_max[i]]
        init[i*4 + 1]=-0.5
    for i in range(nmin):
        init[(i+nmax)*4] = times[pkind_min[i]]
        init[i*4 + 1 ]=0.5
    
    t_start=time()
    popt, pcov= curve_fit(varalpha,times,ydata, p0=init, bounds=([tbeg-50,-inf,0,0]*n +[Vmin], [tend,inf,100,15]*n +[Vmax]))
    t_end=time()
    print("Fitting time : ", t_end-t_start,"s")
    plt.figure()
    plt.plot(times,ydata,lw=1.5)
    yfit=varalpha(times,popt)
    plt.plot(times,yfit,ls='--',lw=2)
    Err=np.linalg.norm(ydata-yfit)
    ratio=1 - Err**2/np.linalg.norm(ydata-np.average(ydata))**2
    print("Fitting Error: ",Err)
    print("Variance account: ",ratio)
    #print("Fitting Param: Amp=",popt[0]," Amp=",popt[1]," tau_d=",popt[2]," tau_r",popt[3])
    return popt

def Multifitting_Seq(times,ydata,nstrt=10,ratio_crit=0.9):
    tbeg=min(times)
    tend=max(times)
    Vmin=min(ydata)
    Vmax=max(ydata)
    SStot=np.linalg.norm(ydata-np.average(ydata))**2
    n=nstrt
    init=[0,0,50,5]*n+[(Vmin+Vmax)/2]
    for i in range(n):
        init[i*4] = ((n-i)*tbeg + i*tend)/n
    t_start_all=time()
    while True:
        print("### alpha function No.: ", n," ###")
        t_start=time()
        popt,_= curve_fit(varalpha,times,ydata, p0=init, bounds=([tbeg-50,-inf,0,0]*n +[Vmin], [tend,inf,100,15]*n +[Vmax]))
        t_end=time()
        print("Fitting time : ", t_end-t_start,"s")
        yfit=varalpha(times,popt)
        res=ydata-yfit
        Err=np.linalg.norm(res)
        ratio=1 - Err**2/SStot
        print("Fitting Error: ",Err)
        print("Variance account: ",ratio)
        if ratio<ratio_crit:
            tmax=times[res.argmax()]
            tmin=times[res.argmin()]
            init=list(popt[0:-1])+[tmax,-0.5,50,5]+[tmin,0.5,50,5]+[popt[-1] ]
            n=n + 2
        else:
            break
    t_end_all=time()
    print("Fitting time Total: ", t_end_all-t_start_all,"s")
    plt.figure()
    plt.plot(times,ydata,lw=1.5)
    plt.plot(times,yfit,ls='--',lw=2)
    #print("Fitting Param: Amp=",popt[0]," Amp=",popt[1]," tau_d=",popt[2]," tau_r",popt[3])
    return popt

#fitting(ydata)
#

def IllustrFit(times,ydata,popt,save=False,Stat=False,method_name="",name_param=""):
    ''' take the time array original ydata and fitted parameter popt to generate figure. 
    ddate,recs,electrode,freqi,splj,trialk  are needed to title properly
    If this file is separated out to be an library we must supply with these '''
    if not name_param=="":
        ddate,recs,electrode,freqi,splj,trialk=name_param
    else:
        ddate,recs,electrode,freqi,splj,trialk="",-1,-1,-1,-1,-1
    n=int(len(popt)//4)
    
    plt.figure(figsize=[25,7])
    ax1=plt.subplot(211)
    plt.plot(times,ydata,lw=1.5)
    yfit=varalpha(times,popt)
    plt.plot(times,yfit,ls='--',lw=2)
    plt.title(ddate+"-rec%d -e%d" % (recs,electrode)+"-freq:%d=%dHz - spl:%d=%ddB - trial:%d" % (freqi,freqlab[freqi],splj,spllab[splj],trialk)+"  funcNo:%d method:%s" % (n,method_name) )
    plt.xticks([])
    ax2=plt.subplot(212)
    for i in range(n):
        plt.plot(times,alpha(times, popt[4*i +0], popt[4*i +1], popt[4*i +2], popt[4*i +3]),ls='-',color='blue',alpha=0.8)
    plt.plot(times,ydata-popt[-1],color="black")
    if save: 
        plt.savefig(ddate+"_%d_e%d_%d_%d_%d_%dalpha%s.png" % (recs,electrode,freqi,splj,trialk,n,method_name))
        plt.close()
    
    # Histogram of Params 
    if Stat:
        plt.figure(figsize=[18,7])
        plt.subplot(131)
        plt.hist(popt[1:-1:4],bins=20)
        plt.xlabel("A")
        As=popt[1:-1:4];tau1=popt[2:-1:4];tau2=popt[3:-1:4];
        IPind=np.nonzero(As<0)[0]
        EPind=np.nonzero(As>0)[0]
        plt.subplot(132)
       # plt.hist(popt[2:-1:4])
        plt.hist(tau1[EPind],alpha=0.7,label="EPSP")
        plt.hist(tau1[IPind],alpha=0.7,label="IPSP")
        plt.legend()
        plt.xlabel("$\tau_1$")
        plt.subplot(133)
        #plt.hist(popt[3:-1:4])
        plt.hist(tau2[EPind],alpha=0.7,label="EPSP")
        plt.hist(tau2[IPind],alpha=0.7,label="IPSP")
        plt.legend()
        plt.xlabel("$\tau_2$")
        plt.suptitle("Param Histogram"+ddate+"-rec%d -e%d" % (recs,electrode)+"-freq:%d=%dHz - spl:%d=%ddB - trial:%d" % (freqi,freqlab[freqi],splj,spllab[splj],trialk)+"  funcNo:%d method:%s" % (n,method_name))
        if save: 
            plt.savefig(ddate+"_%d_e%d_%d_%d_%d_%dalpha%sStat.png" % (recs,electrode,freqi,splj,trialk,n,method_name))
            plt.close()

'''Script Zone'''
import pickle
ddate="140930";recs=16;electrode=1;
tonestsr,freqlab,spllab,times=ReReadin_Data(ddate=ddate,recs=recs,electrode=electrode,platform="laptop")
for splj in range(5):
    for freqi in range(15):
        for trialk in range(5):
            ydata=tonestsr[freqi,splj,trialk,:]
            popt=Multifitting_peak(times,ydata)
            IllustrFit(times,ydata,popt,Stat=True,method_name="PeakInsp",save=True,name_param=(ddate,recs,electrode,freqi,splj,trialk))
            pickle.dump( popt, open( 'popt_'+ddate+'_rec'+str(recs)+'_e'+str(electrode)+'_%d_%d_%d.p' % (freqi,splj,trialk), "wb" ) )

splj=0;freqi=1;trialk=4;
popt=pickle.load(open( 'popt_'+ddate+'_rec'+str(recs)+'_e'+str(electrode)+'_%d_%d_%d.p' % (freqi,splj,trialk), "rb" ))
''''''
#import scipy.interpolate as interp

As=popt[1:-1:4];tau1=popt[2:-1:4];tau2=popt[3:-1:4];
IPind=np.nonzero(As<0)[0]
EPind=np.nonzero(As>0)[0]
plt.figure()
plt.subplot(132)
plt.hist(tau1[EPind],alpha=0.7,label="EPSP")
plt.hist(tau1[IPind],alpha=0.7,label="IPSP")
plt.legend()
plt.subplot(133)
plt.hist(tau2[EPind],alpha=0.7,label="EPSP")
plt.hist(tau2[IPind],alpha=0.7,label="IPSP")
plt.legend()
