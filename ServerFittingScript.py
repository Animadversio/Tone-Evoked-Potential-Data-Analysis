# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 23:35:14 2017

@author: Compute
"""
import pickle
from ReReadin import ReReadin_Data
from PSPFitting import IllustrFit,Multifitting,Multifitting_peak,Multifitting_Seq
ddate="140930";recs=16;electrode=1;
tonestsr,freqlab,spllab,times=ReReadin_Data(ddate=ddate,recs=recs,electrode=electrode,platform="laptop")
#global times
#freqi=0; splj=0; trialk=4;
#ydata=tonestsr[freqi,splj,trialk,:]
##fitting(ydata)
#
#splinen=20
#popt=Multifitting(ydata,splinen)
#IllustrFit(ydata,popt,save=True)
#
#
#popt=Multifitting_peak(ydata)
#IllustrFit(ydata,popt,Stat=True,method_name="PeakInsp")
#
#
#freqi=0; splj=1; trialk=2;
#ydata=tonestsr[freqi,splj,trialk,:]
#popt2=Multifitting_Seq(ydata,nstrt=10,ratio_crit=0.92)
#IllustrFit(ydata,popt2,Stat=True,method_name="SeqFit")

#splj=0;
for splj in range(5):
    for freqi in range(15):
        for trialk in range(5):
            ydata=tonestsr[freqi,splj,trialk,:]
            popt=Multifitting_peak(times,ydata)
            IllustrFit(times,ydata,popt,Stat=True,method_name="PeakInsp",save=True)
            pickle.dump( popt, open( 'popt_'+ddate+'_rec'+str(recs)+'_e'+str(electrode)+'_%d_%d_%d.p' % (freqi,splj,trialk), "wb" ) )
