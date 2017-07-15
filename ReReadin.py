# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 18:22:38 2017

@author: Compute

to see the pdf results 

os.system("sumatrapdf \"D:\\NYUReyesLab\\Tone Evoked Response\\140930\\whole_cell_tones_140930_16-16_e1.pdf\"")
sumatrapdf "140930\\whole_cell_tones_140930_16-16_e1.pdf"
"""



'''
pdf generated



pickle dumped
pickle.dump( tones, open( 'data_whole_cell_tones_'+str(ddate)+'_rec'+str(int(recs[0]))+'-'+str(int(recs[-1]))+'_e'+str(electrode)+'.p', "w" ) )
# save tone identity data
pickle.dump( tonedata, open( 'tone_data_whole_cell_tones_'+str(ddate)+'_rec'+str(int(recs[0]))+'-'+str(int(recs[-1]))+'_e'+str(electrode)+'.p', "w" ) )
# save tone-evoked amplitudes
pickle.dump( amplitude_data, open( 'tone_evoked_amplitudes_'+ddate+'_rec'+str(recs[0])+'-'+str(recs[-1])+'_e'+str(electrode)+'.p', "w" ) )

'''


# spls

'''
tones write by   generate_tone_plots(tones,tonedata,recs,reps,spls,freq,datachannel,ddate)
## read_in_data(tones,rec,cyc,datachan,triggerchan)
if old_format:
    tones[j]['data0_'+str(int(r))] = yalt0[(tonestart[ndata_old]-tbefore/dt):(tonestart[ndata_old]+tduration/dt + tafter/dt+1)]
    if paired:
        tones[j]['data1_'+str(int(r))] = yalt1[(tonestart[ndata_old]-tbefore/dt):(tonestart[ndata_old]+tduration/dt + tafter/dt+1)]
else:
    tones[j]['data0_'+str(int(i))] = yalt0[(tonestart[ndata]-tbefore/dt):(tonestart[ndata]+tduration/dt + tafter/dt+1)]
    if paired:
        tones[j]['data1_'+str(int(i))] = yalt1[(tonestart[ndata]-tbefore/dt):(tonestart[ndata]+tduration/dt + tafter/dt+1)]

## generate_tone_plots
avg0 = 0.
avg1 = 0.
for i in arange(int(reps)):
    c0 = cm.summer(float(i)/float(reps),1)
    # int(tonedata[j][0]) picks out the data for the first, second plot from the right location
    # in the tones array
    ax.plot(times,tones[int(tonedata[j][0])]['data0_'+str(i)],lw=1,color=c0)
    avg0 += tones[int(tonedata[j][0])]['data0_'+str(i)]/float(reps)
    if paired:
        c1 = cm.copper(float(i)/float(reps),1)
        ax.plot(times,tones[int(tonedata[j][0])]['data1_'+str(i)],lw=1,color=c1)
        avg1 += tones[int(tonedata[j][0])]['data1_'+str(i)]/float(reps)


tones[int(tonedata[j][0])]['data0_avg'] = avg0
tones[int(tonedata[j][0])]['data1_avg'] = avg1


'''

'''
tonedata write by

tonedata = read_in_data(tones,recs,cycle,datachannel,triggerchannel)
# first sort along the frequency axistduration = header['toneDuration'][0]
tonedata.sort(key=lambda x: x[1])
# then sort along the spl axis
tonedata.sort(key=lambda x: x[2])


'''


'''
amplitude_data write by generate_tone_amplitude_plots(tones,tonedata,tonesamplitudes,recs,reps,spls,freq,datachannel,ddate):


amplitude_data[0].keys()

amplitude_data[1].keys()

'''

'''
cycle  equiv to reps
'''
import pickle
import numpy as np
import matplotlib.pyplot as plt

def ReReadin_Data(ddate,recs,electrode,freqNo=None,SPLNo=None,TrialNo=None,platform="laptop"):
    '''reRead in data and store in multidimension arrays in np, 
    enable slicing
    '''
    paired= True if electrode==12 else False
    if platform=="lyoto":
        dir_processed='/localdisk/Local/Users/binxu/Tone Evoked Potential proc/'
    else:
        dir_processed='D:/NYUReyesLab/Tone Evoked Response/'
    tonedata=pickle.load(open(dir_processed+ddate+'/tone_data_whole_cell_tones_'+ddate+'_rec'+str(int(recs))+'-'+str(int(recs))+'_e'+str(electrode)+'.p','rb'),encoding='bytes')
    tones=pickle.load(open(dir_processed+ddate+'/data_whole_cell_tones_'+ddate+'_rec'+str(int(recs))+'-'+str(int(recs))+'_e'+str(electrode)+'.p','rb'),encoding='bytes')
    #amplitude_data=pickle.load(open(dir_processed+ddate+'\\tone_evoked_amplitudes_'+ddate+'_rec'+str(int(recs))+'-'+str(int(recs))+'_e'+str(electrode)+'.p','rb'),encoding='bytes')
    print("Data File name: ",'data_whole_cell_tones_'+ddate+'_rec'+str(int(recs))+'-'+str(int(recs))+'_e'+str(electrode)+'.p')
    
    tonedata=np.array(tonedata)
    if platform=="lyoto":
        dir_orig = "/localdisk/Local/Users/binxu/Tone Evoked Potential proc/2p_Orig_datafile/" 
    else:
        dir_orig = "D:/NYUReyesLab/Tone Evoked Response/2p_Orig_datafile/" 
    header=pickle.load(open(dir_orig+ ddate+'/A%03dh.p' % recs,'rb'),encoding='bytes')
    #data=pickle.load(open(dir_orig+'/A%03dd.p' % recs,'rb'),encoding='bytes')
    
    SPLmax = header[b'SPLmax'][0]
    SPLmin = header[b'SPLmin'][0]
    SPLsteps = header[b'SPLsteps'][0]
    spls=1 if not SPLsteps else int( (SPLmax-SPLmin)/SPLsteps ) +1
    reps = int(header[b'repetitions'][0])
    freq = header[b'number of tones']
    ntones = freq*spls
    rate = 10000.
    dt = 1000./rate
    tbefore = 100.
    tafter = 200.
    tduration = header[b'toneDuration'][0]
    times = np.arange(-tbefore,tduration+tafter+dt,dt)
    ticks=len(times)
    
    print("Tensor Structure",freq,"frequencies ", spls, " Sound Pressure Levels ", reps, " Trials \n",ticks, " time slices")
    trace=np.zeros((freq,spls,reps,ticks))
    if paired:
        trace1=np.zeros((freq,spls,reps,ticks))
    for j in range(ntones) :
        freqi=int(j%freq)
        SPLj=int(j//freq)
        toneNo=int(tonedata[j,0]) # ind in tones data set
        for k in range(reps):
            trace[freqi,SPLj,k,:]=tones[toneNo][bytes('data0_'+str(k),'ascii')]
            if paired:
                trace1[freqi,SPLj,k,:]=tones[toneNo][bytes('data1_'+str(k),'ascii')]
    freqlab=tonedata[0:freq,1]
    SPLlab=tonedata[0:ntones:freq,2]
    if not paired:
        return trace,freqlab,SPLlab,times
    else:
        return trace,trace1,freqlab,SPLlab,times

'''Replot and analysis'''
#ddate='140930'
#recs=[16]
#electrode=1
#tonedata=pickle.load(open('D:\\NYUReyesLab\\Tone Evoked Response\\140930\\tone_data_whole_cell_tones_'+str(ddate)+'_rec'+str(int(recs[0]))+'-'+str(int(recs[-1]))+'_e'+str(electrode)+'.p','rb'),encoding='bytes')
#tones=pickle.load(open('D:\\NYUReyesLab\\Tone Evoked Response\\140930\\data_whole_cell_tones_'+str(ddate)+'_rec'+str(int(recs[0]))+'-'+str(int(recs[-1]))+'_e'+str(electrode)+'.p','rb'),encoding='bytes')
#amplitude_data=pickle.load(open('D:\\NYUReyesLab\\Tone Evoked Response\\140930\\tone_evoked_amplitudes_'+str(ddate)+'_rec'+str(int(recs[0]))+'-'+str(int(recs[-1]))+'_e'+str(electrode)+'.p','rb'),encoding='bytes')
##tonedata[1][b'data0_9']
#
#'''Header info reading'''
#rec=recs[0]
#dir_location = "D:/NYUReyesLab/Tone Evoked Response/2p_Orig_datafile/" + str(ddate)
#header=pickle.load(open(dir_location+'/A%03dh.p' % rec,'rb'),encoding='bytes')
#data=pickle.load(open(dir_location+'/A%03dd.p' % rec,'rb'),encoding='bytes')
##del data
### header sturcture
##hnamelst=list(header.keys())
##hnamelst=[hnamelst[i].decode() for i in range(len(hnamelst))]
##hnamelst.sort()
#
#if electrode == 1:
#    datachannel = (['di0P'])
#    paired = False
#elif electrode == 2:
#    datachannel = (['di2P'])
#    paired = False
#elif electrode == 12:
#    datachannel = (['di0P','di2P'])
#    paired = True
## trigger channel
#triggerchannel = 'di4P'
#tduration = header[b'toneDuration'][0]
#intertone = header[b'intervalBetweenTones'][0]
#SPLmax = header[b'SPLmax'][0]
#SPLmin = header[b'SPLmin'][0]
#SPLsteps = header[b'SPLsteps'][0]
#spls=1 if not SPLsteps else int( (SPLmax-SPLmin)/SPLsteps ) +1
#repetitions = int(header[b'repetitions'][0])
#reps=repetitions
#freq = header[b'number of tones']
#ntones = freq*spls
#print(spls, " Sound Pressure Levels ",freq,"frequencies ", reps, " Trials")
#
#rate = 10000.
#dt = 1000./rate
#tbefore = 100.
#tafter = 200.
#tduration=50.
#times = arange(-tbefore,tduration+tafter+dt,dt)
#
#maximum = -100.
#minimum = +100.
#for j in arange(ntones):
#    for i in arange(int(reps)):
#        if (max(tones[j][bytes('data0_'+str(i),'ascii')]) > maximum ):
#            maximum = max(tones[j][bytes('data0_'+str(i),'ascii')])
#        if (min(tones[j][bytes('data0_'+str(i),'ascii')]) < minimum ):
#            minimum = min(tones[j][bytes('data0_'+str(i),'ascii')])
#plt.figure()
#for j in range(15):
#    ax=plt.subplot(1,15,j+1)
#    for i in range(reps):
#        clrtmp=plt.cm.summer(i/repetitions, alpha=0.8)
#        plot(times,tones[j][bytes('data0_'+str(i),'ascii')],color=clrtmp)
#    plot(times,tones[j][b'data0_avg'], color='blue')
#    axvline(x=0,ls='--',color='0.6')
#    axvline(x=tduration,ls='--',color='0.6')
#    ax.spines['top'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#    plt.xticks(visible=False)
#    plt.yticks([])
##    ax.spines['left'].set_visible(False)
##    ax.spines['bottom'].set_visible(False)
#        plt.ylim([minimum, maximum])
#
#j=4
#plt.figure()
#for i in range(reps):
#    clrtmp=plt.cm.summer(i/repetitions, alpha=0.8)
#    plot(times,tones[j][bytes('data0_'+str(i),'ascii')],color=clrtmp)
#plot(times,tones[j][b'data0_avg'], color='blue')
#axvline(x=0,ls='--',color='0.6')
#axvline(x=tduration,ls='--',color='0.6')
#plt.xlabel("%dHz" % tonedata[j][1])

#generate_tone_plots(tones,tonedata,recs,reps,spls,freq,datachannel,ddate)

