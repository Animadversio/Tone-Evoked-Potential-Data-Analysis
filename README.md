Tone Evoked Potential Processing Projects
================
# PreNotes: Setting and Data structure
## Recording Setting 
1 or 2 electrode recording 

## File Sturctue
suffix:  
pref_[Exp data]_[Beg record-End record]_e[electrode No. 1/2] 
e.g.: whole_cell_tones_140930_21-21_e1

whole_cell_tones_[suffix]

e.g. 5 files per record. 
whole_cell_tones_140930_1-1_e1.pdf
tone_evoked_amplitudes_140930_1-1_e1.pdf

tone_evoked_amplitudes_140930_rec1-1_e1.p
tone_data_whole_cell_tones_140930_rec1-1_e1.p
data_whole_cell_tones_140930_rec1-1_e1.p

## Data Sturcture

**whole_cell_tones_[]_.pdf**
- Voltage-Time course plot
- xlabel are frequency of 15 tones 
- xaxis  is  
    + times = arange(-tbefore,tduration+tafter+dt,dt) 
    + tbefore = 100.# consider time before tone
    + tafter = 200. # consider time after tone
    + dt = 1000./rate = 1000./10000
    + tduration specified in experiment. i.e. 50 ;  
- ylabel is SPL and measured voltage 
- dashed vline of onset and offset (0, tduration)
- ylim(minimum,maximum) are set to fit the (max, min) of all trial all tone. [it's a global minimum/maximum]

**tone_evoked_amplitudes_[]_.pdf**
- Amplitude-Tone Frequency (log x axis)
- blue , green , gray dash

**tone_evoked_amplitudes_[]_.p**
- pickle.dump( amplitude_data, open( 'tone_evoked_amplitudes_'+ddate+'_rec'+str(recs[0])+'-'+str(recs[-1])+'_e'+str(electrode)+'.p', "w" ) )
- amplitude_data : dict len=2 
    + 0: dict
        * b'max0', : 
        * b'min0', : 
        * b'freq', : 15 tones frequency Hz
        * b'spl', : 
        * b'vmedian0' :
    + 1: dict
        * b'BFexc0' : 
        * b'BFinh0' : 

**data_whole_cell_tones_[]_.p**
- pickle.dump( tones, open( 'data_whole_cell_tones_'+str(ddate)+'_rec'+str(int(recs[0]))+'-'+str(int(recs[-1]))+'_e'+str(electrode)+'.p', "w" ) )
- tones : list, len=15
    + i : dict, len=nrep+1, keys=b'data0_0',...b'data0_avg'
        * b'data0_i' : 
        * b'data0_avg': len=3501
        * (b'data1_i' means paired recording)

**tone_data_whole_cell_tones_140930_rec1-1_e1.p**
- pickle.dump( tonedata, open( 'tone_data_whole_cell_tones_'+str(ddate)+'_rec'+str(int(recs[0]))+'-'+str(int(recs[-1]))+'_e'+str(electrode)+'.p', "w" ) )
- tonedata: dict, len=15, keys=range(15)
    + list, len=4
        * [order(0,...14) ,freq ,SPL , ??? ]
        * 1st num is the permutation num of tones, denotes which data trace in tones it refers to 
	        * i.e.: ax.plot(times,tones[int(tonedata[j][0])][bytes('data0_'+str(i),'ascii')],lw=1,color=c0)
    + info of 15 tones

(if paired there might be correlation_plots)

**parameter meaning**
- tones,tonedata,
- recs,
- reps, repetitions
- spls, Sound Pressure Level: num of SPL usually 1 
    + $ L_{p}=\ln \!\left({\frac  {p}{p_{0}}}\right)\!~{\mathrm  {Np}}=2\log _{{10}}\!\left({\frac  {p}{p_{0}}}\right)\!~{\mathrm  {B}}=20\log _{{10}}\!\left({\frac  {p}{p_{0}}}\right)\!~{\mathrm  {dB}} $
    + 
- freq, : num of tones, 15 here
- times = arange(-tbefore,tduration+tafter+dt,dt) 

## Data Structure
15 tones 

## Questions about Data

* Baseline? 
    - Holding Potential (see excel)
    - Resting Potential
* What qualify as a spike? 
    - 
* data15-15 
    - synchronizing of inhibitory inputs?


@ 7.12 meeting with Alex D. Reyes: Data Understanding
-------------------------

# What we can do with data? 
* Εxtract spike?
    - Threshold
    - (Quite Obvious on background Data)
* Spike timing statistics? \
    - record SP timing in single trial 
    - plot spike timing
* Spike number/probability statistics. 
    - 
* Relation among spike timings? 
    - Basic: Correlation 
    - Other Time Sequence statistical methods. 
* LFP information 
    - Exc Inh input info: timing / num 
    - EPSP / IPSP qualification 
        + Large slope up
        + Small slope down ? 

# Physiology Constraint
* Single IPSP Single EPSP (Another Exp setting in vitro)
    - pos/neg nearly symmetry 
    - 500 $\mu V$ Height
    - Time course, $\tau_{up}$, $\tau_{down}$
* Main Recepter type
	* Exc: AMPA, sometimes NMDA
	* Inh: GABA
* Summation / interaction of EPSP,IPSP 
	* Nonlinearity exist : roof effect ($V<E_{exc}$) 
	* Channel conductane may sum linearly, due to the tree structure of dendrites 

# Questions we can ask 
* Stat of EPSP and IPSP 
    - High Dim Nonlinear Fitting : multiple alpha funcs' summation
    - alpha function : alpha(t)=@(t) A(exp(-(t-t0)/tau1)-exp(-(t-t0)/tau2))
    - **CRUCIAL problem**: distinguish EPSP & IPSP 
        + 1st approximation, PSP has sharper begining edge, slower decaying edge. --critrion on slope
        + When superposed, 
* Correlations among EPSP and IPSP in 1@0 2@-70 settings (Current Clamping)
	* 1@0 - record only IPSP
	* 2@-70 record only EPSP
* Other kind of Correlation of Paired Trial? 
* Rate of Failure / Success among Trials 
	* Reveal the all/none respond structure in respond. 

@ 7.14 meeting with Alex D. Reyes: Preliminary Fitting Results
--------------------------------------
**Targets next step**
* Algorithm Validation
	* With Simulated Trace
		* Input spike trains to the data. Use alpha function with different parameters as the synapse conductance function. Simulate the conductance 
	* With *in vivo* Data? 
		* Simultaneously get EPSP signal, IPSP signal, and composite signal . Then, use algorithms on composite signal. see if the EPSP / IPSP 
		* Indirectly, compare the fitting distribution of parameters and biological parameters.
* Algorithm Usage
	* Choose suitable set of data to fit! @ rest @ 0 @ -80 , paired or single. 
	* If validated, it can be use to decompose rest state electric trace ! ! And get EPSP and IPSPs
	* do Correlation to validate the Excitatory and Inhibitory balance and synchronize hypothesis
* Inspirations
	* **Gamma Rhythm**  may be the basis of large fluctuation in potential. Info can be extracted out about it, when we separately inspect large A alpha functions, and  smaller ones . 
	* Balance and Simultaneous Synchrony  of Excitatory and Inhibitory inputs

**Questions Remained about data**
* Excel of exp setting for all the record on 140930/141008 ? 
	* **2p_wc_wc_overview_analysisNov2014.ods** is incomplete
* original data? numbering ~ exp no ~ pdf file no.  What's the corresponence. 

