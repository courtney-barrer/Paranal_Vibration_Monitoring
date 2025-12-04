#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:49:26 2023

@author: bcourtne

changes (aim to move away from online tool to offline. i.e. no automatic state detection )
============
4-12-25 : remove dlt import, to remove dependence, in get_state() need to just make dummy state
we can assume guiding, enc open - change so user can input focus! 



"""
import numpy as np
#import h5py
import os
import matplotlib.pyplot as plt 
import scipy.signal as sig
import glob
import datetime
import scipy.signal as sig
import pandas as pd 

try:
    import dlt # needs to be in virtual environment for import to work (source venv/bin/activate)
except:
    print("cannot import dlt, this is ok! We are probably just not in ESO/datalab virtual environment so use this as an offline tool!")


#mapping from sensor indices to mirror position 
i2m_dict = {1:'M3a',2:'M3b',3:'M2',4:'empty',5:'M1+y',6:'M1-x',7:'M1-y',8:'M1+x',9:'M4',10:'M5',11:'M6',12:'M7'} 
m2i_dict = {v: k for k, v in i2m_dict.items()}

# gravity baseline to telescope mapping 
baselabels = ['43','42','41','32','31','21']
base2telname = [[4, 3], [4, 2], [4, 1], [3, 2], [3, 1], [2, 1]]
tel2telname = [4, 3, 2, 1]
base2tel = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
tel2base = [[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]]
M_matrix = np.array([[1.0, -1.0, 0.0, 0.0],
                     [1.0, 0.0, -1.0, 0.0],
                     [1.0, 0.0, 0.0, -1.0],
                     [0.0, 1.0, -1.0, 0.0],
                     [0.0, 1.0, 0.0, -1.0],
                     [0.0, 0.0, 1.0, -1.0]])

V2acc_gain = 0.01  # from MNII amplifiers (should be the same across all amplifiers (otherwise results reported here will be wrong!)
nperseg = 2**11 #for PSDs

def double_integrate(f, acc_psd):
    #double integrate PSD in freq domain 
    
    psd_pos = 1/(2*np.pi)**4 * acc_psd[1:] / f[1:]**4
    
    return( f[1:], psd_pos )

def basic_filter(data, outlier_thresh, replace_value  ) :
    # basic filter to replace any absolute values in data > outlier_thresh with replace_value 
    
    if not (outlier_thresh is None):
                
        #sensor_tmp = mn2_data[time_key][f'sensor{s}'][:].copy()
        # get indicies where abs sensor values are above user 'outlier threshold'
        outlier_indx = abs( data ) > outlier_thresh 

        # replace outliers with users 'replace_value' 
        data[outlier_indx] = replace_value

        # how many outliers are replaced? 
        no_replaced = [np.sum(outlier_indx)] 
                
    else:
        no_replaced  = [0]     
        
    return(data, no_replaced) 

def is_between(time, df):
    if sum( (time <= df['final_time']) & (time >= df['initial_time'] ) ): # if time is between at least 1 initial and final time columns
        return(1)
    else:
        return(0)

def get_state(input_date, ut):

    # define tcs environment that dlt package will query
    env = f'wt{ut}tcs'
    
    # we look +- 1 day around input time to ensure we get queried values with final / inital dates that enclose input date
    initial_input_date , final_input_date = input_date - datetime.timedelta(days=1), input_date + datetime.timedelta(days=1)
    
    #periods of telescope guiding                                                                                                                              
    guiding_periods = dlt.query_guide_times(env, initial_input_date, final_input_date)
    
    #periods of enclosure open                                                                                                                                 
    open_enc_periods = dlt.query_enclosure_times(env, initial_input_date, final_input_date)
           
    # focus periods                                                                                                                                            
    focus_periods = dlt.query_focus_times(env, initial_input_date, final_input_date, names=False)
    
    #print(focus_periods)                                                                                                                                      
    
    focus_periods['focus_name'] = [dlt.get_focus_name(focus_periods['focus'][i]) for i in range(len(focus_periods))]
    
    focus_mask = (input_date <= focus_periods['final_time']) & (input_date >= focus_periods['initial_time'] )
    if sum(focus_mask): # if non empty (i.e. input_date is between a focus value in focus_periods)                                                        
        current_focus = focus_periods['focus_name'][focus_mask].values[0]
    else: #input_date is not between any initial or final input_date in focus periods                                             
        current_focus = np.nan
    
    # put these as boolean values in dictionary that match keys of the historic 'states' dataframe                                                             
    tel_state = {}
    tel_state['open_enclosure'] = bool( is_between(input_date, open_enc_periods) )
    tel_state['guiding'] =  bool( is_between(input_date, guiding_periods) )
    
    for foc_tmp in ['Coude', 'Nasmyth A', 'Nasmyth B', 'Cassegrain']:
        tel_state[foc_tmp] = current_focus==foc_tmp
    
    return(tel_state)



def get_psd_reference_file(reference_path, ut, sensor_name, tel_state_dict):
    # get the reference psd file path + name corresponding to ut, sensor and telescope state. Follows naming convention for reference csv files 
    no_file_flag = 0 # flag to indicate if no file is found (=0 if found files) 

    if tel_state_dict[ut]['open_enclosure'] & tel_state_dict[ut]['guiding']:
        state_name = 'operations'
    else:
        state_name = 'non-operations'

    if tel_state_dict[ut]['Coude']:
        focus_name = 'coude'
    elif tel_state_dict[ut]['Nasmyth A']:
        focus_name = 'nasA'
    elif tel_state_dict[ut]['Nasmyth B']:
        focus_name = 'nasB'
    elif tel_state_dict[ut]['Cassegrain']:
        focus_name = 'cas'
    else:
        print(' no defined focus. We are going to classify on reference PSDs without filtering the focus' )
        focus_name = 'allFoci'


    # Get the reference file that corresponds to the input file state / sensor to compare PSD statistics                                                                        
    psd_reference_file = glob.glob(reference_path + f'UT{ut}/*UT{ut}*_{state_name}_*{focus_name}*{sensor_name}_psd_distribution_features.csv')

    if len(psd_reference_file)==0:
        print(f'\n--------\nNo psd reference file found in {reference_path} for {ut} for {state_name}, {focus_name}, {sensor_name}.\nYou should put one there or make sure naming convention is correct!\n--------\n\n')
        no_file_flag = 1
    elif len(psd_reference_file)>1:

        # pick the most recent one                                                                                                                                              
        psd_reference_file = [ np.sort( psd_reference_file )[-1] ]

        print(f'\n-------\nThere were multiple reference files in {reference_path} for {ut} for {state_name} state, focus={focus_name}, sensor={sensor_name}.\n We will pick themost recent which is {psd_reference_file}')

    print(f'\ncomparing to input MNII data to:\n   {psd_reference_file}')

    return( psd_reference_file, no_file_flag  )



def process_single_mn2_sample(mn2_data, time_key, post_upgrade=True, user_defined_geometry=None, outlier_thresh = None, replace_value=0, ensure_1ms_sampling=False):
    """
    mn2_data = raw data read in from datalabs (e.g. h5py.File(file, 'r')) file name - string (e.g. 'ldlvib{UT}_raw_{yyyy}-{mm}-{dd}.hdf5') 
    ... note needs to be raw file and not psd 
    time_key - string (e.g 'hh:mm:ss' ) 
    post_upgrade - boolean: do we consider new accelerometers 
    user_defined_geometry - list with strings for user defined geometry for mirror piston combination
        e.g. to combine m1, m2, m3 and m5 the user input user_defined_geometry = ['m1', 'm2', 'm3' and 'm5']
    
    outlier_thresh = float or None , if not None then any absolute raw sensor value above outlier_thresh will be 
    replaced with replace_value (default is np.nan in case that replace_value = None
    
    replace_value = float, int etc 
    
    ensure_1ms_sampling = boolean
        if True we manually check that each sample contains exactly 10k samples (1ms sampling), if not then we set the data to np.nan. This is importanat for merging multiple samples
    """
    
    #mn2_data = h5py.File(file, 'r')
    acc = {}
    
    if not post_upgrade: #only consider sensors installed up to m3 
        
        for _,s in enumerate([1,2,3,5,6,7,8]):
            
            # start to fill acc dictionary , 
            # these sensor values should NOT be scaled or manipulated!!! since later functions depend on raw value
            acc[f'sensor_{s}'] = mn2_data[time_key][f'sensor{s}'][:]
            
            #acc[f'sensor_{s}'], acc[f'sensor_{s}_no_outliers_replaced'] = basic_filter(data, outlier_thresh, replace_value  ) 
            
            """
            if not (outlier_thresh is None):
                
                #sensor_tmp = mn2_data[time_key][f'sensor{s}'][:].copy()
                # get indicies where abs sensor values are above user 'outlier threshold'
                outlier_indx = abs( acc[f'sensor_{s}'] ) > outlier_thresh 
                
                # replace outliers with users 'replace_value' 
                acc[f'sensor_{s}'][outlier_indx] = replace_value
                    
                # how many outliers are replaced? 
                acc[f'sensor_{s}_no_outliers_replaced'] = [np.sum(outlier_indx)] 
                
            else:
                acc[f'sensor_{s}_no_outliers_replaced'] = [0]     

            """
                
        acc['m3'] = -V2acc_gain *(mn2_data[time_key]['sensor1'][:] + mn2_data[time_key]['sensor2'][:] )/2.0
        acc['m2'] = -V2acc_gain *mn2_data[time_key]['sensor3'][:]
        
        acc['m1'] = +V2acc_gain *(mn2_data[time_key]['sensor5'][:]  + mn2_data[time_key]['sensor6'][:] \
                        + mn2_data[time_key]['sensor7'][:]  + mn2_data[time_key]['sensor8'][:] )/4.0


        # Convert to piston, based on geometry
        acc['m1'] *= 2.0
        acc['m2'] *= 2.0
        acc['m3'] *= np.sqrt(2.0)

        #combined geometry up to m3
        acc['m1-3'] = acc['m1'] + acc['m2'] + acc['m3']

        tmp_keys = list(acc.keys()) # so we don't iterate on dict keys while changing its keys
        for key in tmp_keys:
            acc[key], acc[f'{key}_no_outliers_replaced'] = basic_filter(acc[key], outlier_thresh, replace_value  ) 
            acc[key]=list(acc[key])

        if ensure_1ms_sampling:
            for key in tmp_keys:
                if len( acc[key] ) != 10000:
                    acc[key] = list(np.nan * np.empty(10000))
                    
                    
        return(acc) 
    
    elif post_upgrade: # consider sensors installed up to m7
        
        for _,s in enumerate([1,2,3,5,6,7,8,9,10,11,12]):
            
            # start to fill acc dictionary , 
            # these sensor values should NOT be scaled or manipulated!!! since later functions depend on raw value
            acc[f'sensor_{s}'] = mn2_data[time_key][f'sensor{s}'][:]
            
            """
            if not (outlier_thresh is None):
                
                #sensor_tmp = mn2_data[time_key][f'sensor{s}'][:].copy()
                # get indicies where abs sensor values are above user 'outlier threshold'
                outlier_indx = abs( acc[f'sensor_{s}'] ) > outlier_thresh 
                
                # replace outliers with users 'replace_value' 
                acc[f'sensor_{s}'][outlier_indx] = replace_value
                    
                # how many outliers are replaced? 
                acc[f'sensor_{s}_no_outliers_replaced'] = [np.sum(outlier_indx)]
                
            else:
                acc[f'sensor_{s}_no_outliers_replaced'] = [0]     
            """

        acc['m3'] = -V2acc_gain * (mn2_data[time_key]['sensor1'][:] + mn2_data[time_key]['sensor2'][:] )/2.0
        acc['m2'] = -V2acc_gain * mn2_data[time_key]['sensor3'][:]
        
        acc['m1'] = V2acc_gain * (mn2_data[time_key]['sensor5'][:]  + mn2_data[time_key]['sensor6'][:] \
                        + mn2_data[time_key]['sensor7'][:]  + mn2_data[time_key]['sensor8'][:] )/4.0


        acc['m4'] = -V2acc_gain *mn2_data[time_key]['sensor9'][:] 
        acc['m5'] = -V2acc_gain *mn2_data[time_key]['sensor10'][:]
        acc['m6'] = -V2acc_gain *mn2_data[time_key]['sensor11'][:] 
        acc['m7'] = -V2acc_gain *mn2_data[time_key]['sensor12'][:]

        # Convert to piston, based on geometry
        acc['m1'] *= 2.0
        acc['m2'] *= 2.0
        acc['m3'] *= np.sqrt(2.0)
        acc['m4'] *= np.sqrt(2.0)
        acc['m5'] *= 1.9941 # valued checked with Salman on Aug 13 2022
        acc['m6'] *= 1.8083 # valued checked with Salman on Aug 13 2022
        acc['m7'] *= 1.9822 # valued checked with Salman on Aug 13 2022

        #combined geometry 
        acc['m1-3'] = acc['m1'] + acc['m2'] + acc['m3']
        acc['m4-7'] = acc['m4'] + acc['m5'] + acc['m6'] + acc['m7']   
        acc['m1-7'] = acc['m1'] + acc['m2'] + acc['m3'] + acc['m4'] + acc['m5'] + acc['m6'] + acc['m7']
        
        #user defined geometry 
        if not (user_defined_geometry is None):
            acc['custom_geom'] = sum([acc[mm] for mm in user_defined_geometry])
        
        tmp_keys = list(acc.keys()) # so we don't iterate on dict keys while changing its keys
        for key in tmp_keys:
            acc[key], acc[f'{key}_no_outliers_replaced'] = basic_filter(acc[key], outlier_thresh, replace_value  ) 
            acc[key]=list(acc[key])
            
            
        if ensure_1ms_sampling:
            for key in tmp_keys:
                if len( acc[key] ) != 10000:
                    acc[key] = list(np.nan * np.empty(10000))
                
            
        return(acc) 


def classify_freq_bin( f,psd, quantiles, opl_thresh, categories, freq_lims = [5, np.inf] ):
    """
    Classify new opl psd (um^2/Hz) based on historic quantiles 
    - f is float (frequency value of psd) 
    - psd is the  power spectral density to be classified, should also be a float
    - quantile is pandas series with index = ['q10','q20','q30','q40','q50','q60','q70','q80','q90'] quantiles indeally from a
    file like: 'UT{ut}/*UT{ut}*_{state_name}_*{focus_name}*{sensor_name}_psd_distribution_features.csv'. 
    - opl_thresh is a user defined opl threshold that we activate an alarm if psd value exceeds it (red classification) 
    """

    if len(categories)==4:
        if (psd < quantiles['q10']):
            c= categories[0] # good category, it is well below reference values (probably some system improvement)
        
        elif (psd >= quantiles['q10']) & (psd <= quantiles['q90']) :
            c = categories[1] # nominal category, it is within the reference quantile boundaries  
        
        elif (psd > quantiles['q90'])& (not psd > opl_thresh):
            c=categories[2] # bad category
        
        elif (psd > quantiles['q90']) & (psd > opl_thresh): 
            
            if (f>freq_lims[0]) & (f<freq_lims[-1]): # if within valid freq bins
                c=categories[3] # bad category with red flag alarm (above a threshold) 
        
            else: # we take down the classification
                c=categories[2] # bad category 
        else:
            raise TypeError('... we"re missing a case here') 
    else:
        raise TypeError('input categories must be a list of exactly length = 4')

    return( c )


def vibration_analysis(psd, detection_method = 'median', window=50, thresh_factor = 3, plot_psd = False, plot_peaks = False):
    
    '''
    
    from an input power spectral density (psd) this function detects 
    peaks above some threshold (determined by detection_method) and
    returns a pandas dataframe with the detected peak frequencies, 
    spectral width, prominence (height above the local psd floor) and the 
    absolute and relative contributions of the peak (the integrated psd and 
    and integrated psd - local floor over spectral width respectively)

    Parameters:
        psd (tuple): PSD tuple (freq, PSD) where freq and PSD are numpy arrays
        
        detection_method (string) : string to indicate the detection method 
        ('fit' or 'median' - fit assumes noise floor follows single power law over frequency domain)
        
        window (int) : the window size to apply rolling aggregate for calculating 
            the peak detection threshold (Only used for detection_method = 'median')
        
        thresh_factor (float) : factor to multiply a reference line (std or median) to set 
            vibration detection threshold 
            
        plot_psd (boolean) : Do you want to plot the PSD with the marked detected peaks?
        
        plot_peaks (boolean) : Do you want to plot the individual (cropped)
            detected peaks ?

    Returns:
        vibration_df (pandas dataframe): output dataframe containing the detected 
        peak frequencies, spectral width, prominence (height above the local 
        psd floor) and the absolute and relative contributions of the peak 
        (the integrated psd and and integrated psd - local floor over spectral 
        width respectively)  
    
    '''
    f,psd = psd 

    if detection_method == 'fit':

        df = np.diff(f)[0] #assumes this is linearly spaced! 
        param, cov = np.polyfit(np.log10(f), np.log10(psd), 1, cov=True) 

        grad,inter = param[0], param[1]
        dg, di = np.sqrt(cov[0,0]), np.sqrt(cov[1,1])

        psd_fit_1 = 10**(grad*np.log10(f) + inter) 
        psd_fit_1_uncert = psd_fit_1 * np.log(10) * np.sqrt( (dg*np.log10(f))**2  + (di)**2 ) #standard error propagration
        #psd_fit_1 = 10**(param[0]*np.log10(f) + param[1]) 


        #======== remove outliers
        outlier_thresh  = psd_fit_1 + 2 * psd_fit_1_uncert  #2 * psd_fit_1
        indx = np.where( abs(psd_fit_1 - psd) < outlier_thresh )
        
        #re-fit after removing outliers 
        param, cov = np.polyfit(np.log10(f[indx]), np.log10(psd)[indx], 1, cov=True)

        grad,inter = param[0], param[1]
        dg, di = np.sqrt(cov[0,0]), np.sqrt(cov[1,1])

        psd_fit_2 = 10**(grad*np.log10(f) + inter) 
        psd_fit_2_uncert = psd_fit_1 * np.log(10) * np.sqrt( (dg*np.log10(f))**2  + (di)**2 )  ##standard error propagration

        if plot_psd:
            fig,ax = plt.subplots(1,1) 
            ax.semilogy(f,psd_fit_1,label='fit without outlier removal',linestyle=':')
            ax.semilogy(f,psd_fit_2,label='fit with outlier removal')
            ax.semilogy(f,psd)
            ax.legend(fontsize=12)
            ax.set_xlabel('frequency [Hz]',fontsize=12)
            ax.set_ylabel('PSD ',fontsize=12)



        #======== get vibration regions (frequency blocks where PSD above threshold )
        vib_thresh = psd_fit_2 + thresh_factor * psd_fit_2_uncert # x2 std error of fit (with outlier rejection) 

        previous_point_also_above_threshold = False #init boolean that the previous point was not above vib_thresh
        block_indxs = [] # list to hold [block start, block end] indicies 
        block_indx = [np.nan, np.nan] # init current [block start, block end] indicies

 
        for i,_ in enumerate(f):
            
            if (i != len(f)) & (i!=0):
                if ( psd[i] > vib_thresh[i] ) & (not previous_point_also_above_threshold):
                    # then we are at the beggining of a block
                    block_indx[0] = i-1 # subtract 1 index from block start (since this is point above threshold)
                    previous_point_also_above_threshold = True
                    #print(block_indx )
        
        
        
                elif ( psd[i] <= vib_thresh[i] ) & previous_point_also_above_threshold:
                    #Then we are at an end of a block 
                    block_indx[1] = i 
        
                    # append block
                    block_indxs.append(block_indx)
                    # re-init block index tuple 
                    previous_point_also_above_threshold = False
                    block_indx = [np.nan, np.nan]
                    
            #deal with special case i=0 - we ignore this case, it is not necessary
            #elif  i==0:
            #    if ( psd[i] > vib_thresh[i] ):
            #        block_indx[0] = i 
            #        previous_point_also_above_threshold = True
                
            #deal with special case that last index is still within a block
            elif (i == len(f)) & (not np.isnan(block_indx[0])):
                
                block_indx[1] = i
                # append block
                block_indxs.append(block_indx)
        
        

        if plot_psd:

    
            fig, ax = plt.subplots(figsize=(8,5)) 
            ax.loglog(f,psd,color='k')
            ax.loglog(f,psd_fit_2, color='green',label=r'$af^{-b}$')
            ax.loglog(f,vib_thresh,color='red',label='detection threshold',linestyle='--')
            
            plt.fill_between(f, vib_thresh, psd, where=vib_thresh < psd,color='r',alpha=0.3,label='detected vibrations')
        
            ax.legend(fontsize=15)
            ax.set_xlabel('frequency [Hz]',fontsize=15)
            ax.set_ylabel('PSD '+r'$[m^2/Hz]$',fontsize=15)
            #lets visualize these 
            """for b in block_indxs: #period ends
                
                ax.axvspan(f[b[0]],f[b[1]],color='r',alpha=0.3)
                #plt.axvline(f[b[0]],color='g',linestyle=':') #green is start of block 
                #plt.axvline(f[b[1]],color='r',linestyle=':')"""


        # for each block detrend the PSD region with power law fit and then do peak detections (if wider then a few Hz)
        # get freq, width, prominence, relative OPL, absolute OPL 


        if plot_peaks: # set up axes to display the relative contributions from each peak 
            fig, ax = plt.subplots(round(np.sqrt( len(block_indxs) ) ), round(np.sqrt( len(block_indxs) ))+1 ,figsize=(15,15))
            axx = ax.reshape(1,-1)

        #init our dictionary to hold features 
        vib_feature_dict = {'vib_freqs':[], 'fwhm':[], 'prominence':[], 'abs_contr':[], 'rel_contr':[]}    
        for jj, (i1,i2) in enumerate(block_indxs):

            #detrend the psd with the power law fit
            psd_detrend = psd[i1:i2]/psd_fit_2[i1:i2] 

            if f[i2]-f[i1] < 50: # if block width < 50Hz we don't worry about fitting and just do basic calculations 



                #calculate frequency of vibration peak (detrend for this )
                i_max = np.argmax(psd_detrend)
                f_peak = f[i1:i2][i_max]

                #calculate prominence 
                prominence = psd[i1:i2][i_max] - psd_fit_2[i1:i2][i_max] # m^2 / Hz

                # interpolate this onto finner grid 
                fn_interp = interp1d(f[i1:i2] , psd_detrend , fill_value='extrapolate')
                f_interp = np.linspace(f[i1],f[i2] , 10*(i2 - i1) ) #grid 10x finer then current
                df_interp = np.diff(f_interp)[0]
                psd_detrend_interp = fn_interp(f_interp)

                # calculate FWHM as df x the how many frequency bins are above the half max within the block
                fwhm = df_interp * np.sum( psd_detrend_interp > np.max(psd_detrend_interp)/2 ) 

                # calculate absolute and relative OPL contributions of the vibration peak 
                abs_contr = np.trapz(psd[i1:i2+1], f[i1:i2+1]) # m^2
                rel_contr = np.trapz(psd[i1:i2+1], f[i1:i2+1]) - np.trapz(psd_fit_2[i1:i2+1], f[i1:i2+1]) # m^2

                if plot_peaks:
                    # for plotting we extend the range just a bit to make peaks clearer 
                    #plt.figure()

                    axx[0,jj].semilogy(f_interp, psd_detrend_interp)
                    axx[0,jj].semilogy(f_interp, np.ones(len(psd_detrend_interp)))

                    #axx[0,jj].semilogy(f[i1:i2+1], psd[i1:i2+1])
                    #axx[0,jj].semilogy(f[i1:i2+1], psd_fit_2[i1:i2+1])
                    #axx[0,jj].fill_between(f[i1:i2+1], psd_fit_2[i1:i2+1], psd[i1:i2+1] , where = psd_fit_2[i1:i2+1] <= psd[i1:i2+1],label='rel contr')
                    """
                    if i2-i1 > 2:
                        axx[0,jj].fill_between(f[i1:i2], psd_fit_2[i1:i2], psd[i1:i2] , where = psd_fit_2[i1:i2] <= psd[i1:i2],label='rel contr')
                    else:
                        axx[0,jj].fill_between(f[i1-2:i2+2], psd_fit_2[i1-2:i2+2], psd[i1-2:i2+2] , where = psd_fit_2[i1-2:i2+2] <= psd[i1-2:i2+2],label='rel contr')
                    #axx[0,jj].legend()"""


                vib_feature_dict['vib_freqs'].append(f_peak) #Hz
                vib_feature_dict['fwhm'].append(fwhm) #Hz 
                vib_feature_dict['abs_contr'].append(abs_contr) #m^2 
                vib_feature_dict['rel_contr'].append(rel_contr) #m^2 
                vib_feature_dict['prominence'].append(prominence) #m^2/Hz

            else: # to do.. implement multiple curve fitting within block
                raise TypeError('THIS CASE IS NOT CODED ',f[i1]) 
                # estimate how many peaks are in this block 

                # for each peak fit a lorenzian profile on the detrended PSD with initial guess centered on peak freq

                # extract features from fit 


                
    elif detection_method == 'median':

        a1, a0 = np.polyfit(np.log10(f), np.log10(psd), 1) 
        linfit = 10**(a0 + a1 * np.log10(f[1:]))


        med_floor = pd.Series(psd).rolling(window,center=True).median()
        h_thresh = thresh_factor * med_floor.values  #freqs**0.2 * pd.Series(psd).rolling(100,center=True).mean().values #2*med_floor.values #peak needs to be x2 neighborhood median to be counted 

        peaks = sig.find_peaks(psd, height = h_thresh) #height = height.values
        peak_proms = sig.peak_prominences(psd, peaks[0], wlen=50)
        peak_widths = sig.peak_widths(psd,peaks[0],prominence_data=peak_proms,rel_height=0.9)
        #calc widths at base where peak prominence is measured 




        #contour height to indicate prominence 
        contour_heights = psd[peaks[0]] - peak_proms[0]

        # look at widths 
        li = (peaks[0]-np.round(peak_widths[0]/2)).astype(int)
        ui = (peaks[0]+np.round(peak_widths[0]/2)).astype(int)

        #plot peaks inidcating their calculated prominence 
        if plot_psd:
            plt.figure()

                # plot psd, median floor and peak detection threshold
            plt.semilogy(f,psd)
            plt.semilogy(f,med_floor,linestyle=':', color='k' )
            plt.semilogy(f, h_thresh, linestyle=':', color='r' )

            plt.semilogy(f[peaks[0]],psd[peaks[0]],'.',color='r')
            plt.vlines(x=f[peaks[0]], ymin=contour_heights, ymax=psd[peaks[0]],color='k')
            #plt.xlim([0,30])

            #plt.semilogy(f[li],psd[li],'x',color='g')
            #plt.semilogy(f[ui],psd[ui],'x',color='b')

            plt.xlabel('Frequency [Hz]',fontsize=14)
            plt.ylabel('PSD',fontsize=14)        
            plt.gca().tick_params(labelsize=14)

        cummulative_psd = np.cumsum( psd[::-1] )[::-1] * np.nanmedian( np.diff(f) ) 
        cummulative_med = np.cumsum( np.nan_to_num(med_floor,nan=0.0)[::-1] )[::-1] * np.nanmedian( np.diff(f) )

        abs_contr, rel_contr = [], [] #lists to hold absolute and relative PSD peak contributions 

        for i, (lo,up) in enumerate(zip(li,ui)):

            if up - lo == 2: #if only 2 samples in upper/lower limits we add one sampe (so symmetric around peak point)
                up+=1
            
            interp_base =  np.nan_to_num( med_floor[lo:up] , nan=0.0)  #linfit[lo:up] 

            abs_contr.append( cummulative_psd[lo] - cummulative_psd[up] )
            #abs_contr.append( np.trapz( psd[lo:up], f[lo:up] ) ) #abs_contr.append( cummulative[up] - cummulative[lo] )  #m^2 (or what ever units psd input is 

            rel_contr.append( cummulative_psd[lo] - cummulative_psd[up] - (cummulative_med[lo] - cummulative_med[up]) )
            #rel_contr.append( np.trapz( psd[lo:up] - interp_base, f[lo:up] ) )  #m^2

            #need to visulize relative level for each peak
            if plot_peaks :
                plt.figure()
                plt.plot(f[lo-5:up+5], psd[lo-5:up+5] ,color='k')
                plt.plot(f[lo:up],interp_base,color='orange')
                plt.plot(f[peaks[0][i]],psd[peaks[0][i]],'.',color='b',label='detected peak')
                plt.fill_between(f[lo:up],y1=interp_base,y2=psd[lo:up],color='red',alpha=0.5,label='relative contr.') 

                plt.xlabel('Frequency [Hz]',fontsize=14)
                plt.ylabel('PSD',fontsize=14)        
                plt.gca().tick_params(labelsize=14)

                plt.text(f[peaks[0][i]], 1.1*psd[peaks[0][i]], 'peak frequency = {:.1f}Hz,\nrel contribution = {:.3e}'.format(f[peaks[0][i]],rel_contr[i]))
                plt.ylim([0, 1.3*psd[peaks[0][i]] ])
                plt.legend(loc='lower left')



        vib_feature_dict = {'vib_freqs':f[peaks[0]], 'fwhm':peak_widths[0], 'prominence':peak_proms[0], \
         'abs_contr':abs_contr, 'rel_contr':rel_contr}


    return(vib_feature_dict)

