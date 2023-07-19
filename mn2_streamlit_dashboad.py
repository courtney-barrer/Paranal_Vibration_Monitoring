#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:11:26 2023

@author: bcourtne

run in virtual environment 
source venv/bin/activate
see https://gitlab.eso.org/datalab/dlab_python_libraries/dlt/-/wikis/Jupyter-Playground-on-your-Laptop 

To do 

Get rid of latest_data returned in prepare_current_status function (need to make ts plotting use current_acc_dict and NOT latest data! 

"""
import streamlit as st
import mpld3
import streamlit.components.v1 as components
import numpy as np
import os
import subprocess
import pandas as pd
import glob
from PIL import Image
import datetime 
import h5py
import matplotlib.pyplot as plt 
import scipy.signal as sig
from scipy.interpolate import interp1d 
import pickle
import plotly.express as px

# for datalab functionality 
import dlt # this will only work if we are in virtual environment (run in cmd line >>cd /Users/bcourtne/Documents/mac1_bk/jupyter_playground. Then >> source venv/bin/activate.
# see https://gitlab.eso.org/datalab/dlab_python_libraries/dlt/-/wikis/Jupyter-Playground-on-your-Laptop 

# then our own functions 
os.chdir('/home/bcourtne/mn2_dashboard/jupyter_playground')
import mn2_dashboard_functions as mn2

#mapping from sensor indices to mirror position 
st.set_page_config(layout="wide")

# To do: ideal will be to cache it, maybe don't return the raw hdf5 files... especially since these are large daily file. only keep processed signal from most recent sample!
#@st.cache(suppress_st_warning=True) # hdf5 files are not easily hashable so difficult for caching 
@st.cache_resource
def prepare_current_status(now, post_upgrade):
    year = str(now.year)
    month = "%02d" % (now.month,)

    progress_text = "Collecting and classfying the most recent data. This can take around 2 minutes. Please wait."
    percent_complete = 0
    my_bar = st.progress(percent_complete, text=progress_text)

    latest_data = {} # 
    latest_files = {} # for each UT
    acc_dict = {} # to hold processed most recent samples for each UT

    for ut in [1,2,3,4]:
        percent_complete += 25
        my_bar.progress(percent_complete, text=progress_text)

        print( f'downloading\n {datalab_reports_url}/raw/{year}/{month}/ldlvib{ut}_raw_{now.strftime("%Y-%m-%d")}.hdf5')
        print( ' to : ',f'{local_data_path}/raw/{year}/{month}/')
        #try: # we should really have a seperate cron job that does this to keep an updated local file and then access this istead of downloading it everytime here 
        tmp_put_path = f'{local_data_path}/raw/{year}/{month}/'
        tmp_get_path = f'{datalab_reports_url}/raw/{year}/{month}/'
        tmp_filename = f'ldlvib{ut}_raw_{now.strftime("%Y-%m-%d")}.hdf5'
        
        tmp_files_in_put_path = glob.glob(f'{local_data_path}/raw/{year}/{month}/ldlvib{ut}_raw_*.hdf5')
        time_ref_filename=f'{tmp_put_path}/tmp_file_time_ref.txt' # this will be used as time reference for the file creation date in current OS (see os.path.getctime))
        
        if not os.path.exists(tmp_put_path):
            print('\n\n\n not os.path.exists(tmp_put_path) \n\n\n')
            os.makedirs(tmp_put_path)                     
            
            # download the data (obviously we don't have it if we had to make a new folder )
            subprocess.check_call(['./get_latest_data.sh', tmp_get_path + tmp_filename, tmp_put_path])           
            #subprocess.check_call(['./get_latest_data.sh',f'{datalab_reports_url}/raw/{year}/{month}/ldlvib{ut}_raw_{now.strftime("%Y-%m-%d")}.hdf5', f'{local_data_path}/raw/{year}/{month}/'])           
            print (['./get_latest_data.sh', tmp_get_path+tmp_filename, tmp_put_path])

        elif os.path.exists(tmp_put_path):
            
            print('\n\n\n os.path.exists(tmp_put_path)\n\n\n')
            
            if len(tmp_files_in_put_path)>0: # if we actually have files in directory
                
                print('\n\n\n len(tmp_files_in_put_path)>0\n\n\n')
            
                if tmp_put_path + tmp_filename not in glob.glob(tmp_put_path + '*'): # if the target file not in folder then we download
                    # if not in the local data folder then download it
                    print('\n\n\n tmp_put_path + tmp_filename not in glob.glob(tmp_put_path + *) \n\n\n')
                    
                    subprocess.check_call(['./get_latest_data.sh', tmp_get_path+tmp_filename, tmp_put_path])  
                    print (['./get_latest_data.sh', tmp_get_path + tmp_filename, tmp_put_path])    
                else: # we have the target file.. now we just have to check how old it is (they are daily files that get updated every ~15 minutes !)
                    
                    print('\n\n\n tmp_put_path + tmp_filename in glob.glob(tmp_put_path + *) --- \n\n\n')
                    tmp_file = open(time_ref_filename,"w") # used as time reference for the file creation date in current OS (see os.path.getctime))
                    time_difference = os.path.getctime(time_ref_filename) - max([os.path.getctime(f) for f in tmp_files_in_put_path]) # seconds 
                    #print( 'time difference =',time_difference )
                    if time_difference > 60*60 : # if time difference is > 1hr then we re-download the data 
                        print('\n\n\n last download of daily file was ',time_difference/60,' minutes ago, therefore downloading an updated file\n\n\n')
                        os.remove(tmp_put_path+tmp_filename) # deleete the old file first 
                        subprocess.check_call(['./get_latest_data.sh', tmp_get_path+tmp_filename, tmp_put_path])  
                        print (['./get_latest_data.sh', tmp_get_path + tmp_filename, tmp_put_path])  
                        
                    else:
                        print(f'we will read the current file we have: {tmp_get_path + tmp_filename}')
                    tmp_file.close() # close the temporal reference file

            else: # no files in tmp_put_path, so we download it 
                print('\n\n\n NO files in tmp_files_in_put_path \n\n\n')
                subprocess.check_call(['./get_latest_data.sh', tmp_get_path+tmp_filename, tmp_put_path])  
                print (['./get_latest_data.sh', tmp_get_path + tmp_filename, tmp_put_path]) 
                
            
        else: # we should NEVER get this case 
            raise TypeError(f'PATH:{local_data_path}/raw/{year}/{month} does not match existence cases.. how?' )
        
        
        
        
        # now read into our dictionaries  
        latest_files[ut] = tmp_put_path + tmp_filename 
        latest_data[ut] = h5py.File(latest_files[ut], 'r') #{t:{frozenset(h5[t][s][:]) for s in h5[t]} for t in h5} # this is to make latest data hashable so it can be caches (cant cache h5py files)!!!  
        
        # need time_key to be one that is closest to now (not necessarily the most recent, in case we want to look back in time!) 
        current_date_tmp = now.strftime("%Y-%m-%d")

        current_time_key = np.array(list( latest_data[ut].keys() ) )[ np.argmin([abs(now - datetime.datetime.strptime(current_date_tmp +'T' + t, '%Y-%m-%dT%H:%M:%S') ) for t in latest_data[ut].keys() ]) ]
        
        acc_dict[ut] = mn2.process_single_mn2_sample(latest_data[ut], current_time_key, \
                                                      post_upgrade=post_upgrade, user_defined_geometry=None, outlier_thresh = 9, replace_value=0, ensure_1ms_sampling=False)


        #acc_dict[ut] = mn2.process_single_mn2_sample(latest_data[ut], list(latest_data[ut].keys())[-1], post_upgrade=True, user_defined_geometry=None, outlier_thresh = 9, replace_value=0, ensure_1ms_sampling=False)
        
        #except:     
        #    print( f'failed to download\n {datalab_reports_url}/raw/{year}/{month}/ldlvib{ut}_raw_{now.strftime("%Y-%m-%d")}.hdf5')  

    # telescope states                                                                                                                   
    tel_state_dict = {}
    for ut in [1,2,3,4]:
        tel_state_dict[ut]=mn2.get_state(now, ut)

    # process and cache PSDs from current data                                                                                           
    current_psd_dict = prepare_psd_data(acc_dict)

    # get reference PSDs relevant for classifying the current psds                                                                       
    current_ref_psds = get_reference_psds(current_psd_dict, tel_state_dict)

    #latest_files, latest_data, acc_dict ,tel_state_dict, current_psd_dict, current_ref_psds
    return(latest_files, latest_data, acc_dict ,tel_state_dict, current_psd_dict, current_ref_psds, current_time_key ) # latest_files, latest_data, acc_dict


@st.cache_data
def prepare_psd_data(acc_dict):
    #from processed accelerometer data (acc_dict) we extract and cache the PSDs
    current_psd_dict = {}
    for ut in acc_dict.keys(): #[1,2,3,4]:

        current_psd_dict[ut] = {}
        
        for s in acc_dict[ut].keys():#sensor_list :
            
            current_psd_dict[ut][s] = {}
            # double integrate                                                                                                                                                  
            f_acc, acc_psd = sig.welch(acc_dict[ut][s], fs=1e3, nperseg=2**11, axis=0)
            f_pos, pos_psd = mn2.double_integrate(f_acc, acc_psd)
            current_psd_dict[ut][s]['pos'] = (f_pos, pos_psd)
            current_psd_dict[ut][s]['acc'] = (f_acc, acc_psd)

    return( current_psd_dict )

@st.cache_data
def get_reference_psds(psd_dict, tel_state_dict):
    
    ref_psd_dict = {} 
    for ut in psd_dict.keys() : #[1,2,3,4]:

        ref_psd_dict[ut] = {}

        for s in psd_dict[ut].keys():

            ref_psd_dict[ut][s] = {}

            no_ref_file_flag = 0 # no_ref_file_flag is a flag to indicate if no reference file is found and non-empty                                                                                                                             
        
            print(f'\n\nBEGIN searching for reference psd file for {s} on UT{ut}')
        
            tmp_reference_psd_file, no_ref_file_flag = mn2.get_psd_reference_file(local_reference_psd_path, ut, s, tel_state_dict)
            #tmp_reference_psd_file = mn2.get_psd_reference_file(local_reference_psd_path, ut, state_name, focus_name, sensor_name)                                                                                                                           
            #####                                                                                                                                                                                                                                            
            #THE REFERENCE PSD (quantiles, moments etc). Used for classification                                                                                                                                                                             
            if not  no_ref_file_flag:
                ref_psd_features = pd.read_csv( tmp_reference_psd_file[0] ,index_col=0)
            else: # if we have a raised no_ref_file_flat we just set the ref_psd_features to empty df which we catch and deal with below                                                                                                                     
                ref_psd_features = pd.DataFrame([])
            #####                                                                                                                                                                                                                                             
            if ref_psd_features.empty: #                                                                                                                                                                                                                     
                print(f'\n!!!!!!\n\nPSD reference file {ref_psd_features} is empty or non-existant in path.\n\n  This can occur when there was insufficient PSD data to derive statistics (e.g. a M1 recoating or instrument change for the focus etc).\n\n   We will look at a new reference file not filtering for focus...')
        
                # copy the state dict and change focus to force get_psd_reference_file to consider allfoci (we dont filter for focus)
                tmp_tel_state_dict= tel_state_dict.copy()  #create copy                                                                                                                                                                                      
                #force focus to undefined state so that we catch the else statement in mn2.get_psd_reference_file                                                                                                                                            
                tmp_tel_state_dict[ut]['Coude']=0
                tmp_tel_state_dict[ut]['Nasmyth A']=0
                tmp_tel_state_dict[ut]['Nasmyth B']=0
                tmp_tel_state_dict[ut]['Cassegrain']=0
                # now try again        
                tmp_reference_psd_file, no_ref_file_flag = mn2.get_psd_reference_file(local_reference_psd_path, ut, s, tmp_tel_state_dict)
                # read in again        
                if not  no_ref_file_flag:
                    ref_psd_features = pd.read_csv( tmp_reference_psd_file[0] ,index_col=0)
                else:
                    ref_psd_features = pd.DataFrame([])
                if ref_psd_features.empty: #check if empty again                                                                                                                                                                                             
                    # if its still empty then we raise the no_ref_file_flag                                                                                                                                                                                  
                    no_ref_file_flag = 1
                    print('\nEven considering all Foci we cannot find a suitable reference psd.\nConsider finding & uploading a new reference file for the given UT, focus, state, and sensor\n!!!!\n')
                else:
                    print('phewwwww... we got one\n\n!!\n')
            
            ref_psd_dict[ut][s] = ref_psd_features
    
    return(ref_psd_dict)



def classify_psds(psd_dict, ref_psd_dict, report_card):
    
    for ut in psd_dict.keys() : #[1,2,3,4]:                                                                                                                                                                                
            
        if ut not in report_card:
            report_card[ut]={}

        for s in psd_dict[ut].keys():
    
            if s not in report_card[ut]:
                report_card[ut][s]={}

            ref_psd = ref_psd_dict[ut][s] # isolate the df from the given sensor and UT
            
            if not ref_psd.empty: 
                if 'sensor' not in s: #looking at combined geometries scaling is different in this case so we need to apply it                                                                                                                             
                    f_tmp, psd_tmp = psd_dict[ut][s]['pos'] # m^2/Hz 
                    # psd_tmp *= 1e12 #  um^2/Hz                                                                                                                                                                                                           
                else: #looking at raw sensors scaling is different in this case so we need to apply it                                                                                                                                                       
                    f_tmp, psd_tmp = psd_dict[ut][s]['pos']
                    #psd_tmp *= mn2.V2acc_gain * 1e12 # convert V^2/Hz to um^2/Hz (V2acc_gain since we are looking at raw sensor values)                                                                                                                     

                f_ref = np.array(list(ref_psd.index)) # reference frequency to interpolate onto                                                                                                                                                      
                interp_fn = interp1d( f_tmp, psd_tmp , kind = 'linear' , bounds_error=False, fill_value=np.nan)
                current_psd_tmp = interp_fn( f_ref  ) # interpolate the current psd onto the reference psd frequencies (AND KEEP IN m^2/Hz)                                                                                                                                     
 
                report_card[ut][s]['psd-data'] = (f_ref, current_psd_tmp)  
                # get some metrics 
            
                # log euclidean distance to reference median  
                log_dist = np.sqrt( np.mean(( np.log10(current_psd_tmp ) - np.log10( 1e-12 * ref_psd['q50']) )**2 ) )  # ref PSD should be in um^2/Hz while current_psd_tmp in m^2/Hz!!! IF REFERENCE FILE CHANGES THAN THIS LINE NEEDS TO BE UPDATED

                # psd frequencies above reference 90th percentile (worse than usual) 
                degraded_freqs = f_ref[ current_psd_tmp > 1e-12 * ref_psd['q90'].values ]
                               
                # psd frequencies below reference 10th percentile (better than usual)
                improved_freqs = f_ref[ current_psd_tmp < 1e-12 * ref_psd['q10'].values ]
            
                # calculate vibration peaks 
                vib_detection_df = mn2.vibration_analysis( (f_ref,current_psd_tmp ), detection_method = 'median', window=vib_det_window, thresh_factor = vib_det_thresh_factor, plot_psd = False, plot_peaks = False) 
                
                if 'sensor' in s:
                    strong_vibs_indx = np.array(vib_detection_df['rel_contr']) > (1/mn2.V2acc_gain * 150e-9)**2 # relative contributions > 100nm RMS  
                else:
                    strong_vibs_indx = np.array(vib_detection_df['rel_contr']) > (150e-9)**2 # relative contributions > 100nm RMS
                
                report_card[ut][s]['psd-log_dist'] = log_dist
                report_card[ut][s]['psd-degraded_freqs'] = degraded_freqs
                report_card[ut][s]['psd-improved_freqs'] = improved_freqs
                report_card[ut][s]['psd-vib_detection_df'] = vib_detection_df #vib_peak_df
                report_card[ut][s]['psd-strong_vibs_indx'] = strong_vibs_indx
                report_card[ut][s]['psd-strong_vibs'] = vib_detection_df['vib_freqs'][strong_vibs_indx]
                report_card[ut][s]['psd-reference_psd_available'] = 1
            else:
                report_card[ut][s]['psd-reference_psd_available'] = 0


            #if np.sqrt( np.mean(( np.log10( current_psd_tmp ) - np.log10( ref_psd['q50'] ) )**2 ) 
            #quantile_df = ref_psds[[f'q{int(x)}' for x in np.linspace(10,90,9)]] # the reference psd quantiles                                                                                                                                    
        
            #freq_classifications = [mn2.classify_freq_bin(f_ref[i], current_psd_tmp[i], quantile_df.iloc[i], opl_thresh=opl_thresh, categories = psd_classification_categories) for i in range(len(quantile_df))]
        
            # make as panda series indexed by frequency and append to dictionary                                                                                                                                                                          
            #freq_classification_dict[ut][s] = pd.Series( freq_classifications , index = f_ref )
        
            # write frequencies with red flags                                                                                                                                                                                                            
            #red_flag_freqs[ut][s] = f_ref[np.where(freq_classification_dict[ut][s].values == psd_classification_categories[-1])]

    return( report_card ) 


def classify_ts(acc_dict): 
    print('to do')


def get_status_from_report_card(report_card):
    
    status_index = ['m1-3','m4-7','m1-7'] + [f'm{i}' for i in range(1,8)] + [f'sensor_{i}' for i in sensor_idx_lab ] # sensors to report
    mn2_status_df = pd.DataFrame({ut:['OK' for _ in range(len(status_index))] for ut in [1,2,3,4]} , index = status_index)
    status_justification_df = {ut:{s:[] for s in status_index} for ut in [1,2,3,4]}
    
    for ut in mn2_status_df.columns:
        print(ut)
        if ut in report_card:

            for s in mn2_status_df.index:

                if s in report_card[ut]:
            
                    if sum(report_card[ut][s]['psd-strong_vibs_indx'])>0:
                        mn2_status_df.at[s,ut]='NOK'
                        status_justification_df[ut][s].append( [f'strong vibrations with relative contribution > 150nm RMS for {s} on UT{ut} at frequencies: {report_card[ut][s]["psd-strong_vibs"]}'] )
                   
                    if report_card[ut][s]['psd-log_dist'] > 3:
                        mn2_status_df.at[s,ut]='NOK'
                        status_justification_df[ut][s].append( [f'psd rms distance is more than 3 orders of magnitude > reference mean (disconnected sensor?)'] )

                    if 0:
                        print('etc')


                else:
                    mn2_status_df.at[s,ut]='NA'
                    status_justification_df[ut][s].append( [f'no {s} in report_card for UT{ut}'] )


        else:
            mn2_status_df[ut]='NA'
            for s in mn2_status_df.index:
                status_justification_df[ut][s].append( [f'no UT{ut} in report_card'] )

    return( mn2_status_df, status_justification_df ) 



#++++++++++++++++++++++++++++++++ Global Variables
#####                                                                                                                                                                           
now = datetime.datetime.utcnow() - datetime.timedelta(days = 1)  #datetime.datetime.strptime('2023-05-12 02:14:00', '%Y-%m-%d %H:%M:%S')   #datetime.datetime.utcnow()                                                                      

# put in check if now.hour = 0 and now.minute < 10 , if so we will be looking at a new daily file that may not be uploaded yet, so we go back to the end of previous day 
if (now.hour==0) & (now.minute<15):
    now = now - datetime.timedelta(minutes = 25)
####     

#paths 
datalab_reports_url = 'http://reports.datalab.pl.eso.org/data/manhattan'
local_data_path = '/home/bcourtne/mn2_dashboard/local_data_path/reports.datalab.pl.eso.org/data/manhattan/'
local_figure_path =  '/home/bcourtne/mn2_dashboard/jupyter_playground/figures'
local_vibdetection_path = '/home/bcourtne/mn2_dashboard/jupyter_playground/detected_vibration_tables'
local_reference_psd_path = '/home/bcourtne/mn2_dashboard/local_data_path/reference_psds/' # path to reference psds used in some classification methods 

#clssification classes  
#psd_classification_categories = ['improved','normal','degraded','dangerous'] # classes for psd classification per frequency bin (red fl
#class should always be last! otherwiseredflag detections will be off
#psd_classification_colors = ['green','silver','orange','red'] # colors to go with classification  

#threshold_dict = {'opl_thresh':1e8}  # put threshold here 
#opl_thresh = 1e8 # (um^2) used for 'dangerous' classification (alarm if exceeds threshold in freq bin (~1Hz)

#mapping from sensor indices to mirror position                                                                                                                                 
i2m_dict = {1:'M3a',2:'M3b',3:'M2',4:'empty',5:'M1+y',6:'M1-x',7:'M1-y',8:'M1+x',9:'M4',10:'M5',11:'M6',12:'M7'}
m2i_dict = {v: k for k, v in i2m_dict.items()}
mirror_lab = [m for m in m2i_dict if m!='empty']
sensor_idx_lab = [i for i in i2m_dict if i2m_dict[i]!='empty']

#define date when MNII upgrade was complete (addition of accelerometers from m4-m7)
mn2_upgrade_date = datetime.datetime.strptime('2023-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')

# define our sensor list (if we consider upgraded network of accelerometers?)                                                                                                   
if now > mn2_upgrade_date:
    post_upgrade = True
    # define our valid sensor keys for acc_dict and current_psd_dict                                         
    # NOTE: that mn2.process_single_mn2_sample( ) function doesn't take this as input 
    # (it has it predefined by post_upgrade variable.. but maybe we should includr the sensor_key_list here 
    sensor_key_list = ['m1-3','m4-7','m1-7'] + [f'm{i}' for i in range(1,8)] + [f'sensor_{i}' for i in sensor_idx_lab ]
else:
    post_upgrade = False
    # define our valid sensor keys for acc_dict and current_psd_dict                                                                                                            
    sensor_key_list = ['m1-3'] + [f'm{i}' for i in range(1,4)] + [f'sensor_{i}' for i in sensor_idx_lab[:7] ]



vib_det_window = 50 # # samples to calculate rolling median 
vib_det_thresh_factor = 3 # how many times above the rolling median does a psd bin have to be in-order to be classified a peak detection 

#++++++++++++++++++++++++++++++++ Download and/or Read in the latest data and prepare it (taken from daily file)

# process the input files with core results stored in current_acc_dict
latest_files,latest_data, current_acc_dict, tel_state_dict,current_psd_dict, current_ref_psds, current_time_key = prepare_current_status(now, post_upgrade) 
#latest_files, latest_data,  current_acc_dict, tel_state_dict,current_psd_dict, current_ref_psds
tel_state_df = pd.DataFrame(tel_state_dict)
tel_state_df.columns = ['UT1','UT2','UT3','UT4']

#++++++++++++++++++++++++++++++++ Classify data  

report_card = {} # init report card

# populate report card with our classification functions 
report_card=classify_psds(current_psd_dict , current_ref_psds, report_card)
#classify_ts(acc_dict, report_card)

# generate a UT/sesnor statuses based on the report card  
mn2_status_df, status_justification_df = get_status_from_report_card( report_card )



#++++++++++++++++++++++++++++++++ Begin Dashboard

st.header('PARANAL VIBRATION DASHBOARD')


tabs = st.tabs(["current status","analyse data", "vibration source inventory"])


with tabs[0]:
    
    st.header(now) # print what date/time we're looking at 
    #++++++++++++++++ Top Level Status of most recent data 
    col1, col2, col3 = st.columns(3)

    #mn2 status 
    #mn2_status_df = pd.DataFrame( {f'UT{i}':['OK' for _ in range(11)] for i in range(1,5)} , index = [f'sensor {i+1} ({m})' for i, m in enumerate(mirror_lab )])

    #put telescope state dictionary into dataframe and rename columns (for display purposes) 
    tel_state_df = pd.DataFrame(tel_state_dict)
    tel_state_df.columns = ['UT1','UT2','UT3','UT4']
    
    # mn2 diagram 
    mn2_diagram = Image.open(f'{local_figure_path}/MNII_system.jpg')

    def color_mn2_status_df(val):
        color = 'red' if val == 'NOK' else 'green'
        return('color: %s' % color)

    # put data in the tabs respective columns 
    col1.dataframe( mn2_status_df.style.applymap(color_mn2_status_df), height = 420)
    #col2.dataframe(tel_state_df)
    col3.image(mn2_diagram, caption='The Manhattan network of accelerometers')

    # prep for classification 
    # now get state names consistent with feature_distribution csv file naming convention
    #get_reference_file(ut, sensor_name, tel_state_dict):

    #st.header(red_flag_freqs[ut][s])

    display_status_justification = st.checkbox('display_status_justification')
    if display_status_justification:
        st.dataframe( pd.DataFrame(status_justification_df ) , width=2000)  

    st.download_button("Download Report Card",data=pickle.dumps(report_card), file_name=f'mn2_report_card_{now}.pkl',)

    #++++++++++++++++ Plot latest time series  

    
    latest_ts_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Most Recent Time Series</p>'
    st.markdown(latest_ts_title, unsafe_allow_html=True)

    which_ts_plot =  st.selectbox('Time Series Plot Options', ('unfiltered', 'filtered'))
    #st.sidebar.selectbox(label='Time Series Plot Options', options=('unfiltered', 'filtered'))

    fig1,ax1 = plt.subplots(2,2,figsize=(10,8))
    plt.subplots_adjust(hspace=0.5,wspace=0.2)
    sat_check_dict = {}

    if post_upgrade:
        for ut,ax in zip([1,2,3,4],ax1.reshape(-1)):
            
            sat_check_dict[ut]={}
            #time_key = list(latest_data[ut].keys())[-1]
            filen = latest_files[ut].split("/")[-1]

            for i,acc in enumerate([1,2,3,5,6,7,8,9,10,11,12]):
                if which_ts_plot == 'unfiltered':
                    acc_data  = np.array( latest_data[ut][current_time_key][f'sensor{acc}'][:] )
                else:
                    acc_data = np.array(current_acc_dict[ut][f'sensor_{acc}'] )
                sat_check_dict[ut][acc] = np.nansum(abs(acc_data) > 10)
                #acc_data = 10*i + latest_data[ut][time_key][f'sensor{acc}'][:]
                ax.plot(10*i + acc_data,'k',alpha=0.9)
                ax.text(0, 10*i+2, i2m_dict[acc], fontsize=12 ,color='red')
            
            ax.set_title(f'UT{ut}\n{filen} @ {current_time_key}UT') #ax.set_title(f'UT{ut}\n{latest_files[ut].split("/")[-1]} @ {time_key}UT')
            #ax.set_xlabel('samples',fontsize=20)
            #ax.set_ylabel('acceleration '+r'$(m/s^2)$'+'\n',fontsize=20)
            ax.set_yticks([])
            ax.tick_params(labelsize=12)

            if np.any([sat_check_dict[ut][a] > 0 for a in sat_check_dict[ut]]):
                ax.set_facecolor("pink")
                ax.text(100, 0, 'sensor(s) {} with some voltages at amplifier\nsaturation limit (do we have ADC Spikes?)'.format(1+np.where([sat_check_dict[ut][a] > 0 for a in sat_check_dict[ut]])[0]), style='italic', bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 10})
    else:
        for ut,ax in zip([1,2,3,4],ax1.reshape(-1)):   
            sat_check_dict[ut]={}
            #time_key = list(latest_data[ut].keys())[-1]
            filen = latest_files[ut].split("/")[-1]
            for i,acc in enumerate([1,2,3,5,6,7,8]):
                acc_data  = latest_data[ut][current_time_key][f'sensor{acc}'][:]
                sat_check_dict[ut][acc] = np.nansum(abs(acc_data) > 10)

                #acc_data = 10*i + latest_data[ut][time_key][f'sensor{acc}'][:]
                ax.plot(10*i + acc_data,'k',alpha=0.9)
                ax.text(0, 10*i+2, i2m_dict[acc], fontsize=12 ,color='red')
            
            ax.set_title(f'UT{ut}\n{filen} @ {current_time_key}UT') #ax.set_title(f'UT{ut}\n{latest_files[ut].split("/")[-1]} @ {time_key}UT')
            #ax.set_xlabel('samples',fontsize=20)
            #ax.set_ylabel('acceleration '+r'$(m/s^2)$'+'\n',fontsize=20)
            ax.set_yticks([])
            ax.tick_params(labelsize=12)
            if np.any([sat_check_dict[ut][a] > 0 for a in sat_check_dict[ut]]):
                ax.set_facecolor("pink")
                ax.text(100, 0, 'sensor(s) {} with some voltages at amplifier\nsaturation limit (do we have ADC Spikes?)'.format(1+np.where([sat_check_dict[ut][a] > 0 for a in sat_check_dict[ut]])[0]), style='italic', bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 10})
             
    #fig1_html = mpld3.fig_to_html(fig1)
    #components.html(fig1_html, height=600)
    st.pyplot(fig1)
    
    if st.button('download timeseries figure'):
        fig1.savefig(os.path.join(local_figure_path, f'streamlit_ts_{now.strftime("%d-%m-%YT%H:%M:%S")}_{which_ts_plot}_fig.png'))






    #++++++++++++++++ Plot PSDs

    # IMPORTANT : what we plot here is report_card[ut][s]['psd-data'] (i.e. the ut/sensor PSD position that has been interpolated onto the reference PSD frequencies.)
    # reference and current PSD frequencies should naturally match, but interpolation is done to ensure this! 
    # It is important the current and reference PSDs have equal freq bins to properly compare the PSDs per freq bin when classifying, 
    # therefore we show here the psd that is actually being useed for classification (i.e the interpolated one)  

    latest_ts_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Most Recent Power Spectral Densities (PSDs)</p>'
    st.markdown(latest_ts_title, unsafe_allow_html=True)
    
    
    
    psd_available_sensors = [s for s in list( current_psd_dict[list(current_psd_dict.keys())[0]].keys() ) if 'outliers' not in s] 

    psd_plot_usr_options = psd_available_sensors[::-1] + ['all mirrors', 'combined geometries']     
    
    # user options 
    usr_which_psd_plot = st.selectbox('PSDs Plot Options', options=tuple(psd_plot_usr_options ))
    
    usr_display_ref = st.checkbox('display 10-90 percentile of the reference PSD')

    usr_display_vib = st.checkbox('display detected vibration peaks')

    # Plot def plot_current_psd( current_psd_dict,  current_ref_psds, usr_which_psd_plot , usr_display_ref) 
    fig2,ax2 = plt.subplots(2,2,sharex=False, figsize=(10,10))                                                         
    plt.subplots_adjust(hspace=0.5,wspace=0.5)   

    for ut,ax in zip([1,2,3,4],ax2.reshape(-1)):  
        if ut in report_card: #current_psd_dict: 
            
            # define the file name and time key we are plotting (for titles). Note this is how it is defined in input processing function 
            filen = latest_files[ut].split("/")[-1]
            #time_key = list(latest_data[ut].keys())[-1]
            if post_upgrade:                                                                                          
                mirror_plot_list = [f'm{i}' for i in range(1,8)]
                mirror_plot_colors = ['r','b','g','orange','purple','grey','cyan']

                comb_g_plot_list = ['m1-3','m4-7','m1-7']
                comb_g_plot_colors = ['r','b','g']
            else:                                                                                                     
                mirror_plot_list = [f'm{i}' for i in range(1,4)]
                mirror_plot_colors = ['r','b','g']           

                comb_g_plot_list = ['m1-3']
                comb_g_plot_colors = ['r']

            if usr_which_psd_plot == 'all mirrors':                                                                            
                for s,col in zip(mirror_plot_list , mirror_plot_colors):         
                    f, psd = report_card[ut][s]['psd-data']  #current_psd_dict[ut][s]['pos'][0], current_psd_dict[ut][s]['pos'][1]                                                        
                    ax.loglog(f, 1e12 * psd, color=col, linestyle='-',label=s)                                                     
                    ax.loglog(f, 1e12 * np.cumsum( psd[::-1])[::-1] * np.diff(f)[1], color=col, linestyle=':')     

                                           
            elif usr_which_psd_plot == 'combined geometries':
                
                for s,col in zip(comb_g_plot_list ,comb_g_plot_colors):
                    f, psd = report_card[ut][s]['psd-data'] #current_psd_dict[ut][s]['pos'][0], current_psd_dict[ut][s]['pos'][1]  
                    ax.loglog(f, 1e12 * psd, color=col, linestyle='-',label=s)
                    ax.loglog(f, 1e12 * np.cumsum( psd[::-1])[::-1] * np.diff(f)[1], label=s, color=col, linestyle=':')


            else:

                # usr_which_psd_plot should be a key for current_psd_dict[ut]
                #try:
                f, psd = report_card[ut][usr_which_psd_plot]['psd-data'] #current_psd_dict[ut][usr_which_psd_plot]['pos'][0], current_psd_dict[ut][usr_which_psd_plot]['pos'][1]
                #except:
                #    raise TypeError('usr_which_psd_plot entry does not in current_psd_dict[ut]')
   
                ax.loglog(f, 1e12 * psd, color='k', linestyle='-',label=usr_which_psd_plot, )
                ax.loglog(f, 1e12 * np.cumsum( psd[::-1])[::-1] * np.diff(f)[1], color='grey', linestyle=':',alpha=0.7)
                
                if usr_display_ref: 
                    f_ref_tmp, psd_ref_df_tmp = np.array(list( current_ref_psds[ut][usr_which_psd_plot].index)) , current_ref_psds[ut][usr_which_psd_plot] 
                    ax.fill_between(f_ref_tmp, psd_ref_df_tmp['q10'], psd_ref_df_tmp['q90'],\
                                    color='green',alpha=0.5,label=f'ref q10-q90')
                    #ax.loglog(current_ref_psds[ut][usr_which_psd_plot].index, current_ref_psds[ut][usr_which_psd_plot]['q10'] , color='green')

                if usr_display_vib:

                    med_floor = pd.Series(psd).rolling(vib_det_window, center=True).median()
                    detection_level = vib_det_thresh_factor * med_floor
                    
                    vib_df = report_card[ut][usr_which_psd_plot]['psd-vib_detection_df'] 
                    
                    # f, psd = report_card[ut][s]['psd-data']
                    #f_ref_tmp = np.array(list( current_ref_psds[ut][usr_which_psd_plot].index)) 
                
                    # very important that psd has same freq index as was used to generate vib_freqs in report card (which enforces this via interpolation)... best to attache original series to report card 


                    #contour height to indicate prominence 
                    #contour_heights = psd[peaks[0]] - peak_proms[0]


                    #np.argminfor f_peak in vib_df['vib_freqs']:
                    _, f_indx, _ = np.intersect1d(f, vib_df['vib_freqs'], return_indices=True)
                    ax.loglog(f[f_indx], 1e12 * psd[f_indx], 'x', color='r', label = 'vib peaks')
                    ax.semilogy(f, 1e12 * med_floor,linestyle=':', color='k' )
                    ax.semilogy(f,  1e12 * detection_level, linestyle=':', color='orange' , label='detection thresh.')
                    ax.vlines(x=f[f_indx], ymin= 1e12 * med_floor[f_indx], ymax= 1e12 * psd[f_indx], color='k')


            ax.grid()                                                                                              
            ax.set_ylabel(r'OPL PSD [$\mu m^2/Hz$]'+'\n'+r'reverse cumulative [$\mu m^2$]',fontsize=15)            
            ax.legend(fontsize=15)                                                                                 
            ax.tick_params(labelsize=15)
            ax.set_ylim(1e-10,1e4)
            ax.set_xlabel(r'frequency [Hz]',fontsize=15)                                                           
            ax.set_title(f'UT{ut}\n{filen} @ {current_time_key}UT')     

        else:
            print(f'UT{ut} not in current_psd_dict.. cannot plot PSDs')

    #current_psd_dict, current_ref_psds
    st.pyplot(fig2) 
    
    if st.button('download PSD figure'):
        fig2.savefig(os.path.join(local_figure_path, f'streamlit_psd_{now.strftime("%d-%m-%YT%H:%M:%S")}_vibdet-{usr_display_vib}_withRef-{usr_display_ref}_{usr_which_psd_plot}_fig.png'))

    usr_display_vib_detection_table = st.checkbox('hide detected vibration tables')

    @st.cache_data
    def display_detected_vibrations_df(report_card,usr_which_psd_plot):

        detect_df_2_display={}
        for ut in [1,2,3,4]:

            detect_df_2_display[ut] = pd.DataFrame(report_card[ut][usr_which_psd_plot]['psd-vib_detection_df'])[['vib_freqs', 'fwhm', 'rel_contr', 'abs_contr']]
            detect_df_2_display[ut][[ 'rel_contr', 'abs_contr']] *= 1e18 # convert from m^2 to nm^2 rms 
            detect_df_2_display[ut][[ 'rel_contr', 'abs_contr']] **= 0.5  #convert from nm^2 to nm rms 
            detect_df_2_display[ut].columns = ['frequency [Hz]', 'FWHM [Hz]', 'RMS contiumn-peak continuum [nm]', 'RMS absolute contribution [nm]'] # make sure order matches the order we read them in e.g. ['vib_freqs', 'fwhm', 'rel_contr', 'abs_contr']
            
        return(detect_df_2_display)


    if (not usr_display_vib_detection_table) and (usr_which_psd_plot not in ['all mirrors', 'combined geometries']) :

        latest_ts_title = f'<p style="font-family:sans-serif; color:Green; font-size: 42px;">DETECTED VIBRATIONS FOR {usr_which_psd_plot}</p>'
        st.markdown(latest_ts_title, unsafe_allow_html=True)



        st.markdown(f'DETECTED VIBRATIONS FOR {usr_which_psd_plot}')
        detect_df_2_display = display_detected_vibrations_df(report_card,usr_which_psd_plot)

        for ut in [1,2,3,4]:
            st.markdown(f'UT{ut}')
            st.dataframe( detect_df_2_display[ut] )

            if st.button(f'download vibration table for UT{ut}'):
                detect_df_2_display[ut].to_csv(os.path.join( local_vibdetection_path , f'streamlit_vibTable_UT{ut}_{usr_which_psd_plot}_{now.strftime("%d-%m-%YT%H:%M:%S")}.csv') )

                #usr_which_frequency = st.selectbox('which frequencies to plot history?', options= tuple(list(report_card[usr_which_ut_4_vib][usr_which_psd_plot]['psd-vib_detection_df']['vib_freqs'])), index=0)                                           
                #usr_which_frequency = st.selectbox('which frequencies to plot history?', options= tuple(list(report_card[usr_which_ut_4_vib][usr_which_psd_plot]['psd-vib_detection_df']['vib_freqs'])), index=0)  

    
    if st.button(f'download all figures and tables'):

        for ut in [1,2,3,4]:
            
            if (not usr_display_vib_detection_table) and (usr_which_psd_plot not in ['all mirrors', 'combined geometries']) :
                detect_df_2_display[ut].to_csv(os.path.join( local_vibdetection_path , f'streamlit_vibTable_UT{ut}_{usr_which_psd_plot}_{now.strftime("%d-%m-%YT%H:%M:%S")}.csv') )

        fig1.savefig(os.path.join(local_figure_path, f'streamlit_ts_{now.strftime("%d-%m-%YT%H:%M:%S")}_{which_ts_plot}_fig.png'))
            
        fig2.savefig(os.path.join(local_figure_path, f'streamlit_psd_{now.strftime("%d-%m-%YT%H:%M:%S")}_vibdet-{usr_display_vib}_withRef-{usr_display_ref}_{usr_which_psd_plot}_fig.png'))




with tabs[1]:

    # We should copy what Vicente did here : http://datalab.pl.eso.org:8866/voila/render/datalake/apps/sao_perfmon/Vibrations.ipynb
    

    #st.header("A dog")
    #st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
    date2look = st.date_input("What date do you want to analyse?" ,datetime.date(2023, 1, 6))

    # we specify the year and month specifically to feed to a cached function 
    year2look = date2look.year
    month2look = date2look.month

    #time2look = st.time_input('at what UT time do you want to analyse (we will look for the nearest sample)?', datetime.time(8, 45))

    which_UT = st.selectbox('what unit telescope (UT) do you want to look at?', ('UT1', 'UT2','UT3','UT4'))


    if post_upgrade:
        which_sensor =  st.selectbox('what sensor or combined geometry do you want to look at?', tuple(['m1-3','m4-7','m1-7'] + [f'm{i}' for i in range(1,8)] + [f'sensor {i+1} ({m})' for i, m in enumerate(mirror_lab )]) )
    else: # pre upgrade 
        which_sensor =  st.selectbox('what sensor or combined geometry do you want to look at?', tuple(['m1-3'] + [f'm{i}' for i in range(1,4)] + [f'sensor {i+1} ({m})' for i, m in enumerate(mirror_lab[:7])]))

    if 'sensor' in which_sensor: # we change the variable name to make compatiple with file naming convention 
        sensor2look = f'{which_sensor.split()[0]}_{which_sensor.split()[1]}'
    else:
        sensor2look = which_sensor

    plot_what = st.selectbox('what should we plot', ('position PSD', 'acceleration PSD'))

    @st.cache_data
    def read_usr_defined_data_for_analysis(year2look,month2look, which_UT, sensor2look ): 
        data2look_tmp = pd.read_csv(f'{local_data_path}/bcb/{year2look}/{"%02d" % (month2look,)}/{which_UT}/{sensor2look}_psds_{which_UT}_{year2look}-{"%02d" % (month2look,)}.csv',index_col=[0])
        return(data2look_tmp)
    
    data2look = read_usr_defined_data_for_analysis(year2look, month2look,  which_UT, sensor2look )

    available_days_in_data = list( np.unique( [ float(d_tmp.split('-')[2].split()[0])  for d_tmp in data2look.index]) ) 
    day_max, day_min = int(max( available_days_in_data )), int( min( available_days_in_data ) )
    #available_days_in_data = np.array([ datetime.datetime.strptime(d_tmp, '%Y-%m-%d %H:%M:%S').day for d_tmp in data2look.index])  
    
    # time and day sliders 
    print( available_days_in_data )
    day2look = st.slider("Change what day to look at?", min_value = day_min, max_value=day_max, value=date2look.day )
    time2look = st.slider("Change what time to look at?", value=datetime.time(11, 30))
    
    usr_time  =  datetime.datetime.strptime(f'{year2look}-{"%02d" % (month2look,)}-{"%02d" % (day2look,)} {time2look.hour}:{time2look.minute}:00', '%Y-%m-%d %H:%M:%S')
    #usr_time = datetime.datetime.combine( date2look, time2look ) # merge the date and time 


    # get the date index nearest to input time                                                                           
    data_key = data2look.index[np.argmin([abs(usr_time - datetime.datetime.strptime(data2look.index[i], '%Y-%m-%d %H:%M:%S') ) for i in range(len(data2look)) ]) ]

    #st.header(f'{time2look}, {usr_time}, {data_key}, {data2look.index[2]},{data2look.index[10]}')
    # plot 
    fig3,ax3 = plt.subplots()
    f = np.array(list(data2look.loc[data_key].index.astype(float)))
    psd = data2look.loc[data_key].values

    if plot_what=='position PSD':
        f,psd = mn2.double_integrate(f,psd)
        if 'sensor' in sensor2look:
            ax3.loglog(f,psd,label=f'{data_key}')
            #ax3.loglog(f, np.cumsum( psd[::-1])[::-1] * np.diff(f)[1], label=f'{date_key}', color='grey', linestyle=':')
            ax3.set_ylabel(r'PSD [$V^2/Hz$]',fontsize=15)
            
        else:
            ax3.loglog(f,1e12 * psd,label=f'{data_key}')
            #ax3.loglog(f, np.cumsum(1e12 * psd[::-1])[::-1] * np.diff(f)[1], label=f'{date_key}', color='grey', linestyle=':')
            ax3.set_ylabel(r'PSD [$\mu m^2/Hz$]',fontsize=15)

    elif plot_what=='acceleration PSD':
        if 'sensor' in sensor2look:
            ax3.loglog(f,psd,label=f'{data_key}')
            #ax3.loglog(f, np.cumsum( psd[::-1])[::-1] * np.diff(f)[1], label=f'{date_key}', color='grey', linestyle=':')
            ax3.set_ylabel(r'PSD [$V^2/s^4/Hz$]',fontsize=15)
            
        else:
            ax3.loglog(f, 1e12 * psd,label=f'{data_key}')
            #ax3.loglog(f, np.cumsum(1e12 * psd[::-1])[::-1] * np.diff(f)[1], label=f'{date_key}', color='grey', linestyle=':')
            ax3.set_ylabel(r'PSD [$\mu m^2/s^4/Hz$]',fontsize=15)
    else:
        print('case not made')

    ax3.set_xlabel(r'frequency [Hz]',fontsize=15)
    ax3.legend( fontsize=15)
    ax3.tick_params(labelsize=15)
    ax3.grid()
    st.pyplot(fig3)







with tabs[2]:
   
    
    vib_inventory_UT1_dict = {'frequency (Hz)':[47,200,73 ],\
                          'origin':['fans','MACAO',''],\
                              'related PR':['-','-','https://wits.pl.eso.org/browse/PR-173294']}  
    vib_inventory_UT2_dict = {'frequency (Hz)':[47,200,73],\
                          'origin':['fans','MACAO',''],\
                              'related PR':['-','-','https://wits.pl.eso.org/browse/PR-173294']}


    vib_inventory_UT3_dict = {'frequency (Hz)':[47,200,73],\
                          'origin':['fans','MACAO',''],\
                              'related PR':['-','-','https://wits.pl.eso.org/browse/PR-173294']}


    vib_inventory_UT4_dict = {'frequency (Hz)':[47,200, 73],\
                          'origin':['fans','MACAO',''],\
                              'related PR':['-','-','https://wits.pl.eso.org/browse/PR-173294']}
      

    st.markdown('UT1')
    st.dataframe( pd.DataFrame( vib_inventory_UT1_dict ) ) 

    st.markdown('UT2')
    st.dataframe( pd.DataFrame( vib_inventory_UT2_dict ) )
    
    st.markdown('UT3')
    st.dataframe( pd.DataFrame( vib_inventory_UT3_dict ) )
    
    st.markdown('UT4')
    st.dataframe( pd.DataFrame( vib_inventory_UT4_dict ) )

css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:2rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

#UT{ut}/sensor_{sensor}_psds_UT{ut}_{year}-{month}.csv -P local_data_path')

#ldlvib1_psd_2023-05-01.hdf5 


