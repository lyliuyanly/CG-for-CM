# -*- coding: utf-8 -*-
"""
Created on Tues Jan 16 19:16:10 2022
@author: LY
"""
'''
@File    :   Mixed_Swissdata_test.py
'''
import math
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
#from sklearn import preprocessing
import time
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy.stats import qmc
import matplotlib.pyplot as plt

mm = MinMaxScaler()
ss = StandardScaler()
t0=time.time()
# In[2]:
inputLocation =""
# Load input data
data = pd.read_csv(inputLocation+'apollo_swissRouteChoiceData.csv').iloc[:3492,:]

attributes=['tt1','tc1','hw1','ch1','tt2','tc2','hw2','ch2']
x_train= np.array(data[attributes].values, dtype=np.float64)
x_train_alt1=np.array(data[attributes[0:4]].values, dtype=np.float64)
x_train_alt2=np.array(data[attributes[4:8]].values, dtype=np.float64)
x_train_alt1=x_train_alt1-x_train_alt2
x_train_alt2=x_train_alt2-x_train_alt2
attributes_num=len(attributes)

# Availability variable
avail_list = {'av_1': data['av1'], 
              'av_2': data['av2']}

avail_val = np.transpose(list(avail_list.values()))

avails=['av1','av2']
M_avails=np.array(data[avails].values, dtype=np.float64)


# Endogenous variable
choice_list = ['choice1', 'choice2']
for i in range(len(choice_list)):     
    data[choice_list[i]] = np.where(data['choice']==i+1,1,0) 
    
# One-hot encode
y_train = np.array(data[choice_list].values.reshape(len(data[choice_list]), len(choice_list)), dtype=np.float64)
y_train1 = np.zeros((len(data[choice_list]), 2), dtype=np.float64)
y_train1[:, 0] = 1.0
y_train2 = np.zeros((len(data[choice_list]), 2), dtype=np.float64)
y_train2[:, 1] = 1.0

# Define Parameters
# Initialize unknown parameters
pi_set= [-1.484029048679663, -0.1551809237036665, -0.3479483774127985, 1.759510906278205, 1.258726378727356,
          -0.2542044877272236, -1.477775040484163, -1.347479743601841, -0.2187777565653435, -1.382340295362344]

msc1 = tf.Variable(pi_set[0]*tf.ones([1,], dtype=tf.float64), dtype=tf.float64)
msc2 = tf.constant(1, shape=(1,),  dtype=tf.float64) # baseline coefficients

beta_tt_a=tf.Variable(pi_set[1]*tf.ones([1,], dtype=tf.float64), dtype=tf.float64)
beta_tt_b=tf.Variable(pi_set[2]*tf.ones([1,], dtype=tf.float64), dtype=tf.float64)
beta_tc_a=tf.Variable(pi_set[3]*tf.ones([1,], dtype=tf.float64), dtype=tf.float64)
beta_tc_b=tf.Variable(pi_set[4]*tf.ones([1,], dtype=tf.float64), dtype=tf.float64)
beta_hw_a=tf.Variable(pi_set[5]*tf.ones([1,], dtype=tf.float64), dtype=tf.float64)
beta_hw_b=tf.Variable(pi_set[6]*tf.ones([1,], dtype=tf.float64), dtype=tf.float64)
beta_ch_a=tf.Variable(pi_set[7]*tf.ones([1,], dtype=tf.float64), dtype=tf.float64)
beta_ch_b=tf.Variable(pi_set[8]*tf.ones([1,], dtype=tf.float64), dtype=tf.float64)

beta_ref=tf.constant(1, shape=(1,),  dtype=tf.float64) # baseline coefficients

delta_a=tf.Variable(pi_set[9]*tf.ones([1,], dtype=tf.float64), dtype=tf.float64)
delta_b=tf.constant(1, shape=(1,),  dtype=tf.float64) 

params=[msc1,beta_tt_a,beta_tt_b,beta_tc_a,beta_tc_b,
      beta_hw_a,beta_hw_b,beta_ch_a,beta_ch_b,delta_a]
 
params_ini = [pi_set[i] * ref for i, ref in enumerate([msc2,beta_ref,beta_ref,beta_ref,beta_ref,
                                                       beta_ref,beta_ref,beta_ref,beta_ref,delta_b])]

'''halton sample defination'''
number_choices=9 
number_observations=x_train.shape[0]
number_person=int(number_observations/number_choices)

def model_fun(x_train, y_train, params,individual=None):
    
    x_train_alt1=np.array(x_train[:,0:4], dtype=np.float64)
    x_train_alt2=np.array(x_train[:,4:8], dtype=np.float64)
    x_train_alt1=x_train_alt1-x_train_alt2
    x_train_alt2=x_train_alt2-x_train_alt2
    
    v1_latenta = params[0]+params[1]*x_train_alt1[:,0]+params[3]*x_train_alt1[:,1]+params[5]*x_train_alt1[:,2]+params[7]*x_train_alt1[:,3]    
    v1_latenta_max = tf.ones_like(v1_latenta) * 700
    v1_latenta_min = tf.ones_like(v1_latenta) *(-700)
    v1_latenta= tf.where(v1_latenta<-700, v1_latenta_min, v1_latenta)
    v1_latenta= tf.where(v1_latenta>700, v1_latenta_max, v1_latenta)
    
    v2_latenta = 0.0+params[1]*x_train_alt2[:,0]+params[3]*x_train_alt2[:,1]+params[5]*x_train_alt2[:,2]+params[7]*x_train_alt2[:,3] 
    #tf.print("Shape of v1_latenta:", tf.shape(v1_latenta), summarize=-1)               
    exp_v1_latenta=tf.exp(v1_latenta)#*M_avails[:,0].reshape(-1, 1)
    exp_v2_latenta=tf.exp(v2_latenta)#*M_avails[:,1].reshape(-1, 1)
    exp_sum_latenta=exp_v1_latenta+exp_v2_latenta
    #tf.print("Shape of exp_sum:", tf.shape(exp_sum), summarize=-1) 
    exp_sum_log_latenta=tf.math.log(exp_sum_latenta)
    exp_fenzi_latenta=exp_v1_latenta*y_train[:,0]+exp_v2_latenta*y_train[:,1]
    #tf.print("Shape of exp_fenzi:", tf.shape(exp_fenzi), summarize=-1) 
    exp_fenzi_log_latenta=tf.math.log(exp_fenzi_latenta)
    P_latenta=tf.exp(exp_fenzi_log_latenta-exp_sum_log_latenta)   
    #tf.print("Shape of P_latenta:", tf.shape(P), summarize=-1) 
    '''probability'''
    P_panel_latenta = tf.reshape(P_latenta, [-1, number_choices])    
    P_panel_prod_latenta = tf.reduce_prod(P_panel_latenta, axis=1) 
    #tf.print("P_panel_prod_latenta:", P_panel_prod_latenta)
    
    v1_latentb = params[0]+params[2]*x_train_alt1[:,0]+params[4]*x_train_alt1[:,1]+params[6]*x_train_alt1[:,2]+params[8]*x_train_alt1[:,3]              
    v1_latentb= tf.where(v1_latentb<-700, v1_latenta_min, v1_latentb)
    v1_latentb= tf.where(v1_latentb>700, v1_latenta_max, v1_latentb)
    
    v2_latentb = 0.0+params[2]*x_train_alt2[:,0]+params[4]*x_train_alt2[:,1]+params[6]*x_train_alt2[:,2]+params[8]*x_train_alt2[:,3]              
    exp_v1_latentb=tf.exp(v1_latentb)#*M_avails[:,0].reshape(-1, 1)
    exp_v2_latentb=tf.exp(v2_latentb)#*M_avails[:,1].reshape(-1, 1)
    exp_sum_latentb=exp_v1_latentb+exp_v2_latentb
    exp_sum_log_latentb=tf.math.log(exp_sum_latentb)
    exp_fenzi_latentb=exp_v1_latentb*y_train[:,0]+exp_v2_latentb*y_train[:,1] 
    exp_fenzi_log_latentb=tf.math.log(exp_fenzi_latentb)
    P_latentb=tf.exp(exp_fenzi_log_latentb-exp_sum_log_latentb)      
    '''probability'''
    P_panel_latentb = tf.reshape(P_latentb, [-1, number_choices])
    P_panel_prod_latentb = tf.reduce_prod(P_panel_latentb, axis=1) 
    #tf.print("P_panel_prod_latentb:", P_panel_prod_latentb)

    v_latenta=params[9]
    v_latentb=0.0
    exp_latenta=tf.exp(v_latenta)
    exp_latentb=tf.cast(tf.exp(v_latentb), tf.float64)#tf.exp(v_latentb)
    exp_sum_latent=exp_latenta+exp_latentb
    P_latenta=exp_latenta/exp_sum_latent
    P_latentb=exp_latentb/exp_sum_latent
    #tf.print("P_latenta:", P_latenta)
    #tf.print("P_latentb:", P_latentb)
   
    P_panel_prod=P_panel_prod_latenta*P_latenta+P_panel_prod_latentb*P_latentb
    # tf.print("Shape of P_panel_prod:", tf.shape(P_panel_prod), summarize=-1)    
    # tf.print("P_panel_prod:", P_panel_prod)    
    
    return P_panel_prod


x1=np.concatenate((np.ones([len(data[choice_list]),1]),np.zeros([len(data[choice_list]),1])), axis=1)

def model_fun_analytical_mean(x_train, y_train,params, individual=None):
    dll=1
    
    x_train_alt1=np.array(x_train[:,0:4], dtype=np.float64)
    x_train_alt2=np.array(x_train[:,4:8], dtype=np.float64)
    x_train_alt1=x_train_alt1-x_train_alt2
    x_train_alt2=x_train_alt2-x_train_alt2
    x_diff = x_train_alt1 - x_train_alt2
    
    ones_column = tf.ones([tf.shape(x_diff)[0], 1], dtype=x_diff.dtype)
   
    x_diff_extended = tf.concat([x_diff, ones_column], axis=1)
    # tf.print("Shape of x_diff_extended:", tf.shape(x_diff_extended), x_diff_extended)
    
    v1_latenta = params[0]+params[1]*x_train_alt1[:,0]+params[3]*x_train_alt1[:,1]+params[5]*x_train_alt1[:,2]+params[7]*x_train_alt1[:,3]    
    v1_latenta_max = tf.ones_like(v1_latenta) * 700
    v1_latenta_min = tf.ones_like(v1_latenta) *(-700)
    v1_latenta= tf.where(v1_latenta<-700, v1_latenta_min, v1_latenta)
    v1_latenta= tf.where(v1_latenta>700, v1_latenta_max, v1_latenta)
    
    v2_latenta = 0.0+params[1]*x_train_alt2[:,0]+params[3]*x_train_alt2[:,1]+params[5]*x_train_alt2[:,2]+params[7]*x_train_alt2[:,3] 
    #tf.print("Shape of v1_latenta:", tf.shape(v1_latenta), summarize=-1)               
    exp_v1_latenta=tf.exp(v1_latenta)#*M_avails[:,0].reshape(-1, 1)
    exp_v2_latenta=tf.exp(v2_latenta)#*M_avails[:,1].reshape(-1, 1)
    exp_sum_latenta=exp_v1_latenta+exp_v2_latenta
    #tf.print("Shape of exp_sum:", tf.shape(exp_sum), summarize=-1) 
    exp_sum_log_latenta=tf.math.log(exp_sum_latenta)
    exp_fenzi_latenta=exp_v1_latenta*y_train[:,0]+exp_v2_latenta*y_train[:,1]

    #tf.print("Shape of exp_fenzi:", tf.shape(exp_fenzi), summarize=-1) 
    exp_fenzi_log_latenta=tf.math.log(exp_fenzi_latenta)
    P_latenta_all=tf.exp(exp_fenzi_log_latenta-exp_sum_log_latenta)   
    #tf.print("Shape of P_latenta:", tf.shape(P), summarize=-1) 
    '''probability'''
    P_panel_latenta = tf.reshape(P_latenta_all, [-1, number_choices])    
    P_panel_prod_latenta = tf.reduce_prod(P_panel_latenta, axis=1) 
    #tf.print("P_panel_prod_latenta:", P_panel_prod_latenta)
    
    v1_latentb = params[0]+params[2]*x_train_alt1[:,0]+params[4]*x_train_alt1[:,1]+params[6]*x_train_alt1[:,2]+params[8]*x_train_alt1[:,3]              
    v1_latentb= tf.where(v1_latentb<-700, v1_latenta_min, v1_latentb)
    v1_latentb= tf.where(v1_latentb>700, v1_latenta_max, v1_latentb)
    
    v2_latentb = 0.0+params[2]*x_train_alt2[:,0]+params[4]*x_train_alt2[:,1]+params[6]*x_train_alt2[:,2]+params[8]*x_train_alt2[:,3]              
    exp_v1_latentb=tf.exp(v1_latentb)#*M_avails[:,0].reshape(-1, 1)
    exp_v2_latentb=tf.exp(v2_latentb)#*M_avails[:,1].reshape(-1, 1)
    exp_sum_latentb=exp_v1_latentb+exp_v2_latentb
    exp_sum_log_latentb=tf.math.log(exp_sum_latentb)
    exp_fenzi_latentb=exp_v1_latentb*y_train[:,0]+exp_v2_latentb*y_train[:,1] 
    
    exp_fenzi_log_latentb=tf.math.log(exp_fenzi_latentb)
    P_latentb_all=tf.exp(exp_fenzi_log_latentb-exp_sum_log_latentb)      
    '''probability'''
    P_panel_latentb = tf.reshape(P_latentb_all, [-1, number_choices])
    P_panel_prod_latentb = tf.reduce_prod(P_panel_latentb, axis=1) 
    # tf.print("P_panel_prod_latentb:", P_panel_prod_latentb)
    
    v_latenta=params[9]
    v_latentb=0.0
    exp_latenta=tf.exp(v_latenta)
    exp_latentb=tf.cast(tf.exp(v_latentb), tf.float64)#tf.exp(v_latentb)
    exp_sum_latent=exp_latenta+exp_latentb
    P_latenta=exp_latenta/exp_sum_latent
    P_latentb=exp_latentb/exp_sum_latent
    #tf.print("P_latenta:", P_latenta)
    #tf.print("P_latentb:", P_latentb)
   
    P_panel_prod=P_panel_prod_latenta*P_latenta+P_panel_prod_latentb*P_latentb
    # tf.print("Shape of P_panel_prod:", tf.shape(P_panel_prod), summarize=-1)    
    # tf.print("P_panel_prod:", P_panel_prod)    
    
    '''dll_beta_a'''
    exp_v1_latenta_2d = tf.expand_dims(exp_v1_latenta, axis=1)  
    exp_v2_latenta_2d = tf.expand_dims(exp_v2_latenta, axis=1)  

    numerator_a = exp_v1_latenta_2d * exp_v2_latenta_2d * x_diff_extended
    # tf.print("Shape of numerator_a:", tf.shape(numerator_a), numerator_a)
    
    P_latenta_derivative = (y_train[:,0] - y_train[:,1])[:, tf.newaxis] * numerator_a / (exp_sum_latenta[:, tf.newaxis] ** 2)
    # tf.print("Shape of P_latenta_derivative:", tf.shape(P_latenta_derivative), P_latenta_derivative)
    
    P_latenta_d_o = tf.reshape(P_latenta_derivative / P_latenta_all[:, tf.newaxis], [-1, number_choices, 5]) * tf.reshape(P_panel_prod_latenta, [-1, 1, 1])
    # tf.print("Shape of P_latenta_d_o:", tf.shape(P_latenta_d_o), P_latenta_d_o)
    
    P_latenta_person = tf.reduce_sum(P_latenta_d_o, axis=1)
    dll_beta_a = -tf.reduce_sum((P_latenta[:, tf.newaxis] * P_latenta_person / P_panel_prod[:, tf.newaxis]), axis=0) / 388
    # tf.print("Shape of dll_beta_a:", tf.shape(dll_beta_a), dll_beta_a)

    '''dll_beta_b'''
    exp_v1_latentb_2d = tf.expand_dims(exp_v1_latentb, axis=1) 
    exp_v2_latentb_2d = tf.expand_dims(exp_v2_latentb, axis=1) 

    numerator_b = exp_v1_latentb_2d * exp_v2_latentb_2d * x_diff_extended
    # tf.print("Shape of numerator_b:", tf.shape(numerator_b), numerator_b)
  
    P_latentb_derivative = (y_train[:,0] - y_train[:,1])[:, tf.newaxis] * numerator_b / (exp_sum_latentb[:, tf.newaxis] ** 2)
    # tf.print("Shape of P_latentb_derivative:", tf.shape(P_latentb_derivative), P_latentb_derivative)
    
    P_latentb_d_o = tf.reshape(P_latentb_derivative / P_latentb_all[:, tf.newaxis], [-1, number_choices, 5]) * tf.reshape(P_panel_prod_latentb, [-1, 1, 1])
    # tf.print("Shape of P_latentb_d_o:", tf.shape(P_latentb_d_o), P_latentb_d_o)
    
    P_latentb_person = tf.reduce_sum(P_latentb_d_o, axis=1)
    dll_beta_b = -tf.reduce_sum((P_latentb[:, tf.newaxis] * P_latentb_person / P_panel_prod[:, tf.newaxis]), axis=0) / 388
    # tf.print("Shape of dll_beta_a:", tf.shape(dll_beta_a), dll_beta_a)

    '''dll_msc1''' 
    dll_msc1  =-tf.reduce_sum(((P_latenta[:, tf.newaxis] * P_latenta_person+P_latentb[:, tf.newaxis] * P_latentb_person) / P_panel_prod[:, tf.newaxis]), axis=0) / 388
    # dll_beta_a=tf.reduce_sum((P_latenta/P_panel_prod),axis=0)/388
    # tf.print("Shape of dll_beta_a:", tf.shape(dll_beta_a),dll_beta_a) 
     
    dll_wa1 =tf.reduce_sum((P_panel_prod_latenta*(exp_latenta*exp_sum_latent-exp_latenta*exp_latenta)/tf.square(exp_sum_latent,2))/P_panel_prod,axis=0)/388
    dll_wa2 =tf.reduce_sum((P_panel_prod_latentb*(0*exp_sum_latent-exp_latentb*exp_latenta)/tf.square(exp_sum_latent,2))/P_panel_prod,axis=0)/388
    dll_wa=-1*(dll_wa1+dll_wa2)
    dll_wa=-tf.reduce_sum((P_latenta*P_latentb*(P_panel_prod_latenta-P_panel_prod_latentb))/P_panel_prod,axis=0)/388
    # tf.print("Shape of dll_wa :", tf.shape(dll_wa),dll_wa ) 

    #dll=[dll_msc1,dll_beta_a,dll_beta_b,dll_wa]
    dll=[dll_msc1[4],dll_beta_a[0],dll_beta_b[0],dll_beta_a[1],dll_beta_b[1],dll_beta_a[2],dll_beta_b[2],dll_beta_a[3],dll_beta_b[3],dll_wa]
    # #tf.print(dll_dmsc2)
    return dll

def model_fun_analytical(x_train, y_train,params, individual=None):
    dll=1
    
    x_train_alt1=np.array(x_train[:,0:4], dtype=np.float64)
    x_train_alt2=np.array(x_train[:,4:8], dtype=np.float64)
    x_train_alt1=x_train_alt1-x_train_alt2
    x_train_alt2=x_train_alt2-x_train_alt2
    x_diff = x_train_alt1 - x_train_alt2
    
    ones_column = tf.ones([tf.shape(x_diff)[0], 1], dtype=x_diff.dtype)
    
    x_diff_extended = tf.concat([x_diff, ones_column], axis=1)
    # tf.print("Shape of x_diff_extended:", tf.shape(x_diff_extended), x_diff_extended)
    
    v1_latenta = params[0]+params[1]*x_train_alt1[:,0]+params[3]*x_train_alt1[:,1]+params[5]*x_train_alt1[:,2]+params[7]*x_train_alt1[:,3]    
    v1_latenta_max = tf.ones_like(v1_latenta) * 700
    v1_latenta_min = tf.ones_like(v1_latenta) *(-700)
    v1_latenta= tf.where(v1_latenta<-700, v1_latenta_min, v1_latenta)
    v1_latenta= tf.where(v1_latenta>700, v1_latenta_max, v1_latenta)
    
    v2_latenta = 0.0+params[1]*x_train_alt2[:,0]+params[3]*x_train_alt2[:,1]+params[5]*x_train_alt2[:,2]+params[7]*x_train_alt2[:,3] 
    #tf.print("Shape of v1_latenta:", tf.shape(v1_latenta), summarize=-1)               
    exp_v1_latenta=tf.exp(v1_latenta)#*M_avails[:,0].reshape(-1, 1)
    exp_v2_latenta=tf.exp(v2_latenta)#*M_avails[:,1].reshape(-1, 1)
    exp_sum_latenta=exp_v1_latenta+exp_v2_latenta
    #tf.print("Shape of exp_sum:", tf.shape(exp_sum), summarize=-1) 
    exp_sum_log_latenta=tf.math.log(exp_sum_latenta)
    exp_fenzi_latenta=exp_v1_latenta*y_train[:,0]+exp_v2_latenta*y_train[:,1]
    
    #tf.print("Shape of exp_fenzi:", tf.shape(exp_fenzi), summarize=-1) 
    exp_fenzi_log_latenta=tf.math.log(exp_fenzi_latenta)
    P_latenta_all=tf.exp(exp_fenzi_log_latenta-exp_sum_log_latenta)   
    #tf.print("Shape of P_latenta:", tf.shape(P), summarize=-1) 
    '''probability'''
    P_panel_latenta = tf.reshape(P_latenta_all, [-1, number_choices])    
    P_panel_prod_latenta = tf.reduce_prod(P_panel_latenta, axis=1) 
    #tf.print("P_panel_prod_latenta:", P_panel_prod_latenta)

    v1_latentb = params[0]+params[2]*x_train_alt1[:,0]+params[4]*x_train_alt1[:,1]+params[6]*x_train_alt1[:,2]+params[8]*x_train_alt1[:,3]              
    v1_latentb= tf.where(v1_latentb<-700, v1_latenta_min, v1_latentb)
    v1_latentb= tf.where(v1_latentb>700, v1_latenta_max, v1_latentb)
    
    v2_latentb = 0.0+params[2]*x_train_alt2[:,0]+params[4]*x_train_alt2[:,1]+params[6]*x_train_alt2[:,2]+params[8]*x_train_alt2[:,3]              
    exp_v1_latentb=tf.exp(v1_latentb)#*M_avails[:,0].reshape(-1, 1)
    exp_v2_latentb=tf.exp(v2_latentb)#*M_avails[:,1].reshape(-1, 1)
    exp_sum_latentb=exp_v1_latentb+exp_v2_latentb
    exp_sum_log_latentb=tf.math.log(exp_sum_latentb)
    exp_fenzi_latentb=exp_v1_latentb*y_train[:,0]+exp_v2_latentb*y_train[:,1] 
    
    exp_fenzi_log_latentb=tf.math.log(exp_fenzi_latentb)
    P_latentb_all=tf.exp(exp_fenzi_log_latentb-exp_sum_log_latentb)      
    '''probability'''
    P_panel_latentb = tf.reshape(P_latentb_all, [-1, number_choices])
    P_panel_prod_latentb = tf.reduce_prod(P_panel_latentb, axis=1) 
    # tf.print("P_panel_prod_latentb:", P_panel_prod_latentb)
    
    v_latenta=params[9]
    v_latentb=0.0
    exp_latenta=tf.exp(v_latenta)
    exp_latentb=tf.cast(tf.exp(v_latentb), tf.float64)#tf.exp(v_latentb)
    exp_sum_latent=exp_latenta+exp_latentb
    P_latenta=exp_latenta/exp_sum_latent
    P_latentb=exp_latentb/exp_sum_latent
    #tf.print("P_latenta:", P_latenta)
    #tf.print("P_latentb:", P_latentb)
   
    P_panel_prod=P_panel_prod_latenta*P_latenta+P_panel_prod_latentb*P_latentb
    
    exp_v1_latenta_2d = tf.expand_dims(exp_v1_latenta, axis=1)  
    exp_v2_latenta_2d = tf.expand_dims(exp_v2_latenta, axis=1)  
    
    numerator_a = exp_v1_latenta_2d * exp_v2_latenta_2d * x_diff_extended
    # tf.print("Shape of numerator_a:", tf.shape(numerator_a), numerator_a)
    
    P_latenta_derivative = (y_train[:,0] - y_train[:,1])[:, tf.newaxis] * numerator_a / (exp_sum_latenta[:, tf.newaxis] ** 2)
    # tf.print("Shape of P_latenta_derivative:", tf.shape(P_latenta_derivative), P_latenta_derivative)
    
    P_latenta_d_o = tf.reshape(P_latenta_derivative / P_latenta_all[:, tf.newaxis], [-1, number_choices, 5]) * tf.reshape(P_panel_prod_latenta, [-1, 1, 1])
    # tf.print("Shape of P_latenta_d_o:", tf.shape(P_latenta_d_o), P_latenta_d_o)
    
    P_latenta_person = tf.reduce_sum(P_latenta_d_o, axis=1)
    dll_beta_a = -tf.reduce_sum((P_latenta[:, tf.newaxis] * P_latenta_person / P_panel_prod[:, tf.newaxis]), axis=0) 
    # tf.print("Shape of dll_beta_a:", tf.shape(dll_beta_a), dll_beta_a)

    exp_v1_latentb_2d = tf.expand_dims(exp_v1_latentb, axis=1)  
    exp_v2_latentb_2d = tf.expand_dims(exp_v2_latentb, axis=1)  

    numerator_b = exp_v1_latentb_2d * exp_v2_latentb_2d * x_diff_extended
    # tf.print("Shape of numerator_b:", tf.shape(numerator_b), numerator_b)

    P_latentb_derivative = (y_train[:,0] - y_train[:,1])[:, tf.newaxis] * numerator_b / (exp_sum_latentb[:, tf.newaxis] ** 2)
    # tf.print("Shape of P_latentb_derivative:", tf.shape(P_latentb_derivative), P_latentb_derivative)
    
    P_latentb_d_o = tf.reshape(P_latentb_derivative / P_latentb_all[:, tf.newaxis], [-1, number_choices, 5]) * tf.reshape(P_panel_prod_latentb, [-1, 1, 1])
    # tf.print("Shape of P_latentb_d_o:", tf.shape(P_latentb_d_o), P_latentb_d_o)
    
    P_latentb_person = tf.reduce_sum(P_latentb_d_o, axis=1)
    dll_beta_b = -tf.reduce_sum((P_latentb[:, tf.newaxis] * P_latentb_person / P_panel_prod[:, tf.newaxis]), axis=0) 
    # tf.print("Shape of dll_beta_a:", tf.shape(dll_beta_a), dll_beta_a)

    dll_msc1=-tf.reduce_sum(((P_latenta[:, tf.newaxis] * P_latenta_person+P_latentb[:, tf.newaxis] * P_latentb_person) / P_panel_prod[:, tf.newaxis]), axis=0) 
    dll_wa1=tf.reduce_sum((P_panel_prod_latenta*(exp_latenta*exp_sum_latent-exp_latenta*exp_latenta)/tf.square(exp_sum_latent,2))/P_panel_prod,axis=0)
    dll_wa2=tf.reduce_sum((P_panel_prod_latentb*(0*exp_sum_latent-exp_latentb*exp_latenta)/tf.square(exp_sum_latent,2))/P_panel_prod,axis=0)
    dll_wa=-1*(dll_wa1+dll_wa2)
    dll_wa=-tf.reduce_sum((P_latenta*P_latentb*(P_panel_prod_latenta-P_panel_prod_latentb))/P_panel_prod,axis=0)

    #dll=[dll_msc1,dll_beta_a,dll_beta_b,dll_wa]
    dll=[dll_msc1[4],dll_beta_a[0],dll_beta_b[0],dll_beta_a[1],dll_beta_b[1],dll_beta_a[2],dll_beta_b[2],dll_beta_a[3],dll_beta_b[3],dll_wa]

    dll_reshaped = [tf.reshape(item, shape=(1,)) for item in dll]
    # dll=[dll_msc1,dll_beta_a,dll_beta_b,dll_wa]
    return dll_reshaped
    
# multi-class cost_entropy
# @tf.function
def cost_fun(y_train,yhat):
    # tf.print(yhat)
    #ll=-tf.reduce_sum(tf.math.log(yhat+1e-8))
    '''ll=-tf.reduce_mean(tf.math.log(yhat), axis=0)'''
    ll=-tf.reduce_sum(tf.math.log(yhat), axis=0)
    # ll=-tf.reduce_mean(tf.math.log(yhat), axis=0)
    #tf.print('ll',ll*number_person)
    return  ll


def numerical_gradient(params,delta):
    #tf.keras.backend.clear_session()
    #tf.compat.v1.get_default_graph().clear()
    '''print numerical gradient'''
    num_params=len(params)
    # print(num_params)
    estimated_grad_numerical=[]
    for i in range(num_params):
        params[i]=params[i]+delta
        #tf.print('params-1',params[i])
        a= cost_fun(y_train,model_fun(x_train, y_train, params))
        #tf.print('a',a)
        params[i]=params[i]-2*delta
        #params[i].assign_sub(2*delta)
        #tf.print('params-1',params[i])
        b= cost_fun(y_train,model_fun(x_train, y_train, params))
        #tf.print('b',b)
        gradi=(a-b)/(2*delta)
        #tf.print('gradi',gradi) 
        #params[i].assign_add(delta)  # recover x[i]
        gradi=tf.reshape(gradi, shape=(1,))
        estimated_grad_numerical.append(gradi)
    #tf.print('numerical',estimated_grad_numerical) 
    
    return  estimated_grad_numerical

"""
To create the gradients of the log-likelihood with respect to parameters, 
we edited the code originally written by Pi-Yueh Chuang <pychuang@gwu.edu>
"""
# obtain the shapes of all trainable parameters in the model
def loss_gradient(params, x_train,y_train,individual=None):
    # x_train_expenditure[x_train_expenditure== 0] = 10
    # print(x_train_expenditure)
    shapes = tf.shape_n(params)
    n_tensors = len(shapes)
    delta=1e-6
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n

    @tf.function
    def assign_new_model_parameters(params_1d):

        pparams = tf.dynamic_partition(params_1d, part, n_tensors)

        for i, (shape, param) in enumerate(zip(shapes, pparams)):
            params[i].assign(tf.reshape(param, shape))

    #@tf.function
    def est_grad(params_1d):
        # Derive the Tensorflow gradient
        with tf.GradientTape() as tape:
            # Call the function to update and convert the shape of parameters
            assign_new_model_parameters(params_1d)
            # Estimated Choice Probability
            yhat= model_fun(x_train, y_train, params, individual)
            # Call the loss function
            loss_value = cost_fun(y_train,yhat)
        
        
        if gradient_type=='automatic':
            '''print automatic gradient'''
            # Calculate the gradient for each parameter
            estimated_grad_automatic = tape.gradient(loss_value, params)  
            tf.print('automatic',estimated_grad_automatic) 
            grads_1dim = tf.dynamic_stitch(idx, estimated_grad_automatic)
        
        if gradient_type=='analytical':
            '''print analytical gradient'''
            estimated_grad_analytical= model_fun_analytical(x_train, y_train, params, individual=None)     
            tf.print('analytical',estimated_grad_analytical)
            grads_1dim = tf.dynamic_stitch(idx, estimated_grad_analytical)
        

        if gradient_type=='numerical':
            '''print numerical gradient'''
            params_const = [tf.constant(param).numpy() for param in params]
            #tf.print('params const',params_const)
            estimated_grad_numerical=numerical_gradient(params_const,delta) 
            #print(estimated_grad_numerical)
            tf.print('numerical',estimated_grad_numerical) 
            grads_1dim = tf.dynamic_stitch(idx, estimated_grad_numerical)#estimated_grad_numerical#
        
        return loss_value, grads_1dim
    est_grad.idx = idx

    return est_grad


# gradient_type='automatic'
# gradient_type='analytical'
gradient_type='numerical'

# Define the positions of initial parameters
init_params = tf.dynamic_stitch(loss_gradient(params, x_train,y_train).idx, params)

#optimizer = tfp.optimizer.Adam(learning_rate=0.01)
Trained_Results = tfp.optimizer.bfgs_minimize(value_and_gradients_function=loss_gradient(params, x_train,y_train,gradient_type),
                                      initial_position=init_params,
                                      tolerance=1e-8,##-2
                                      max_iterations=1000,max_line_search_iterations=100,
                                      parallel_iterations=4)                                      
                
# In[5]:
# Estimated Parameters
est_para = pd.DataFrame(Trained_Results.position.numpy(), columns=['Coef.'])
est_para1=est_para.copy()
#Hessian matrix
Hessian=Trained_Results.inverse_hessian_estimate.numpy()
Hessian_modified = Hessian.copy()  



gradients = []  
for i in range(int(len(y_train)/number_choices)):
    print('save_gradients',i)
    gradient =loss_gradient(params,
                            x_train[i*number_choices:(i+1)*number_choices,:],
                            y_train[i*number_choices:(i+1)*number_choices,:],
                            i)(params)[1].numpy()
    gradients.append(gradient)
gradients_array = np.concatenate(gradients, axis=0).reshape(-1,len(params))
varcov_matrix=pd.DataFrame(Hessian_modified/(len(y_train)/number_choices))

# Standard Errors
Std_err = pd.DataFrame(np.sqrt(np.diag(varcov_matrix.values)),columns=['Std.err'])

a11=est_para.values
a12=Std_err.values
a13=a11/a12
# t-ratio
t_ratio = pd.DataFrame(est_para.values / Std_err.values, columns=['t-ratio'])

# Correlation matrix     
cor_matrix=pd.DataFrame(varcov_matrix.values/ np.dot(Std_err.values, Std_err.values.T))
gradients_B=np.dot( gradients_array.T,gradients_array)
# gradients_B2=np.var(gradients_array)
gradients_B_modified = gradients_B.copy()  


robvarcov_matrix=pd.DataFrame(np.dot(np.dot(varcov_matrix.values, gradients_B_modified), varcov_matrix.values))

# Robust Standard Errors
robStd_err = pd.DataFrame(np.sqrt(np.diag(robvarcov_matrix.values)),columns=['robStd.err'])

# Robust t-ratio
robt_ratio = pd.DataFrame(est_para.values / robStd_err.values, columns=['robt-ratio'])

# Robust correlation matrix      
robcor_matrix=robvarcov_matrix.values/ np.dot(robStd_err.values, robStd_err.values.T)

# Estimation results table
Est_result = pd.concat([est_para, Std_err, t_ratio, robStd_err, robt_ratio], axis=1)
print(Est_result)
t1=time.time()
print('LC estimation using Time: ', t1 - t0, '\n')

# Loglikelihood Function
LL_initi =tf.reduce_sum(tf.math.log(model_fun(x_train, y_train, params_ini)))
LL_final =tf.reduce_sum(tf.math.log(model_fun(x_train, y_train, params)))

LAMADA=-2*LL_initi+2*LL_final
print("LL(initial):", LL_initi.numpy())
print("LL(final):  ", LL_final.numpy())
print("LAMADA:  ", LAMADA.numpy())

# Akaike information criterion (AIC), less is better
Estimated_parameters = len(params)
AIC = -2 * LL_final + 2 * Estimated_parameters
print("AIC:        ", AIC.numpy())
# Bayesian information criterion (BIC), less is better
BIC = -2 * LL_final + Estimated_parameters * np.log(x_train.shape[0])
print("BIC:        ", BIC.numpy())

stats = pd.DataFrame({
    'Statistic': ['LL(initial)', 'LL(final)', 'LAMADA', 'AIC', 'BIC','Time'],
    'Value': [LL_initi.numpy(), LL_final.numpy(), LAMADA.numpy(), AIC.numpy(), BIC.numpy(),t1-t0]
})

Est_result = pd.concat([Est_result, stats], ignore_index=True)

Est_result.to_csv(inputLocation+"output_params_testout_LC2-ND.csv", index=True)