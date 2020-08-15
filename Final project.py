import numpy as np
import pandas as pd
import math
recent_trust_buffer_array = []

def normalize(p):
    max_val = np.max(p)
    min_val = np.min(p)
    delta = (2 * np.sqrt(min_val * max_val))/(min_val + max_val)
    return delta 

def performance(per):
    print("---- performance ----")
    p = ((per['knowledge'] + per['skill'])/per['eb']) * per['attitude']
    print(p)
    print("normalized performance : {}".format(normalize(p)))
    nor_p = normalize(p)
    return p
    
def recenttrust(per):
    print("---- recent trust ----")
    rval = (0.5 * per['direct_trust']) + (0.5 * per['indirect_trust'])
    #print(rval)
    global recent_trust_buffer_array
    recent_trust_buffer_array = rval
    nor_val = normalize(rval)
    print("normalized recent trust",nor_val)
    return rval

def historictrust(per):
    print("---- historic trust ----")
    h_array = []
    flag = 0
    index = 0
    flag = 0
    for index in range(len(recent_trust_buffer_array)):
        if(flag == 0): 
            h_array.append(0.5)  #assuming the initial impression is good
            flag = 1
        else:
            v = ((0.5 * h_array[index - 1]) + recent_trust_buffer_array[index - 1]) / 2
            h_array.append(v)
    #print("h_array:",h_array)
    his_trust_vals = pd.Series(h_array)
    #print(his_trust_vals)
    nor_his = normalize(his_trust_vals)
    print("normalized historic vlaue {}".format(nor_his))
    return his_trust_vals
    
def directobservation(per):
    print("---- direct obs ----")
    row , col = per.shape
    satisf = per.iloc[1:,6].values
    saf_cur = per.iloc[:,7].values
    satisf = list(satisf)
    satisf = [0] + satisf
    satisf = np.array(satisf)
    alpha = 0.5
    saf_v = (alpha * saf_cur) + (1 - alpha)*satisf
    nor_saf = normalize(saf_v)
    print(nor_saf)
    return saf_v

def feedback(per):
    print("---- feedback ----")
    rec=(per.w*per.ptr_te_r)
    print(rec)
    nor_rec = normalize(rec)
    print(nor_rec)

    phi_t=(0.2*(per.omega_t-per.v_t_p))+(0.8*(per.phi_t1+per.psi_t1))
    print(phi_t)
    psi_t=(0.1*(phi_t-per.phi_t1))+(0.9*per.psi_t1)
    print(psi_t)
    v_t=(0.25*(per.omega_t-psi_t))+(0.75*per.v_t_p)
    print(v_t)
    rep=phi_t+(per.m*psi_t)+v_t
    nor_rep = normalize(rep)
    print(nor_rep)
    
    f=nor_rep + nor_rec
    f = rep + rec
    # print(f)
    # print(normalize(f))
    print(f)
    return f

def log_normalize(v):
    t = v / abs(1-v)
    return math.log(t)    

def sigmoid(x):
    return np.exp(x) / ( 1 + np.exp(x) )

def taninv(x):
    return math.atan(x)


per = pd.read_csv('person_one11.csv')

print("*****************************************************************")
print(per.head())
p = performance(per)
recent_trust = recenttrust(per)
historic_trust = historictrust(per)
#print("expected trust : {} | normalized e-trust: {}".format(historic_trust + recent_trust,normalize(historic_trust + recent_trust)))
do = directobservation(per)
fb = feedback(per)

trust = p + historic_trust + recent_trust + do + fb
print("sum : {}".format(np.sum(trust)))
print("r value: {}".format(sigmoid(np.sum(trust)/100)))

'''
print("final trust value: {}".format(normalize( p + historic_trust + recent_trust + do + fb)))
#print("log value:{}".format(log_normalize(p + normalize(historic_trust + recent_trust) + do + fb)))
print("log value:{}".format(log_normalize(np.sum( p + historic_trust + recent_trust + do + fb))))
print("sigmoid : {}".format(sigmoid(np.sum( p + historic_trust + recent_trust + do + fb))))
print("tan intv: {}".format(taninv(np.sum( p + historic_trust + recent_trust + do + fb))))
'''