# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:54:12 2020

@author: canlinzhang
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numpy as np
#import random
#import tensorflow as tf
#import nltk
#import datetime
#import re
#import csv
#import os
#import matplotlib.pyplot as plt
#import pylab as pl
#import zipfile

from itertools import combinations 
from copy import deepcopy
#from nltk.corpus import wordnet as wn
#from scipy.stats import spearmanr
#from scipy.stats import binom

regular_threshold = 0.95

#bd_total = 2 #for size-2 and: need these many entities b
#bd_out = 1 #for size-2 and: need these many entities in b/a
#prob_gap_bd = 0.05 #prob difference between a|(b_and_c) and a|(b/c)
#count_bd = 100 #for each b, we consider at most these many c
tiny_bd = 5
count_bd = 100 #for each b, we consider at most these many c

################################################
####load the train, valid and test set##########
with open ('train', 'r') as f:
    train = f.readlines()
    train_ = set({})
    for i in range(len(train)-1):
        x = train[i].split()
        x_ = tuple(x)
        train_.add(x_)
    del(train)

with open ('valid', 'r') as f_v:
    valid = f_v.readlines()
    valid_ = set({})
    for i in range(len(valid)-1):
        x = valid[i].split()
        x_ = tuple(x)
        valid_.add(x_)
    del(valid)

with open ('test', 'r') as f_t:
    test = f_t.readlines()
    test_ = set({})
    for i in range(len(test)-1):
        x = test[i].split()
        x_ = tuple(x)
        test_.add(x_)
    del(test)


'''
1. the self_dict storing the occurred entity of each structure
2. the regular_dict storing the co-occurrence of each [r,t] with other structures
'''

####relation dict#################
index = 0
relation_dict = {}
inverse_relation_dict = {}
for key in train_:
    if key[1] not in relation_dict:
        relation_dict.update({key[1]:index})
        inverse_relation_dict.update({index:key[1]})
        index += 1

#we exclude the inverse relations for counting the number of relations
num_r = len(relation_dict)

#add in the inverse relation
for i in range(num_r):
    relation_ = inverse_relation_dict.get(i)
    inverse_r = '_inverse' + relation_
    relation_dict.update({inverse_r:index})
    inverse_relation_dict.update({index:inverse_r})
    index += 1

####entity dict###################
index = 0
entity_dict = {}
inverse_entity_dict = {}
for key in train_:
    if key[0] not in entity_dict:
        entity_dict.update({key[0]:index})
        inverse_entity_dict.update({index:key[0]})
        index += 1
    if key[2] not in entity_dict:
        entity_dict.update({key[2]:index})
        inverse_entity_dict.update({index:key[2]})
        index += 1
for key in test_:
    if key[0] not in entity_dict:
        entity_dict.update({key[0]:index})
        inverse_entity_dict.update({index:key[0]})
        index += 1
    if key[2] not in entity_dict:
        entity_dict.update({key[2]:index})
        inverse_entity_dict.update({index:key[2]})
        index += 1
for key in valid_:
    if key[0] not in entity_dict:
        entity_dict.update({key[0]:index})
        inverse_entity_dict.update({index:key[0]})
        index += 1
    if key[2] not in entity_dict:
        entity_dict.update({key[2]:index})
        inverse_entity_dict.update({index:key[2]})
        index += 1

num_e = len(entity_dict)
    
#create the set of triples using id instead of string
train = set({})
valid = set({})
test = set({})
for key in train_:
    s = entity_dict[key[0]]
    r = relation_dict[key[1]]
    t = entity_dict[key[2]]
    if (s,r,t) not in train:
        train.add((s,r,t))
for key in valid_:
    s = entity_dict[key[0]]
    r = relation_dict[key[1]]
    t = entity_dict[key[2]]
    if (s,r,t) not in valid:
        valid.add((s,r,t))
for key in test_:
    s = entity_dict[key[0]]
    r = relation_dict[key[1]]
    t = entity_dict[key[2]]
    if (s,r,t) not in test:
        test.add((s,r,t))


########################################################
####pre-construction: n-hop structure and rule dict#####

'''n-hop are sets in dictionary as {entity_id: {structure_id}}'''
'''only consider one hop here'''
one_hop = {}

index = 0
structure_dict = {}
inverse_structure_dict = {}

#basic_set stores all the one-hop eas (r,t)
#we also call one-hop eas basic eas
basic_set = set({}) 
for i in range(num_e):
    one_hop[i] = set({})

#build the one_hop set. Only one_hop directly comes from triples
for key in train:
    s,r,t = key[0],key[1],key[2]
    if (r,t) not in structure_dict:
        structure_dict.update({(r,t):index})
        inverse_structure_dict.update({index:(r,t)})
        basic_set.add(index)
        index += 1
    index_ = structure_dict[(r,t)]
    if index_ not in one_hop[s]:
        one_hop[s].add(index_)
    if (r + num_r,s) not in structure_dict:
        structure_dict.update({(r + num_r,s):index})
        inverse_structure_dict.update({index:(r + num_r,s)})
        basic_set.add(index)
        index += 1
    index_ = structure_dict[(r + num_r,s)]
    if index_ not in one_hop[t]:
        one_hop[t].add(index_)
'''
for key in valid:
    s,r,t = key[0],key[1],key[2]
    if (r,t) not in structure_dict:
        structure_dict.update({(r,t):index})
        inverse_structure_dict.update({index:(r,t)})
        basic_set.add(index)
        index += 1
    index_ = structure_dict[(r,t)]
    if index_ not in one_hop[s]:
        one_hop[s].add(index_)
    if (r + num_r,s) not in structure_dict:
        structure_dict.update({(r + num_r,s):index})
        inverse_structure_dict.update({index:(r + num_r,s)})
        basic_set.add(index)
        index += 1
    index_ = structure_dict[(r + num_r,s)]
    if index_ not in one_hop[t]:
        one_hop[t].add(index_)
'''
print('one_hop structure filled in')
print('num_of_one_hop', len(basic_set))


####################################################
#######build the regularity dictionary##############
#sets in self_dict stores entity id: {structure_id:{entity_id}}
#sets in regular_dict stores structure id and the size of overlapping: 
#{structure_id:{structure_id: size}}
#equal_dict stores the ids for each id whose groundings are the same

#self_dict and equal_dict are sets in dictionary, 
#regular_dict is a nested dictionary
#first we build the self dictionary and frame out regularity dictionary,

self_dict = {}
regular_dict = {}
#equal_dict = {}
for i in range(num_e):
    for id_ in one_hop[i]:
        if id_ not in self_dict:
            self_dict[id_] = {i}
            regular_dict[id_] = {}
            #equal_dict[id_] = set({})
        else:
            if i not in self_dict[id_]:
                self_dict[id_].add(i)
print('self dictionary built and regualr dict frame built')

#fill in giant set
#giant_set = set({})
#giant_bd = int(np.sqrt(float(num_e)))
#for key in self_dict:
#    if len(self_dict[key]) >= giant_bd:
#        giant_set.add(key)

#fill in tiny set:
#tiny_set = set({})
#for key in self_dict:
#    if len(self_dict[key]) <= tiny_bd:
#        tiny_set.add(key)

#fill in the regularity dictionary
#in the meanwhile, create the equal_dict
for i in range(num_e):
    for id_0 in one_hop[i]:
        for id_1 in one_hop[i]:
            if (id_1 != id_0) and (id_1 not in regular_dict[id_0]):
                temp = self_dict[id_0].intersection(self_dict[id_1])
                regular_dict[id_0].update({id_1:len(temp)})
                #if (len(self_dict[id_0]) == len(temp)) and (
                    #len(self_dict[id_1]) == len(temp)):
                    #equal_dict[id_0].add(id_1)
                    #equal_dict[id_1].add(id_0)
    if i <= 1000 and i % 20 == 0:
        print('fill in regularity dict', i)
    elif i % 100 == 0:
        print('fill in regularity dict', i)
del(one_hop)


############################################
######functions#############################

#store the scores for (k,m,n,n_total)
binomial_prob_holder = {}
#store the regularity bound of k for (m,n,n_total)
regular_bd_holder = {}
#store the promote minable m for (n,n_total)
chance_up_holder = {}
#store the repel minable m for (n,n_total)
chance_dn_holder = {}
#store the undeniable (n,n_total)
undeny_holder = set({})

##define the descending function
def Sort(sub_li,i):
    sub_li.sort(key = lambda x: x[i], reverse=True)
    return sub_li


##the binomial prob function:
#assumed single probability is n/n_total, k positive in m
def binomial_prob(k,m,n,n_total):
    if (k,m,n,n_total) in binomial_prob_holder:
        log_prob = binomial_prob_holder[(k,m,n,n_total)]
    else:
        log_prob = 0.
        for i in range(m-k+1,m+1):
            log_prob += np.log(float(i))
        for i in range(1,k+1):
            log_prob -= np.log(float(i))
        log_prob += float(k)*(np.log(float(n)+0.0000000001)-np.log(float(n_total)))
        log_prob += float(m-k)*(np.log(float(n_total-n)+0.0000000001)-np.log(float(n_total)))
        binomial_prob_holder[(k,m,n,n_total)] = log_prob
    return(log_prob)


##the function providing the regularity boundary on k for a given (m,n,n_total)
##single_prob = n/n_total
##the output is: repe_bd, prom_bd
##for the given (m,n,n_total), if k < repe_bd, then it is repelling. If k > prom_bd, it is promoting.
##But it is possible that repe_bd =0 or prom_bd = m
def regular_bd(m,n,n_total):
    if (m,n,n_total) in regular_bd_holder:
        list_ = regular_bd_holder[(m,n,n_total)]
    else:
        holder = []
        for k in range(0, m + 1):
            log_prob = binomial_prob(k,m,n,n_total)
            prob = np.exp(log_prob)
            holder.append([k,prob])
        Sort(holder,-1)
        holder_  = []
        prob_ = 0.
        i  = 0
        while prob_ <= regular_threshold and i < len(holder):
            holder_.append(holder[i][0])
            prob_ += holder[i][1]
            i += 1
        holder_.sort()
        list_ = [holder_[0],holder_[-1]]
        regular_bd_holder[(m,n,n_total)] = list_
    return(list_[0],list_[1])

##the function checking for a given (n,n_total), all the possible m
#that can have a (m,n,n_total) with positive repe_bd,
#or prom_bd less than m
#i.e., 'minable'
def regular_minable(n,n_total):
    if (n,n_total) in chance_up_holder:
        chance_up = chance_up_holder[(n,n_total)]
        chance_dn = chance_dn_holder[(n,n_total)]
    else:
        chance_up = set({})
        chance_dn = set({})
        if n > 0 and n < n_total:
            for m in range(1,n_total+1):
                min_ = max(0,m+n-n_total)
                max_ = min(m,n)
                repe_bd,prom_bd = regular_bd(m, n, n_total)
                if repe_bd > min_: #repelling minable
                    chance_dn.add(m)
                if prom_bd < max_: #promoting minable
                    chance_up.add(m)
        chance_up_holder[(n,n_total)] = chance_up
        chance_dn_holder[(n,n_total)] = chance_dn
        if len(chance_up) == 0 and len(chance_dn) == 0:
            undeny_holder.add((n,n_total))
    return(chance_dn,chance_up)


##############################################
####reasoning#################################
prob_dict = {} #{(s,id_):probability}
cond_dict = {} #{(s,id_):condition}, the reason of this prediction
sgl_rule_dict = {} #{(id_0,id_1):probability}
cpd_rule_dict = {} #{(id_0,(id_1,id_2)):probability}

count_prom = 0
count_repe = 0


#update function:
def update(s,id_0,prob,cond):
#    global count_prom
#    global count_repe
    if (s,id_0) not in prob_dict:
        prob_dict[(s,id_0)] = prob
        cond_dict[(s,id_0)] = cond
    elif prob > prob_dict[(s,id_0)]:
        prob_dict[(s,id_0)] = prob
        cond_dict[(s,id_0)] = cond
    if type(cond) == type(1):
        sgl_rule_dict[(id_0,cond)] = prob
    if type(cond) == type((1,2)):
        cpd_rule_dict[(id_0,cond)] = prob
#    if ('prom' in str_) or ('prom_a' in str_) or (
#        'prom_n' in str_):
#        count_prom += 1
#    if ('repe' in str_) or ('repe_a' in str_) or (
#        'repe_n' in str_):
#        count_repe += 1
            
#update the undeniable hoder in ahead
for n_total in range(1,100):
    if n_total <= 30:
        for n in range(0,n_total+1):
            set_dn,set_up = regular_minable(n,n_total)
    else:
        for n in range(0,4):
            set_dn,set_up = regular_minable(n,n_total)
        for n in range(n_total-3,n_total+1):
            set_dn,set_up = regular_minable(n,n_total)
    print('fill in undeny', n_total)

count_ = 0
sub_count = 0
#complete the single condition
for id_0 in basic_set:
    if sub_count == 5:
        break
    (r,t) = inverse_structure_dict[id_0]
    if r == 4:
        sub_count += 1
        n = len(self_dict[id_0])
        inner_count = 0
        for id_1 in regular_dict[id_0]:
            k,m = regular_dict[id_0][id_1],len(self_dict[id_1])
            prob = float(k)/float(m)
            repe_bd,prom_bd = regular_bd(m,n,num_e)
            str_0 = set({})
            if k > prom_bd:
                str_0.add('prom')
                count_prom += 1
            if k < repe_bd:
                str_0.add('repe')
                count_repe += 1
            #dependency
            if len(str_0) != 0:
                if ((k,m) in undeny_holder) or (prob >= 0.7):
                    for s in self_dict[id_1]:
                        update(s,id_0,prob,id_1)
                else:
                    invalid = 0
                    id_2_count = 0 #how many id_2 worked out
                    for id_2 in regular_dict[id_1]:
                        if id_2_count >= count_bd or invalid >= 2:
                            break
                        elif (id_2 != id_0) and (regular_dict[id_1][id_2] < m):
                            #b and c
                            temp_1a2 = self_dict[id_1].intersection(self_dict[id_2])
                            temp_0a1a2 = self_dict[id_0].intersection(temp_1a2)
                            k_a,m_a = len(temp_0a1a2),len(temp_1a2)
                            prob_a = float(k_a)/float(m_a)
                            repe_bd_a, prom_bd_a = regular_bd(m_a,k,m)
                            if id_2 <= id_1:
                                temp_a = (id_2,id_1)
                            else:
                                temp_a = (id_1,id_2)
                            #b not c
                            temp_1n2 = self_dict[id_1].difference(self_dict[id_2])             
                            k_n, m_n = k-k_a, m-m_a
                            prob_n = float(k_n)/float(m_n)
                            repe_bd_n, prom_bd_n = regular_bd(m_n,k,m)
                            temp_n = (-1*id_2,id_1)
                            str_ = set({})
                            if k_a > prom_bd_a:
                                str_.add('prom_a')
                                count_prom += 1
                            if k_a < repe_bd_a:
                                str_.add('repe_a')
                                count_repe += 1
                            if k_n > prom_bd_n:
                                str_.add('prom_n')
                                count_prom += 1
                            if k_n < repe_bd_n:
                                str_.add('repe_n')
                                count_repe += 1
                            if len(str_) != 0:
                                invalid += 1
                                if ((k_a,m_a) in undeny_holder) or (
                                prob_a >= 0.7):
                                    for s in temp_1a2:
                                        update(s,id_0,prob_a,temp_a)
                                if ((k_n,m_n) in undeny_holder) or (
                                prob_n >= 0.7):
                                    for s in temp_1n2:
                                        update(s,id_0,prob_n,temp_n)
                            id_2_count += 1
            inner_count += 1
            if count_ <= 5000 and inner_count % 50 == 0:
                print('inner reasoning', inner_count,len(regular_dict[id_0]))
    count_ += 1
    if count_ <= 100 and count_ % 1 == 0:
        print('reasoning completed on index', count_)
    elif count_ <= 1000 and count_ % 5 == 0:
        print('reasoning completed on index', count_)
    elif count_ <= 5000 and count_ % 20 == 0:
        print('reasoning completed on index', count_)
    elif count_ % 100 == 0:
        print('reasoning completed on index', count_)

#delete the regular_dict
del(regular_dict)

##count how many triples are decided by size-2 'and'
count_2 = 0
for key in cond_dict:
    if type(cond_dict[key]) == type((1,2)):
        count_2 += 1

#the function based on the entire triple evaluating
def new_prob(s,r,t):
    prob, prob_0, prob_1 = 0., 0., 0.
    iv_r = r + num_r
    if (r,t) in structure_dict:
        id_0 = structure_dict[(r,t)]
        if (s,id_0) in prob_dict:
            prob_0 = prob_dict[(s,id_0)]
    if (iv_r,s) in structure_dict:
        id_1 = structure_dict[(iv_r,s)]
        if (t,id_1) in prob_dict:
            prob_1 = prob_dict[(t,id_1)]
    prob = max(prob_0, prob_1)
    return(prob)


###############################################
###############################################
###############################################
#entity to text 
with open ('entity2text', 'r') as f_e2t:
    e2t = f_e2t.readlines()
    e2t_ = set({})
    for i in range(len(e2t)):
        x = e2t[i].split()
        x_ = tuple(x)
        e2t_.add(x_)
    del(e2t)
    
#relation to text 
with open ('relation2text', 'r') as f_r2t:
    r2t = f_r2t.readlines()
    r2t_ = set({})
    for i in range(len(r2t)):
        x = r2t[i].split()
        x_ = tuple(x)
        r2t_.add(x_)
    del(r2t)

##the dictionary for entity to text
e2t_dict = {}
for key in e2t_:
    temp_str = '_'
    for j in range(1,len(key)):
        temp_str += key[j]
        temp_str += '_'
    e2t_dict.update({key[0]:temp_str})

##the dictionary for relation to text
#however, we will not use it sense the relation is clear enough
r2t_dict = {}
for key in r2t_:
    temp_str = '_'
    for j in range(1,len(key)):
        temp_str += key[j]
        temp_str += '_'
    r2t_dict.update({key[0]:temp_str})


##output results
single_grd_reason = open("single_grounding_reason.txt","w")
out_count = 0
for key in prob_dict:
    if out_count == 5000000:
        break
    prob = prob_dict[key]
    if prob >= 0.6:
        s_ = inverse_entity_dict[key[0]]
        s = e2t_dict[s_]
        tuple_ = inverse_structure_dict[key[1]]
        r = inverse_relation_dict[tuple_[0]]
        t_ = inverse_entity_dict[tuple_[1]]
        t = e2t_dict[t_]
        cond = cond_dict[key]
        if type(cond) == type(1):
            cond_tuple_ = inverse_structure_dict[cond]
            cond_r = inverse_relation_dict[cond_tuple_[0]]
            cond_t_ = inverse_entity_dict[cond_tuple_[1]]
            cond_t = e2t_dict[cond_t_]
            single_grd_reason.write(s)
            single_grd_reason.write('\n')
            single_grd_reason.write(r)
            single_grd_reason.write('\n')
            single_grd_reason.write(t)
            single_grd_reason.write('\n')
            single_grd_reason.write(str(prob))
            single_grd_reason.write('\n')
            single_grd_reason.write(cond_r)
            single_grd_reason.write('\n')
            single_grd_reason.write(cond_t)
            single_grd_reason.write('\n')
            single_grd_reason.write('\n')
            single_grd_reason.write('\n')
            single_grd_reason.write('\n')
            out_count += 1
            
cpd_grd_reason = open("compound_grounding_reason.txt","w")
out_count = 0
for key in prob_dict:
    if out_count == 5000000:
        break
    prob = prob_dict[key]
    if prob >= 0.6:
        s_ = inverse_entity_dict[key[0]]
        s = e2t_dict[s_]
        tuple_ = inverse_structure_dict[key[1]]
        r = inverse_relation_dict[tuple_[0]]
        t_ = inverse_entity_dict[tuple_[1]]
        t = e2t_dict[t_]
        cond = cond_dict[key]
        if type(cond) == type((1,2)):
            if cond[0] > 0:
                cond_tuple_0 = inverse_structure_dict[cond[0]]
                cond_r_0 = inverse_relation_dict[cond_tuple_0[0]]
                cond_t_0 = inverse_entity_dict[cond_tuple_0[1]]
                cond_t0 = e2t_dict[cond_t_0]
                cond_tuple_1 = inverse_structure_dict[cond[1]]
                cond_r_1 = inverse_relation_dict[cond_tuple_1[0]]
                cond_t_1 = inverse_entity_dict[cond_tuple_1[1]]
                cond_t1 = e2t_dict[cond_t_1]
                cpd_grd_reason.write(s)
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write(r)
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write(t)
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write(str(prob))
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write(cond_r_0)
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write(cond_t0)
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write(cond_r_1)
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write(cond_t1)
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write('\n')
            else:
                cond_tuple_0 = inverse_structure_dict[-1*cond[0]]
                cond_r_0 = inverse_relation_dict[cond_tuple_0[0]]
                cond_t_0 = inverse_entity_dict[cond_tuple_0[1]]
                cond_t0 = e2t_dict[cond_t_0]
                cond_tuple_1 = inverse_structure_dict[cond[1]]
                cond_r_1 = inverse_relation_dict[cond_tuple_1[0]]
                cond_t_1 = inverse_entity_dict[cond_tuple_1[1]]
                cond_t1 = e2t_dict[cond_t_1]
                cpd_grd_reason.write(s)
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write(r)
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write(t)
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write(str(prob))
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write('neg' + cond_r_0)
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write('neg' + cond_t0)
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write(cond_r_1)
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write(cond_t1)
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write('\n')
                cpd_grd_reason.write('\n')                
            out_count += 1




