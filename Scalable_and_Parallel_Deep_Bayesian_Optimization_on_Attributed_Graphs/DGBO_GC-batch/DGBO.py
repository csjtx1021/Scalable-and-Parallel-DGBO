#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:24:07 2018

@author: cuijiaxu
"""
import random
import time, sys
import numpy as np
import pylab as pl
import math
import pickle

import networkx as nx
import scipy.sparse as sp

##used to init the setting of params on overfitting at init stage
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
import gc
##

import ChooseNext
import ChooseNext_batch
import AdaptiveBasis
import base_prior

import parse_arg
import DeepSurrogateModel

##
import load_data
##
evalnum=0

def load_candidates(info):
    if info.gcn==True: 
        smile_lists,edge_lists,feature_lists,label_lists,attr_lists=load_data.load_data(info.dataset,topN=topN)
        cans=[]
        #attr_lists=[]
        for idx in range(len(smile_lists)):
            if idx%5000==0:
                print "prepocessed %s/%s"%(idx,len(smile_lists))
            num_nodes=feature_lists[idx].shape[0]
            adjs=[]
            for rel_idx in range(len(edge_lists[idx])):
                num_edges=len(edge_lists[idx][rel_idx])
                edges=np.array(edge_lists[idx][rel_idx],dtype=np.integer).reshape(num_edges,2) 
                #print rel_idx,"edges:",edges
                adj = sp.coo_matrix((np.ones(edges.shape[0]*2), (np.array(list(edges[:,0])+list(edges[:,1])),np.array(list(edges[:,1])+list(edges[:,0])))),shape=(num_nodes, num_nodes), dtype=np.float32)
                ##this line is just for model == 'rgcn_no_reg' or model == 'rgcn_with_reg'
                adj=sp.coo_matrix(DeepSurrogateModel.preprocess_adj(adj))
                ###
                adjs.append(adj)
                #print adj.todense()
            
            feature=np.array(feature_lists[idx],dtype=np.float32)

            if len(attr_lists[idx])==0:
                #---------
                #"There aren't global attributes, so we add the following features as global attributes."
                #---------
                G=nx.Graph()
                for rel_idx in range(len(edge_lists[idx])):
                    G.add_edges_from(edge_lists[idx][rel_idx])
                nodenum=G.number_of_nodes()
                edgenum=G.number_of_edges()
                avgdeg=np.mean(nx.degree_centrality(G).values())
                avgbet=np.mean(nx.betweenness_centrality(G).values())
                avgclo=np.mean(nx.closeness_centrality(G).values())
                avgclu=nx.average_clustering(G)
                attr=np.array([nodenum,edgenum,avgdeg,avgbet,avgclo,avgclu])
            else:
                attr=np.array(attr_lists[idx])
            
            cans.append([idx,adjs,feature,attr])
        #np.random.shuffle(cans)######
        return np.array(cans)
    else:   
        return np.linspace(0,1,100).reshape(100,1)
    
def load_y(data,info):
    
    y=load_data.load_data_y(info.dataset,topN=topN)
    
    return np.array(y)
        
    
def evaluate(info,x):
    print "evaluate %s ..."%(evalnum)
    start = time.time()
    if info.gcn==True:
        y=evaluate_a_graph(x)
    else:    
        y=evaluate_a_vector(x)
        pl.figure(3)
        pl.scatter(x,y)
    global evalnum
    evalnum+=1
    end = time.time()
    print "feedback : %s (COST %s)"%(y,end-start)
    return y

def evaluate_a_graph(x):
    y=data.y[x[0]]
    return y

def evaluate_a_vector(x):
    y=np.sinc(np.array(x).reshape(1,1) * 10 - 5).sum(axis=1)[:, None][0,0] 
    return y

def initialization(data,n,info):
    print "-----------initialization-----------"
    xlist=np.linspace(0,len(data.candidates)-1,len(data.candidates))
    xlistinit=[]
    if len(data.candidates)%n==0:
        delta=len(data.candidates)/n
    else:
        delta=len(data.candidates)/n+1
    for i in range(n):
        xlistinit.append(random.sample(xlist[i*delta:min((i+1)*delta,len(data.candidates))],1)[0])
    xlist=xlistinit
    xlist=np.array(xlist,dtype=np.integer)
    for i in range(n):
        print "initialization [%s] ..."%(i)
        y=evaluate(info,data.candidates[xlist[i]])
        info.add(xlist[i],y)
    if info.overfit_type=="opt":
        print "init the setting of params on overfitting..."
        bo = BayesianOptimization(initsettingCV,{'dropout_': (0,1), 'weight_decay_': (1e-5,1e-1)},random_state=rng)
        bo.maximize(init_points=21, n_iter=30, acq='ei')
        print bo.res['max']
        with open("results/initprocess_%s.pk"%(info.rseed), 'w') as f2:
            pickle.dump(bo.res['all'], f2)
        #exit(1)
        dropout_=bo.res['max']['max_params']['dropout_']
        weight_decay_=bo.res['max']['max_params']['weight_decay_']

        info.dgcn.dropout=dropout_
        info.dgcn.weight_decay=weight_decay_
    if choose_model=="EIMCMC_BATCH":
        basis=AdaptiveBasis.AdaptiveBasis(data,info,data.candidates[info.observedx],True)
        info.refresh_phi(basis)

def initsettingCV(dropout_,weight_decay_):
    X = np.linspace(0,len(info.observedy)-1,len(info.observedy),dtype=np.integer)
    kf = KFold(n_splits=5)
    models=[]
    count_k=0
    for train, test in kf.split(X):
        count_k+=1
        print("[cv %s] %s %s" % (count_k,train, test))
        model_temp=DeepSurrogateModel.CombDGCNWithDNN(model=model_name,basis_num=basis_num,conv_layers=conv_layers,  conv_hidden=conv_hidden, pool_hidden=pool_hidden, conv_act_type=conv_act_type, pool_act_type=pool_act_type, learning_rate=learning_rate, epochs=epochs, dropout=dropout_, weight_decay=weight_decay_,dnn_layers=dnn_layers, dnn_hidden=dnn_hidden,dnn_act_type=dnn_act_type,inputvec_dim=inputvec_dim,ALPHA=ALPHA,overfit_type=overfit_type)
        models.append(model_temp)
        print("add a model")
    models_acc=[]
    for model_ in models:
        acc=model_.train(data.candidates[np.array(info.observedx)[train]],np.array(info.observedy)[train].reshape(len(train),),True,x_test=data.candidates[np.array(info.observedx)[test]],y_test=np.array(info.observedy)[test].reshape(len(test),))
        models_acc.append(acc)
        print "acc=",acc
        if acc==np.inf or math.isnan(acc):
            return -10000.0
    ##delete models to release memory space
    for model_ in models:
        del model_
    gc.collect()
    return -np.mean(models_acc)

def run_one(data,i,info):
    print "-----------iteration [%s]-----------"%(i)
    start = time.time()
    info.iter=i
    if choose_model=="Random":
        Next_list,_=ChooseNext_batch.ChooseNext_Random(data,info)
    elif choose_model=="EIMCMC_BATCH":
        Next_list,_=ChooseNext_batch.ChooseNext_batch(data,info)
    else:
        print "There is no choose model [%s] in options. And we use default EIMCMC_BATCH model instead."%choose_model
        Next_list,_=ChooseNext_batch.ChooseNext_batch(data,info)
    for Next in Next_list:
        if info.eval_jump(info.forcejump,Next):
            Next_jump=random.randint(0,len(data.candidates)-1)
            print "Force jump from [%s] to [%s]."%(Next,Next_jump)
            Next=Next_jump
        y=evaluate(info,data.candidates[Next])
        info.add(Next,y)
        info.pending_set.remove(Next)
    if choose_model=="EIMCMC_BATCH":
        relearn_NN=False
        if (i+1)%info.relearn_NN_period == 0:
            relearn_NN=True
        info.refresh_phi(AdaptiveBasis.AdaptiveBasis(data,info,data.candidates[info.observedx],relearn_NN))
    end = time.time()
    print "iteration [%s] COST %s"%(i,end-start)
    with open("results/timecost-%s.txt"%(info.rseed),'a') as ff:
        ff.write("%s\t%s\n"%(i,end-start))
    #info.plot_curve()
    
def run_loop(data,maxiter,info):
    for i in range(maxiter):
        run_one(data,i,info)
        if max(info.observedy)>=max(data.y):
            info.observedx=info.observedx+[info.observedx[np.argmax(info.observedy)] for fill_idx in range(maxiter-(i+1))]
            info.observedy=info.observedy+[max(info.observedy) for fill_idx in range(maxiter-(i+1))]
            break
def BO_process(data,params,info):
    initialization(data,params.initn,info)
    #info.plot_curve()
    run_loop(data,params.maxiter,info)
    
class params():
    def __init__(self, initn, maxiter):
        self.initn = initn
        self.maxiter = maxiter
        self.y=None
class data():
    def __init__(self):
        self.candidates = None
        self.y=None
class store():
    def __init__(self,rseed, rng,resample_period=10,relearn_NN_period=10,forcejump=3,gcn=True,dataset="datasets/synthetic_datasets",max_pending=1,pending_samples=100,overfit_type=None):
        self.observedx = []
        self.observedy = []
        self.phi_matrix = []
        self.dataset=dataset
        self.rseed=rseed
        self.rng=rng
        # Prior for alpha=1/sigma2
        self.ln_prior_alpha=base_prior.LognormalPrior(sigma=0.1, mean=-10, rng=self.rng)
        # Prior for noise^2 = 1 / beta
        self.prior_noise2 = base_prior.HorseshoePrior(scale=0.1, rng=self.rng)
        self.hyp_samples=[]
        self.resample_period=resample_period
        self.relearn_NN_period=relearn_NN_period
        self.iter=0
        self.pos=[]
        self.forcejump=forcejump
        self.w_m0=np.zeros(dnn_hidden).reshape(dnn_hidden,1)
        self.gcn=gcn
        self.All_cand_node_num=0
        self.pending_set=[]
        self.max_pending=max_pending
        self.pending_samples=pending_samples
        self.overfit_type=overfit_type
        if self.gcn==True:
            self.dgcn=DeepSurrogateModel.CombDGCNWithDNN(model=model_name,basis_num=basis_num,conv_layers=conv_layers,  conv_hidden=conv_hidden, pool_hidden=pool_hidden, conv_act_type=conv_act_type, pool_act_type=pool_act_type, learning_rate=learning_rate, epochs=epochs, dropout=dropout, weight_decay=weight_decay,dnn_layers=dnn_layers, dnn_hidden=dnn_hidden,dnn_act_type=dnn_act_type,inputvec_dim=inputvec_dim,ALPHA=ALPHA,overfit_type=overfit_type)
        
    def eval_jump(self,lastnum,newx):
        for i in self.observedx[-min(lastnum,len(self.observedx)):]:
            if newx!=i:
                return False
        return True
    def add(self,x,y):
        self.observedx=self.observedx+[x]
        self.observedy=self.observedy+[y]
    def add_phi(self,phi,num):
        if num==1:
            self.phi_matrix=self.phi_matrix+[phi]
        else:
            self.phi_matrix=self.phi_matrix+list(phi)
    #store last layer output of the retrained neural network
    def refresh_phi(self,phi_matrix):
        self.phi_matrix=phi_matrix
    def set_w_m0(self,w_m0):
        self.w_m0=w_m0
        
    def print_observed(self):
        with open("results/result-RGBODGCN-r%s.txt"%self.rseed, 'w') as fout:
            for i in range(len(self.observedx)):
                fout.write("%s\t%s\n"%(self.observedx[i],self.observedy[i]))
        print self.observedx,self.observedy
        
    def plot_curve(self):
        pl.figure(1)
        pl.scatter(np.linspace(1,len(self.observedy),len(self.observedy)),self.observedy,marker='+',c='k')
        curbest=[np.max(self.observedy[0:i+1]) for i in range(len(self.observedy))]
        pl.plot(np.linspace(1,len(self.observedy),len(self.observedy)),curbest,'r')
        pl.title('Iteration # = %s, best = %s'%(self.iter,max(curbest)))
        pl.savefig("results/curve-RGBODGCN-r%s.pdf"%self.rseed)
        #pl.show()
        #pl.close(1)


if __name__ == "__main__":
    args=parse_arg.parse_arg()
    if args.run==False:
        print "You need to open the [--run=True] flag to run the model, or open [-h] option to see the help message."
    else:
        conv_layers=int(args.conv_layers)
        conv_hidden=int(args.conv_hidden)
        pool_hidden=int(args.pool_hidden)
        conv_act_type=int(args.conv_act_type)
        pool_act_type=int(args.pool_act_type)
        learning_rate=float(args.learning_rate)
        epochs=int(args.epochs)
        dropout=float(args.dropout)
        weight_decay=float(args.weight_decay)
        dnn_layers=int(args.dnn_layers)
        dnn_hidden=int(args.dnn_hidden)
        dnn_act_type=int(args.dnn_act_type)
        model_name=args.model_name
        ALPHA=float(args.ALPHA)
        if args.rseed==None:
            rseed=args.rseed
        else:
            rseed=int(args.rseed)
        dataset=args.dataset
        basis_num=int(args.basis_num)
        inputvec_dim=int(args.inputvec_dim)
        if args.topN==None:
            topN=args.topN
        else:
            topN=int(args.topN)
        params.initn=int(args.initn)
        params.maxiter=int(args.maxiter)
        resample_period=int(args.resample_period)
        relearn_NN_period=int(args.relearn_NN_period)
        max_pending=int(args.max_pending)
        choose_model=args.choose_model

        overfit_type=args.overfit_type
        
        random.seed(rseed)
        np.random.seed(rseed)
        rng=np.random.RandomState(rseed)
        
        rseed="%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-Comb-%s-%s-%s-alpha%s-%s_%s_%s_%s_%s-pend%s"%(rseed,conv_layers,conv_hidden,pool_hidden,conv_act_type,pool_act_type,learning_rate,epochs,dropout,weight_decay,dnn_layers,dnn_hidden,dnn_act_type,ALPHA,dataset,model_name,basis_num,choose_model,overfit_type,max_pending)
        
        info=store(rseed=rseed,rng=rng,resample_period=resample_period,relearn_NN_period=relearn_NN_period,dataset="datasets/%s"%dataset,max_pending=max_pending,overfit_type=overfit_type)
        
        data=data()
        data.candidates=load_candidates(info)
        data.y=load_y(data,info)
        
        BO_process(data,params,info)
        info.print_observed()


    
    
