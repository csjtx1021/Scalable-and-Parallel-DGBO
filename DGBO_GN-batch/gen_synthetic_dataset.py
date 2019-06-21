#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:18:43 2017

@author: cuijiaxu
"""

import numpy as np
import multiprocessing as mp
import networkx as nx
#from gensim.models import Word2Vec
from itertools import chain, combinations
from collections import defaultdict
import os,sys, copy, time, math, pickle
import itertools
import scipy.io
#import pynauty
import random
from scipy.spatial.distance import pdist, squareform
#import pyGPs
import scipy.stats
import pylab as pl
#import GraphMeasure

def gen_syn_ds(OUTPUTDIR):
    print "genrating synthetic dataset..."
    GraphSet=[]
    numgraphs=2000
    n_set=np.array((20,30,40,50,60,70,80,90,100,110))#for ER and BA
    p_set=np.array((0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3))#for ER
    m_set=np.array((1,2,3,4,5,6,7,8,9,10))#for BA
    rseed_set=np.array((314150,312213,434234,264852,231255,659956,435347,898232,675665,234690))
    key=-1
    ##ER
    for n in range(len(n_set)):
        for p in range(len(p_set)):
            for rseed in range(len(rseed_set)):
                key+=1
                G1=nx.fast_gnp_random_graph(n_set[n],p_set[p],rseed_set[rseed])
                nx.write_edgelist(G1, "%s/%s-ER_%s_%s_%s.edgelist"%(OUTPUTDIR,key,n_set[n],p_set[p],rseed_set[rseed]))
                GraphSet.append(G1)
                if np.mod(key,50)==0:
                    nx.draw_circular(G1, with_labels=True, font_weight='bold')
                    pl.savefig("%s/%s-ER_%s_%s_%s.svg"%(OUTPUTDIR,key,n_set[n],p_set[p],rseed_set[rseed]))
#                pl.show()
    ##BA
    for n in range(len(n_set)):
        for m in range(len(m_set)):
            for rseed in range(len(rseed_set)):
                key+=1
                G1=nx.barabasi_albert_graph(n_set[n],m_set[m],rseed_set[rseed])
                nx.write_edgelist(G1, "%s/%s-BA_%s_%s_%s.edgelist"%(OUTPUTDIR,key,n_set[n],m_set[m],rseed_set[rseed]))
                GraphSet.append(G1)               
                if np.mod(key,50)==0:
                    nx.draw_circular(G1, with_labels=True, font_weight='bold')
                    pl.savefig("%s/%s-BA_%s_%s_%s.svg"%(OUTPUTDIR,key,n_set[n],m_set[m],rseed_set[rseed]))
#                pl.show()
                    
def get_syn_ds_name_idx(idx):
    numgraphs=2000
    n_set=np.array((20,30,40,50,60,70,80,90,100,110))#for ER and BA
    p_set=np.array((0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3))#for ER
    m_set=np.array((1,2,3,4,5,6,7,8,9,10))#for BA
    rseed_set=np.array((314150,312213,434234,264852,231255,659956,435347,898232,675665,234690))
    
    filename="%s-ER_%s_%s_%s.edgelist"%(idx,n_set[n],p_set[p],rseed_set[rseed])

def read_syn_ds(OUTPUTDIR):
    print "loading synthetic dataset..."
    GraphSet=[]
    numgraphs=500
    n_set=np.array((20,30,40,50,60))#for ER and BA
    p_set=np.array((0.1,0.15,0.2,0.25,0.3))#for ER
    m_set=np.array((1,2,3,4,5))#for BA
    rseed_set=np.array((314150,312213,434234,264852,231255,659956,435347,898232,675665,234690))
    #
    nodenum_max=0
    nodenum_min=1000000000
    edgenum_max=0
    edgenum_min=1000000000
    avgdeg_max=0
    avgdeg_min=1000000000
    avgbet_max=0
    avgbet_min=1000000000
    avgclo_max=0
    avgclo_min=1000000000
    avgclu_max=0
    avgclu_min=1000000000
    num_cliques_max=0
    num_cliques_min=100000000
    num_con_max=0
    num_con_min=100000000
    #    test=[]
    key=-1
    ##ER
    for n in range(len(n_set)):
        for p in range(len(p_set)):
            for rseed in range(len(rseed_set)):
                key+=1
                G1=nx.read_edgelist("%s/%s-ER_%s_%s_%s.edgelist"%(OUTPUTDIR,key,n_set[n],p_set[p],rseed_set[rseed]))
                GraphSet.append(G1)
                """
                    nodenum_max=max(nodenum_max,G1.number_of_nodes())
                    nodenum_min=min(nodenum_min,G1.number_of_nodes())
                    edgenum_max=max(edgenum_max,G1.number_of_edges())
                    edgenum_min=min(edgenum_min,G1.number_of_edges())
                    avgdeg_max=max(avgdeg_max,np.mean(nx.degree_centrality(G1).values()))
                    avgdeg_min=min(avgdeg_min,np.mean(nx.degree_centrality(G1).values()))
                    avgbet_max=max(avgbet_max,np.mean(nx.betweenness_centrality(G1).values()))
                    avgbet_min=min(avgbet_min,np.mean(nx.betweenness_centrality(G1).values()))
                    #avgclo_max=max(avgclo_max,np.mean(nx.closeness_centrality(G1).values()))
                    #avgclo_min=min(avgclo_min,np.mean(nx.closeness_centrality(G1).values()))
                    avgclu_max=max(avgclu_max,nx.average_clustering(G1))
                    avgclu_min=min(avgclu_min,nx.average_clustering(G1))
                    #                num_cliques_max=max(num_cliques_max,nx.graph_number_of_cliques(G1))
                    #                num_cliques_min=min(num_cliques_min,nx.graph_number_of_cliques(G1))
                    #                num_con_max=max(num_con_max,nx.number_connected_components(G1))
                    #                num_con_min=min(num_con_min,nx.number_connected_components(G1))
                    #                c=nx.degree_histogram(G1)
                    #                idxc=c.index(max(c))
                    #                num_con_max=max(num_con_max,idxc+1)
                    #                num_con_min=min(num_con_min,idxc+1)
                    
                    #                print idxc+1
                    #                test.append(idxc+1)
                    """

    ##BA
    for n in range(len(n_set)):
        for m in range(len(m_set)):
            for rseed in range(len(rseed_set)):
                key+=1
                G1=nx.read_edgelist("%s/%s-BA_%s_%s_%s.edgelist"%(OUTPUTDIR,key,n_set[n],m_set[m],rseed_set[rseed]))
                GraphSet.append(G1)
                """
                    nodenum_max=max(nodenum_max,G1.number_of_nodes())
                    nodenum_min=min(nodenum_min,G1.number_of_nodes())
                    edgenum_max=max(edgenum_max,G1.number_of_edges())
                    edgenum_min=min(edgenum_min,G1.number_of_edges())
                    avgdeg_max=max(avgdeg_max,np.mean(nx.degree_centrality(G1).values()))
                    avgdeg_min=min(avgdeg_min,np.mean(nx.degree_centrality(G1).values()))
                    avgbet_max=max(avgbet_max,np.mean(nx.betweenness_centrality(G1).values()))
                    avgbet_min=min(avgbet_min,np.mean(nx.betweenness_centrality(G1).values()))
                    #                avgclo_max=max(avgclo_max,np.mean(nx.closeness_centrality(G1).values()))
                    #                avgclo_min=min(avgclo_min,np.mean(nx.closeness_centrality(G1).values()))
                    avgclu_max=max(avgclu_max,nx.average_clustering(G1))
                    avgclu_min=min(avgclu_min,nx.average_clustering(G1))
                    #                num_cliques_max=max(num_cliques_max,nx.graph_number_of_cliques(G1))
                    #                num_cliques_min=min(num_cliques_min,nx.graph_number_of_cliques(G1))
                    #                num_con_max=max(num_con_max,nx.number_connected_components(G1))
                    #                num_con_min=min(num_con_min,nx.number_connected_components(G1))
                    #                c=nx.degree_histogram(G1)
                    #                idxc=c.index(max(c))
                    #                num_con_max=max(num_con_max,idxc+1)
                    #                num_con_min=min(num_con_min,idxc+1)
                    #                print idxc+1
                    #                test.append(idxc+1)
                    """
    #    print num_con_max,num_con_min
#    print num_cliques_max,num_cliques_min ,num_con_max,num_con_min
#    print nodenum_max,nodenum_min,edgenum_max,edgenum_min,avgdeg_max,avgdeg_min,avgbet_max,avgbet_min,avgclo_max,avgclo_min,avgclu_max,avgclu_min
#    print nodenum_max,nodenum_min,edgenum_max,edgenum_min,avgdeg_max,avgdeg_min,avgbet_max,avgbet_min,avgclu_max,avgclu_min
#    pl.plot(range(500),test)
    return GraphSet
def read_syn_ds_2000(OUTPUTDIR):
    print "loading synthetic dataset..."
    GraphSet=[]
    numgraphs=2000
    n_set=np.array((20,30,40,50,60,70,80,90,100,110))#for ER and BA
    p_set=np.array((0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3))#for ER
    m_set=np.array((1,2,3,4,5,6,7,8,9,10))#for BA
    rseed_set=np.array((314150,312213,434234,264852,231255,659956,435347,898232,675665,234690))
#
    nodenum_max=0
    nodenum_min=1000000000
    edgenum_max=0
    edgenum_min=1000000000
    avgdeg_max=0
    avgdeg_min=1000000000
    avgbet_max=0
    avgbet_min=1000000000
    avgclo_max=0
    avgclo_min=1000000000
    avgclu_max=0
    avgclu_min=1000000000
    num_cliques_max=0
    num_cliques_min=100000000
    num_con_max=0
    num_con_min=100000000
#    test=[]
    key=-1
    ##ER
    for n in range(len(n_set)):
        for p in range(len(p_set)):
            for rseed in range(len(rseed_set)):
                key+=1
                G1=nx.read_edgelist("%s/%s-ER_%s_%s_%s.edgelist"%(OUTPUTDIR,key,n_set[n],p_set[p],rseed_set[rseed]))
                GraphSet.append(G1) 
                """
                nodenum_max=max(nodenum_max,G1.number_of_nodes())
                nodenum_min=min(nodenum_min,G1.number_of_nodes())
                edgenum_max=max(edgenum_max,G1.number_of_edges())
                edgenum_min=min(edgenum_min,G1.number_of_edges())
                avgdeg_max=max(avgdeg_max,np.mean(nx.degree_centrality(G1).values()))
                avgdeg_min=min(avgdeg_min,np.mean(nx.degree_centrality(G1).values()))
                avgbet_max=max(avgbet_max,np.mean(nx.betweenness_centrality(G1).values()))
                avgbet_min=min(avgbet_min,np.mean(nx.betweenness_centrality(G1).values()))
                #avgclo_max=max(avgclo_max,np.mean(nx.closeness_centrality(G1).values()))
                #avgclo_min=min(avgclo_min,np.mean(nx.closeness_centrality(G1).values()))
                avgclu_max=max(avgclu_max,nx.average_clustering(G1))
                avgclu_min=min(avgclu_min,nx.average_clustering(G1))
#                num_cliques_max=max(num_cliques_max,nx.graph_number_of_cliques(G1))
#                num_cliques_min=min(num_cliques_min,nx.graph_number_of_cliques(G1))
#                num_con_max=max(num_con_max,nx.number_connected_components(G1))
#                num_con_min=min(num_con_min,nx.number_connected_components(G1))
#                c=nx.degree_histogram(G1)
#                idxc=c.index(max(c)) 
#                num_con_max=max(num_con_max,idxc+1)
#                num_con_min=min(num_con_min,idxc+1)
                
#                print idxc+1
#                test.append(idxc+1)
                """
                
    ##BA
    for n in range(len(n_set)):
        for m in range(len(m_set)):
            for rseed in range(len(rseed_set)):
                key+=1
                G1=nx.read_edgelist("%s/%s-BA_%s_%s_%s.edgelist"%(OUTPUTDIR,key,n_set[n],m_set[m],rseed_set[rseed]))
                GraphSet.append(G1)
                """
                nodenum_max=max(nodenum_max,G1.number_of_nodes())
                nodenum_min=min(nodenum_min,G1.number_of_nodes())
                edgenum_max=max(edgenum_max,G1.number_of_edges())
                edgenum_min=min(edgenum_min,G1.number_of_edges())
                avgdeg_max=max(avgdeg_max,np.mean(nx.degree_centrality(G1).values()))
                avgdeg_min=min(avgdeg_min,np.mean(nx.degree_centrality(G1).values()))
                avgbet_max=max(avgbet_max,np.mean(nx.betweenness_centrality(G1).values()))
                avgbet_min=min(avgbet_min,np.mean(nx.betweenness_centrality(G1).values()))
#                avgclo_max=max(avgclo_max,np.mean(nx.closeness_centrality(G1).values()))
#                avgclo_min=min(avgclo_min,np.mean(nx.closeness_centrality(G1).values()))
                avgclu_max=max(avgclu_max,nx.average_clustering(G1))
                avgclu_min=min(avgclu_min,nx.average_clustering(G1))
#                num_cliques_max=max(num_cliques_max,nx.graph_number_of_cliques(G1))
#                num_cliques_min=min(num_cliques_min,nx.graph_number_of_cliques(G1))
#                num_con_max=max(num_con_max,nx.number_connected_components(G1))
#                num_con_min=min(num_con_min,nx.number_connected_components(G1))
#                c=nx.degree_histogram(G1)
#                idxc=c.index(max(c)) 
#                num_con_max=max(num_con_max,idxc+1)
#                num_con_min=min(num_con_min,idxc+1)
#                print idxc+1
#                test.append(idxc+1)
                 """
#    print num_con_max,num_con_min
#    print num_cliques_max,num_cliques_min ,num_con_max,num_con_min
#    print nodenum_max,nodenum_min,edgenum_max,edgenum_min,avgdeg_max,avgdeg_min,avgbet_max,avgbet_min,avgclo_max,avgclo_min,avgclu_max,avgclu_min
#    print nodenum_max,nodenum_min,edgenum_max,edgenum_min,avgdeg_max,avgdeg_min,avgbet_max,avgbet_min,avgclu_max,avgclu_min
#    pl.plot(range(500),test)
    return GraphSet
def statistics_info(GraphSet):
    x1set=[]
    x2set=[]
    x3set=[]
    x4set=[]
    for idx in range(len(GraphSet)):
        G=GraphSet[idx]
        nodenum=G.number_of_nodes()
        edgenum=G.number_of_edges()
        avgdeg=np.mean(nx.degree_centrality(G).values())
        avgbet=np.mean(nx.betweenness_centrality(G).values())
        x1=(nodenum-12.0)/(60.0-12.0)
        x2=(edgenum-11.0)/(579.0-11.0)
        x3=(avgdeg-0.0333)/(0.3948-0.0333)
        x4=(avgbet-0.0116)/(0.1683-0.0116)
        x1set.append(x1)
        x2set.append(x2)
        x3set.append(x3)
        x4set.append(x4)
    pl.figure(1)
    pl.hist(np.array(x1set))
    pl.title('x1')
    pl.figure(2)
    pl.hist(np.array(x2set))
    pl.title('x2')
    pl.figure(3)
    pl.hist(np.array(x3set))
    pl.title('x3')
    pl.figure(4)
    pl.hist(np.array(x4set))
    pl.title('x4')
    pl.show()
if __name__ == "__main__":
    OUTPUTDIR="datasets/synthetic_datasets_2000"
    #gen_syn_ds(OUTPUTDIR)
    read_syn_ds(OUTPUTDIR)
    #statistics_info(read_syn_ds(OUTPUTDIR))
    
