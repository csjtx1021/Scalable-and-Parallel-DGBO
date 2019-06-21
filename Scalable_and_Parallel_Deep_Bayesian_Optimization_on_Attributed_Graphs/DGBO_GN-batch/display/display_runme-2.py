#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:06:21 2017

@author: cuijiaxu
"""

import numpy as np
import scipy.io
import scipy.stats
import pylab as pl
from plotshadedErrorBar import plotshadedErrorBar
from matplotlib.ticker import FuncFormatter

import load_data_from_result_file
import plot_scatter
import plot_convcurve

max_y=3.22608925
max_x=1000

fontsize=15
rseed_list=[23456, 312213, 314150, 434234, 264852]
rseed_list_draw=[23456,314150]
    
fig=pl.figure(1, figsize=(8, 4))
fig.subplots_adjust(left=0.08,bottom=0.12,right=0.98,top=0.98,wspace=0.2,hspace=0.55)

count=-1
dataset=[]
dataset2=[]
dataset0=[]
dataset_r=[]
dataset_gn=[]

for rseed in rseed_list:
    
    data_r=load_data_from_result_file.load('../results/random/result-RGBODGCN-r%s-5-48-50-2-0-0.0001-2000-0.03-0.07-Comb-5-45-2-alpha0.8-250k_rndm_zinc_drugs_clean_3_rgcn_with_reg_4_Random.txt'%rseed)
    dataset_r.append(data_r)
    
    """
    data=load_data_from_result_file.load('../results/fixed/result-RGBODGCN-r%s-5-48-50-2-0-0.0001-2000-0.0-1e-05-Comb-5-45-2-alpha0.8-250k_rndm_zinc_drugs_clean_3_rgcn_with_reg_4.txt'%rseed)
    dataset0.append(data)
    #scatter0,trendline0=plot_scatter.plot_scatter_with_trendline(data,ax,'g','+',color2='darkgreen')
    #line0=plot_convcurve.plot_one_convcurve(data,ax,'g')
    """
    
    """
    data=load_data_from_result_file.load('../results/fixed/result-RGBODGCN-r%s-5-48-50-2-0-0.0001-2000-0.03-0.07-Comb-5-45-2-alpha0.8-250k_rndm_zinc_drugs_clean_3_rgcn_with_reg_4.txt'%rseed)
    dataset.append(data)
    scatter1,trendline1=plot_scatter.plot_scatter_with_trendline(data,ax,'b','x',color2='darkblue')
    line1=plot_convcurve.plot_one_convcurve(data,ax,'b')
    """
    
    #data2=load_data_from_result_file.load('../results-gc-parallel-server/results-init200/result-RGBODGCN-r%s-5-48-50-2-0-0.0001-2000-0.279-1e-05-Comb-5-45-2-alpha0.8-250k_rndm_zinc_drugs_clean_3_rgcn_with_reg_4_EIMCMC_BATCH_heur.txt'%rseed)
    data2=load_data_from_result_file.load('../results/result-RGBODGCN-r%s-1e-05-250k_rndm_zinc_drugs_clean_3.txt'%rseed)
    dataset2.append(data2)
    
    data=load_data_from_result_file.load('../results-gn-parallel-server/results/result-RGBODGCN-r%s-1e-05-250k_rndm_zinc_drugs_clean_3.txt'%rseed)
    dataset_gn.append(data)

    if rseed in rseed_list_draw:
        count+=1
        ax = fig.add_subplot(1,2,count+1)

        scatter_r,trendline_r=plot_scatter.plot_scatter_with_trendline(data_r,ax,'k','o',color2='k')
        line_r=plot_convcurve.plot_one_convcurve(data_r,ax,'k')

        scatter2,trendline2=plot_scatter.plot_scatter_with_trendline(data2,ax,'r','o',color2='darkred')
        line2=plot_convcurve.plot_one_convcurve(data2,ax,'r')
        
        scatter_gn,trendline_gn=plot_scatter.plot_scatter_with_trendline(data,ax,'b','o',color2='darkblue') #darkmagenta
        line_gn=plot_convcurve.plot_one_convcurve(data,ax,'b')
        
        lineopt,=ax.plot([1,data.shape[0]],[max_y,max_y],'k:')
        
        pl.ylabel("y",fontsize=fontsize)
        pl.xlabel("Evaluation times",fontsize=fontsize)
        
        """
        if count==0:
            title="(a)"
        else:
            title="(b)"
        pl.title('%s'%title,fontsize=fontsize)
        """
    """
    if (count+1)==len(rseed_list):
        ax.legend([scatter0,scatter1,scatter2,trendline0,trendline1,trendline2,line0,line1,line2,lineopt],['observed by fixed','observed by fixed0','observed by heur','trend line of fixed0','trend line of fixed','trend line of heur','current optimal curve of fixed0','current optimal curve of fixed','current optimal curve of heur','optimal value'])
    """

fig=pl.figure(2, figsize=(6, 4))
fig.subplots_adjust(left=0.12,bottom=0.12,right=0.98,top=0.98,wspace=0.28,hspace=0.55)

ax = fig.add_subplot(1,1,1)

line_r=plot_convcurve.plot_multi_convcurves(dataset_r,max_x,ax,'k','-',max_y=max_y)
#line0=plot_convcurve.plot_multi_convcurves(dataset0,max_x,ax,'g','-',max_y=max_y)
#line1=plot_convcurve.plot_multi_convcurves(dataset,max_x,ax,'b','-',max_y=max_y)
line2=plot_convcurve.plot_multi_convcurves(dataset2,max_x,ax,'r','-',max_y=max_y)
line_gn=plot_convcurve.plot_multi_convcurves(dataset_gn,max_x,ax,'b','-',max_y=max_y)

ax.plot([1,max_x],[max_y,max_y],'k:')
pl.ylabel("y",fontsize=fontsize)
pl.xlabel("Evaluation times",fontsize=fontsize)
pl.ylim(1.2,max_y+0.1)
pl.legend([r"Random",r"PDGBO$_{GC}$",r"PDGBO$_{GN}$"],fontsize=fontsize)
#pl.title('mean')
#ax.legend([line0,line1,line2],['fixed-0-1e-5','fixed-0.03-0.07','heur'])


#pl.savefig("scatters-fixed-0.03-0.07.pdf")
#pl.savefig("scatters-heuristic-fixed-0.03-0.07-0-1e-5.pdf")

pl.show()




