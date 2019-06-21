#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:35:40 2018

@author: cuijiaxu
"""
import numpy as np
import time
import scipy.stats
import scipy.linalg as spla
import numpy.random as npr
import pylab as pl
import emcee
#import scipy.optimize as op
#import scipy.stats as sps

import AdaptiveBasis

lnnoise2_lower=-13.0
lnnoise2_upper=1.0
lnsigma2_lower=lnnoise2_lower
lnsigma2_upper=lnnoise2_upper


def ChooseNext_batch(data,info):
    start = time.time()
    
    Next_list=[]
    
    start1=time.time()
    phi_matrix=np.array(info.phi_matrix)
    n,k=phi_matrix.shape
    NSample=20
    if info.iter%info.resample_period==0:
        samples=sampler(info,phi_matrix, np.array(info.observedy).reshape(n,1),NSample)
        info.hyp_samples=samples
    else:
        samples=info.hyp_samples
    end1=time.time()
    print "Sampling cost: %s"%(end1-start1)

    while len(info.pending_set)<info.max_pending:
        
        if len(info.pending_set)==0:
            ##there are no pending
            
            #sampling
            Ainvs=[]
            w_mNs=[]
            for i in range(NSample):
                alpha=samples[i,0]
                beta=samples[i,1]
                A=beta * np.dot(phi_matrix.T, phi_matrix)
                A += np.eye(k) * alpha
                Ainv=np.linalg.inv(A)
                    
                y=np.array(info.observedy).reshape(n,1)
                w_mN=Ainv.dot(alpha*info.w_m0+beta*phi_matrix.T.dot(y))
                    
                Ainvs.append(Ainv)
                w_mNs.append(w_mN)
                #print "mean(w)=%s"%(w_mN)

            start2=time.time()
            vals=[]
            #mmus=[]
            #vvars=[]

            n,k=phi_matrix.shape
            x_can=np.linspace(0,len(data.candidates)-1,len(data.candidates),dtype=np.integer)
            x_can=list(set(list(x_can))-set(info.observedx))
            x_can=list(set(list(x_can))-set(info.pending_set))
            np.random.shuffle(x_can)
            random_size=5000
            random_can=list(x_can[0:min(random_size,len(x_can))])
            phi_star=[]
            for x_idx in random_can:
                xstar=data.candidates[x_idx]
                phi_star.append(AdaptiveBasis.AdaptiveBasis(data,info,xstar,False,True))
            phi_star=np.array(phi_star).reshape(len(random_can),k)
            for j in range(NSample):
                val,_,_=EI(data,info,phi_matrix,info.observedy,1.0/samples[j,0],1.0/samples[j,1],Ainvs[j],w_mNs[j],data.candidates[random_can],phi_star)
                vals.append(val)
                #mmus.append(mu)
                #vvars.append(var)
            #print val,val.shape
            vvals=np.mean(np.array(vals),axis=0)
            #mmus=np.mean(np.array(mmus),axis=0)
            #vvars=np.mean(np.array(vvars),axis=0)
            
            #vvals[info.observedx]=0
            
            Next=random_can[np.argmax(vvals)]
            maxVal=max(vvals)
            end2=time.time()
            print "Max acquisition function cost: %s"%(end2-start2)
            end = time.time()
            print "COST of current iteration %s"%(end-start)
            info.phi_matrix=list(info.phi_matrix)
            print "next idx=%s, acq=%s"%(Next,maxVal)
        
        else:
            ##there are some pending experiments
            phi_matrix_pending=[]
            for x_idx in info.pending_set:
                x_pending=data.candidates[x_idx]
                phi_matrix_pending.append((AdaptiveBasis.AdaptiveBasis(data,info,x_pending,False,True)).reshape(k,))
            phi_matrix_pending=np.array(phi_matrix_pending) #.reshape(len(info.pending_set),k)
            #print phi_matrix.shape,phi_matrix_pending.shape
            phi_matrix_pending=np.concatenate((phi_matrix, phi_matrix_pending))
            
            n,k=phi_matrix.shape
            m_pending=len(info.pending_set)

            #sampling
            #Ainvs=[]
            #w_mNs=[]
            Ainvs_pending=[]
            w_mNs_pending=[]
            ys_pending=[]
            for i in range(NSample):
                #for training n
                alpha=samples[i,0]
                beta=samples[i,1]
                A=beta * np.dot(phi_matrix.T, phi_matrix)
                A += np.eye(k) * alpha
                Ainv=np.linalg.inv(A)
                
                y=np.array(info.observedy).reshape(n,1)
                w_mN=Ainv.dot(alpha*info.w_m0+beta*phi_matrix.T.dot(y))
                
                #Ainvs.append(Ainv)
                #w_mNs.append(w_mN)
                
                #for pending n+m
                A_pending=beta * np.dot(phi_matrix_pending.T, phi_matrix_pending)
                A_pending += np.eye(k) * alpha
                Ainv_pending=np.linalg.inv(A_pending)
                
                mu,var=predict(data,info,phi_matrix,1.0/alpha,1.0/beta,Ainv,w_mN,data.candidates[info.pending_set],phi_matrix_pending[n:,:],var_diag=False)
                var[var<1e-25]=1e-25
                #Avoid numerical errors
                #var=var+np.eye(len(var))*1e-20
                #pend_chol = spla.cholesky(var, lower=True)
                #pending_vals = (np.dot(pend_chol, npr.randn(m_pending,info.pending_samples)) + mu)
                pending_vals = np.random.multivariate_normal(mu.reshape(mu.shape[0],),var,info.pending_samples)
                #print pending_vals.shape
                pending_vals = np.concatenate((np.tile(y.reshape(n,)[:,np.newaxis],(1,info.pending_samples)), pending_vals.T))
                
                #y=np.array(info.observedy).reshape(n,1)
                w_mN_pending=Ainv_pending.dot(alpha*info.w_m0+beta*phi_matrix_pending.T.dot(pending_vals))
                
                ys_pending.append(pending_vals)
                Ainvs_pending.append(Ainv_pending)
                w_mNs_pending.append(w_mN_pending)

                #print "mean(w)=%s"%(w_mN)
            
            start2=time.time()
            vals=[]
            #mmus=[]
            #vvars=[]

            n,k=phi_matrix.shape
            x_can=np.linspace(0,len(data.candidates)-1,len(data.candidates),dtype=np.integer)
            x_can=list(set(list(x_can))-set(info.observedx))
            x_can=list(set(list(x_can))-set(info.pending_set))
            np.random.shuffle(x_can)
            random_size=5000
            random_can=list(x_can[0:min(random_size,len(x_can))])
            phi_star=[]
            for x_idx in random_can:
                xstar=data.candidates[x_idx]
                phi_star.append(AdaptiveBasis.AdaptiveBasis(data,info,xstar,False,True))
            phi_star=np.array(phi_star).reshape(len(random_can),k)
            for j in range(NSample):
                val,_,_=EI_batch(data,info,phi_matrix_pending,ys_pending[j],1.0/samples[j,0],1.0/samples[j,1],Ainvs_pending[j],w_mNs_pending[j],data.candidates[random_can],phi_star)
                vals.append(val)
                #mmus.append(mu)
                #vvars.append(var)
                #print val,val.shape
            vvals=np.mean(np.array(vals),axis=0)
            #mmus=np.mean(np.array(mmus),axis=0)
            #vvars=np.mean(np.array(vvars),axis=0)
            
            #vvals[info.observedx]=0

            Next=random_can[np.argmax(vvals)]
            maxVal=max(vvals)
            end2=time.time()
            print "Max acquisition function cost: %s"%(end2-start2)
            end = time.time()
            print "COST of current iteration %s"%(end-start)
            info.phi_matrix=list(info.phi_matrix)
            print "next idx=%s, acq=%s"%(Next,maxVal)
    
        Next_list.append(Next)
        info.pending_set.append(Next)


    return Next_list,None

def ChooseNext_Random(data,info):
    Next_list=[]
    while len(info.pending_set)<info.max_pending:
        x_can=np.linspace(0,len(data.candidates)-1,len(data.candidates),dtype=np.integer)
        x_can=list(set(list(x_can))-set(info.observedx))
        x_can=list(set(list(x_can))-set(info.pending_set))
        np.random.shuffle(x_can)
        random_size=1
        random_can=list(x_can[0:min(random_size,len(x_can))])
        
        Next=random_can[0]
        
        Next_list.append(Next)
        info.pending_set.append(Next)
    return Next_list,None

def UCB(data,info,phi_matrix,y,sigma2,noise2,Ainv,w_mN,xstar,phi_star):
    mu,var=predict(data,info,phi_matrix,sigma2,noise2,Ainv,w_mN,xstar,phi_star)
    var[var < 1e-25]=1e-25
    result=mu+0.01*np.sqrt(var)
    
    #print mu,var
    return result,mu,var

def EI(data,info,phi_matrix,y,sigma2,noise2,Ainv,w_mN,xstar,phi_star):
    mu,var=predict(data,info,phi_matrix,sigma2,noise2,Ainv,w_mN,xstar,phi_star)
    mu=mu.reshape(mu.shape[0],)
    var[var<1e-25]=1e-25
    maxy=np.max(y)
    std=np.sqrt(var)
    
    #print "mu=%s,std=%s"%(mu,std)
    gamma=(mu-maxy)/std
    #print "gamma=%s"%gamma
    pdfgamma=scipy.stats.norm.pdf(gamma,0,1)
    cdfgamma=scipy.stats.norm.cdf(gamma,0,1)
    result=std*(pdfgamma+gamma*cdfgamma)
    
    #print mu,var
    return result,mu,var

def EI_batch(data,info,phi_matrix,y,sigma2,noise2,Ainv,w_mN,xstar,phi_star):
    mu,var=predict(data,info,phi_matrix,sigma2,noise2,Ainv,w_mN,xstar,phi_star)
    var[var<1e-25]=1e-25
    maxy=np.max(y, axis=0)
    std=np.sqrt(var[:,np.newaxis])
    
    #print "mu=%s,std=%s"%(mu,std)
    gamma=(mu-maxy[np.newaxis,:])/std
    #print "gamma=%s"%gamma
    pdfgamma=scipy.stats.norm.pdf(gamma,0,1)
    cdfgamma=scipy.stats.norm.cdf(gamma,0,1)
    result=std*(pdfgamma+gamma*cdfgamma)
    
    result=np.mean(result, axis=1)
    #print mu,var
    return result,mu,var

def predict(data,info,phi_matrix,sigma2,noise2,Ainv,w_mN,xstar,phi_star,var_diag=True):
    #print Ainv,w_mN
    n,k=phi_matrix.shape
    phi_star=phi_star
    mu=phi_star.dot(w_mN)
    
    if var_diag==True:
        var=np.diag(np.dot(np.dot(phi_star, Ainv), phi_star.T))+noise2
        var=np.array(var).reshape(len(xstar),)
    else:
        var=np.dot(np.dot(phi_star, Ainv), phi_star.T)+noise2
        var=np.array(var).reshape(len(xstar),len(xstar))
    mu=np.array(mu).reshape(len(xstar),w_mN.shape[1])
    #print var,noise2
    return mu,var

def lnprior(theta, info):
    
    lnalpha, lnbeta = theta
    """
    if lnnoise2_lower < lnnoise2 < lnnoise2_upper and lnsigma2_lower < lnsigma2 < lnsigma2_upper:
        return 0.0
    return -np.inf
    """
    #lnalpha=np.log(1.0/np.exp(lnnoise2))
    #lnbeta=np.log(1.0/np.exp(lnsigma2))
    lp = 0
    lp += info.ln_prior_alpha.lnprob(lnalpha)
    lp += info.prior_noise2.lnprob(1.0/lnbeta)
    return lp
    
    """
    # sigma2
    if np.any(lnsigma2 == 0.0):
        return np.inf
    scale=0.1
    logp_sigma2=np.log(np.log(1 + 3.0 * (scale / np.exp(lnsigma2)) ** 2))
    
    #noise2
    mean=-10.0
    sigma=0.1
    logp_noise2=sps.lognorm.logpdf(lnnoise2, sigma, loc=mean)
    
    return logp_sigma2+logp_noise2
    """
    
#marginal_log_likelihood
def lnlik(theta,phi_matrix, y):
    lnalpha, lnbeta =theta
    
    n,k=phi_matrix.shape
    alpha=np.exp(lnalpha)
    beta=np.exp(lnbeta)
    A=beta * np.dot(phi_matrix.T, phi_matrix)
    A += np.eye(k) * alpha
    Ainv=np.linalg.inv(A)
    
    m=beta * np.dot(Ainv, phi_matrix.T)
    m = np.dot(m, y)
    
    #logp_=(k/2.0)*(np.log(1.0/sigma2))+(n/2.0)*(np.log(1.0/noise2))-(n/2.0)*(np.log(2*np.pi))-(0.5/noise2)*((y-phi_matrix.dot(m)).T.dot((y-phi_matrix.dot(m))))-(0.5/sigma2)*(m.T.dot(m))-0.5*(np.log(np.linalg.det(A)))
    mll = k / 2 * np.log(alpha)
    mll += n / 2 * np.log(beta)
    mll -= n / 2 * np.log(2 * np.pi)
    mll -= beta / 2. * np.linalg.norm(y - np.dot(phi_matrix, m), 2)
    mll -= alpha / 2. * np.dot(m.T, m)
    mll -= 0.5 * np.log(np.linalg.det(A))
        
    return mll[0,0]
def lnprob(theta, info, phi_matrix, y):
    
    if np.any((-5 > theta) + (theta > 10)):
        return -np.inf
    
    lp = lnprior(theta,info)
    
    """
    if not np.isfinite(lp):
        return -np.inf
    """
    lp = lp + lnlik(theta,phi_matrix, y)   
    if np.isnan(lp)==True or not np.isfinite(lp):
        return -np.inf
    return lp
def sampler(info,phi_matrix, y, sample_num=20):
    #nll = lambda *args: -lnlik(*args)
    #result = op.minimize(nll, [np.log(1e-3), np.log(1.0)], args=(phi_matrix, y))
    #lnnoise2, lnsigma2 = result["x"]
    
    ndim, nwalkers = 2, sample_num
    burnin_step=2000
    chain_length=2000
    #pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(info,phi_matrix, y))
    
    if info.iter==0:
        #Do a burn-in in the first iteration
        """
        lnlow=lnnoise2_lower
        lnup=lnnoise2_upper
        info.pos=info.rng.uniform(lnlow,lnup,size=(sample_num,ndim))
        """
        p0 = np.zeros([sample_num, ndim])
        p0[:, 0] = info.ln_prior_alpha.sample_from_prior(sample_num)[:, 0]

        # Noise sigma^2
        noise2 = info.prior_noise2.sample_from_prior(sample_num)[:, 0]
        # Beta
        p0[:, -1] = np.log(1.0 / np.exp(noise2))
        
        info.pos = p0
        """
        p0 = np.zeros([sample_num, ndim])
        #sampling log noise2 from prior
        scale=0.1
        lamda = np.abs(info.rng.standard_cauchy(size=sample_num))
        p0[:,-1] = np.log(1.0/np.exp(np.log(np.abs(info.rng.randn() * lamda * scale))[:, np.newaxis][:, 0]))
        #sampling log sigma2 from prior
        mean=-10.0
        sigma=0.1
        p0[:,0] = info.rng.lognormal(mean=mean,sigma=sigma,size=sample_num)[:, np.newaxis][:, 0]
        """
                
        info.pos, _, _ =sampler.run_mcmc(info.pos, burnin_step,rstate0=info.rng)
        
    info.pos, _, _ =sampler.run_mcmc(info.pos, chain_length,rstate0=info.rng)    
    samples = np.exp(sampler.chain[:, -1])
    #print "samples=%s"%samples
    return samples
        
        
        
