#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.datasets import make_moons, make_circles
def generate_normal_PU_tr_data(pi_plus, n_p, n_u, **kwargs):
    #Generate normal training dataset for PU
    #Result: in dictionary {X, PU_labels, PN_labels} for experimental purpose.
    #X: independent variables
    #PU_labels: 0 for unlabled and 1 for labeled
    #PN_labels: -1 for true negative and 1 for true positive
    
    # Parameters to generate
    mu_p = kwargs.get('mu_p', np.array([1,1])/np.sqrt(2) )
    mu_n = kwargs.get('mu_n', np.array([-1,-1])/np.sqrt(2) )
    cov_p = kwargs.get('cov_p', np.eye(2) )
    cov_n = kwargs.get('cov_n', np.eye(2) )
    
    # Generate dataset
    X_p_tr = np.random.multivariate_normal(mu_p, cov_p, n_p)
    Y_u = np.sort(np.random.choice([-1,1], n_u, p=[1-pi_plus,pi_plus])) 
    X_u_tr = np.vstack((np.random.multivariate_normal(mu_n, cov_n, np.count_nonzero(Y_u==-1)),
                        np.random.multivariate_normal(mu_p, cov_p, np.count_nonzero(Y_u==1)))
                      )
    PU_label = np.hstack((np.ones(n_p),np.zeros(n_u)))
    PN_label = np.hstack((np.ones(n_p),Y_u))
    X_PU_tr = np.vstack((X_p_tr, X_u_tr))
    PU_tr = np.hstack((X_PU_tr, PU_label.reshape(-1,1), PN_label.reshape(-1,1))).astype('float32')
    
    #shuffle row only
    np.random.shuffle(PU_tr)
    
    #make it to dictionary
    return {'X':PU_tr[:,:-2], 'PU_label':PU_tr[:,-2], 'PN_label':PU_tr[:,-1]}


def generate_normal_te_data(pi_plus, n_te, **kwargs):
    #Generate normal test set
    #Result: in dictionary {X, PN_labels}
    #X: independent variables
    #PN_labels: -1 for true negative and 1 for true positive
    
    # Parameters to generate
    mu_p = kwargs.get('mu_p', np.array([1,1])/np.sqrt(2) )
    mu_n = kwargs.get('mu_n', np.array([-1,-1])/np.sqrt(2) )
    cov_p = kwargs.get('cov_p', np.eye(2) )
    cov_n = kwargs.get('cov_n', np.eye(2) )

    # Generate datasets
    Y_te = np.sort(np.random.choice([-1,1], n_te, p=[1-pi_plus,pi_plus]))
    X_te = np.concatenate((np.random.multivariate_normal(mu_n, cov_n, np.count_nonzero(Y_te==-1)),
                              np.random.multivariate_normal(mu_p, cov_p, np.count_nonzero(Y_te==1))),
                             axis=0)
    te = np.hstack((X_te, Y_te.reshape(-1,1))).astype('float32')
    
    #shuffle row only
    np.random.shuffle(te) 
    
    #make it to dictionary
    return {'X':te[:,:-1], 'PN_label':te[:,-1]}


def generate_moons_PU_tr_data(pi_plus, n_p, n_u, **kwargs):
    #Generate the two moons training dataset for PU
    #Result in dictionary {X, PU_labels, PN_labels} for experimental purpose.
    #X: independent variables
    #PU_labels: 0 for unlabled and 1 for labeled
    #PN_labels: -1 for true negative and 1 for true positive
    
    # Parameters to generate
    noise = kwargs.get('noise', 0.1)
    
    # Generate dataset
    X_p_tr = 4*make_moons(2*n_p, noise=0.1, shuffle=False)[0][n_p:2*n_p]
    generated_unlabeled = make_moons(n_u, noise=0.1)
    X_u_tr = 4*generated_unlabeled[0]
    PU_label = np.hstack((np.ones(n_p),np.zeros(n_u)))
    PN_label = np.hstack((np.ones(n_p),2*generated_unlabeled[1]-1))
    X_PU_tr = np.vstack((X_p_tr, X_u_tr))
    PU_tr = np.hstack((X_PU_tr, PU_label.reshape(-1,1), PN_label.reshape(-1,1))).astype('float32')
    
    #shuffle row only
    np.random.shuffle(PU_tr)
    
    #make it to dictionary
    return {'X':PU_tr[:,:-2], 'PU_label':PU_tr[:,-2], 'PN_label':PU_tr[:,-1]}


def generate_circles_PU_tr_data(pi_plus, n_p, n_u, **kwargs):
    #Generate the two circles training dataset for PU
    #Result in dictionary {X, PU_labels, PN_labels} for experimental purpose.
    #X: independent variables
    #PU_labels: 0 for unlabled and 1 for labeled
    #PN_labels: -1 for true negative and 1 for true positive
    
    # Parameters to generate
    noise = kwargs.get('noise', 0.1)
    factor = kwargs.get('factor', 0.5)
    
    # Generate dataset
    X_p_tr = 4*make_circles(2*n_p, noise=0.1, factor= 0.5, shuffle=False)[0][n_p:2*n_p]
    generated_unlabeled = make_circles(n_u, noise=0.1, factor=0.5)
    X_u_tr = 4*generated_unlabeled[0]
    PU_label = np.hstack((np.ones(n_p),np.zeros(n_u)))
    PN_label = np.hstack((np.ones(n_p),2*generated_unlabeled[1]-1))
    X_PU_tr = np.vstack((X_p_tr, X_u_tr))
    PU_tr = np.hstack((X_PU_tr, PU_label.reshape(-1,1), PN_label.reshape(-1,1))).astype('float32')
    
    #shuffle row only
    np.random.shuffle(PU_tr)
    
    #make it to dictionary
    return {'X':PU_tr[:,:-2], 'PU_label':PU_tr[:,-2], 'PN_label':PU_tr[:,-1]}
