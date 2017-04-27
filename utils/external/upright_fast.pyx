# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 15:09:43 2016

@author: Chrisr Russell
"""

import cython
cimport cython
import scipy.optimize as op
from cython.view cimport array as cvarray
cimport numpy as np
import numpy as np
from libc.math cimport sqrt
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
import scipy.linalg
cimport scipy.linalg
from numpy.core.umath_tests import matrix_multiply

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)

def pick_e(np.ndarray[DTYPE_t, ndim=3] w,
           np.ndarray[DTYPE_t, ndim=4] e,
           np.ndarray[DTYPE_t, ndim=3] s0,
           np.ndarray[DTYPE_t, ndim=2] camera_r=np.asarray([[1,0,0],
                                                            [0,0,-1],
                                                            [0,1,0]]),
           np.ndarray[DTYPE_t, ndim=2] Lambda=np.ones((0,0)),
           np.ndarray[DTYPE_t, ndim=3] weights=np.ones((0,0,0)),
           DTYPE_t scale_prior=-0.0014,
           DTYPE_t interval=0.01,DTYPE_t depth_reg=0.0325):
    """Brute force over charts from the manifold to find the best one.
        Returns best chart index and its a and r coefficients
        Returns assignment, and a and r coefficents"""
    charts=e.shape[0]
    frames=w.shape[0]
    basis=e.shape[1]
    points=e.shape[3]
    assert(s0.shape[0]==charts)
    r=np.empty((charts,2,frames))
    a=np.empty((charts,frames,e.shape[1]))
    score=np.empty((charts,frames))
    check=np.arange(0,1,interval)*2*np.pi
    cache_a=np.empty((check.size,basis,frames))
    residue=np.empty((check.size,frames))

    if (Lambda.size!=0):
        res=np.zeros((frames,points*2+basis+points))
        proj_e=np.zeros((basis,2*points+basis+points))
    else:
        res=np.empty((frames,points*2))
        proj_e=np.empty((basis,2*points))
    Ps=np.empty((2,points))

    if weights.size==0:
        for i in xrange(charts):
            if Lambda.size!=0:
                a[i], r[i], score[i]=estimate_a_and_r_with_res(w,e[i],
                                                        s0[i],camera_r,Lambda[i],
                                                        check,cache_a,weights,
                                                        res,proj_e,residue,Ps,
                                                        depth_reg,scale_prior)
            else:
                a[i], r[i], score[i]=estimate_a_and_r_with_res(w,e[i],
                                                        s0[i],camera_r,Lambda,
                                                        check,cache_a,weights,
                                                        res,proj_e,residue,Ps,
                                                        depth_reg,scale_prior)
    else:
        w2=weights.reshape(weights.shape[0],-1)
        for i in xrange(charts):
            if Lambda.size!=0:
                a[i], r[i], score[i]=estimate_a_and_r_with_res_weights(w,e[i],
                                                        s0[i],camera_r,Lambda[i],
                                                        check,cache_a,w2,
                                                        res,proj_e,residue,Ps,
                                                        depth_reg,scale_prior)
            else:
                a[i], r[i], score[i]=estimate_a_and_r_with_res_weights(w,e[i],
                                                        s0[i],camera_r,Lambda,
                                                        check,cache_a,w2,
                                                        res,proj_e,residue,Ps,
                                                        depth_reg,scale_prior)

    remaining_dims=3*w.shape[2]-e.shape[1]
    assert(np.all(score>0))
    assert(remaining_dims>=0)
    #Zero problems in log space due to unregularised first co-efficient
    l=Lambda.copy()
    l[Lambda==0]=1
    llambda=-np.log(l)
    lgdet=np.sum(llambda[:,:-1],1)+llambda[:,-1]*remaining_dims
    score/=2
    return  score,a,r

cdef estimate_a_and_r_with_res(np.ndarray[DTYPE_t, ndim=3] w,
                               np.ndarray[DTYPE_t, ndim=3] e,
                               np.ndarray[DTYPE_t, ndim=2] s0,
                               np.ndarray[DTYPE_t, ndim=2] camera_r,
                               np.ndarray[DTYPE_t, ndim=1] Lambda,
                               np.ndarray[DTYPE_t, ndim=1] check,
                               np.ndarray[DTYPE_t, ndim=3] a,
                               np.ndarray[DTYPE_t, ndim=3] weights,
                               np.ndarray[DTYPE_t, ndim=2] res,
                               np.ndarray[DTYPE_t, ndim=2] proj_e,
                               np.ndarray[DTYPE_t, ndim=2] residue,
                               np.ndarray[DTYPE_t, ndim=2] Ps,
                               DTYPE_t depth_reg,
                               DTYPE_t scale_prior
                               ):
    """So local optima are a problem in general.
    However:

        1. This problem is convex in a but not in r, and

        2. each frame can be solved independently.

    So for each frame, we can do a grid search in r and take the globally
    optimal solution.

    In practice, we just brute force over 100 different estimates of r, and take
    the best pair (r,a*(r)) where a*(r) is the optimal minimiser of a given r.

    Arguments:

        w is a 3d measurement matrix of form frames*2*points

        e is a 3d set of basis vectors of from basis*3*points

        s0 is the 3d rest shape of form 3*points

        Lambda are the regularisor coefficients on the coefficients of the weights
        typically generated using PPCA

        interval is how far round the circle we should check for break points
        we check every interval*2*pi radians

    Returns:

        a (basis coefficients) and r (representation of rotations as a complex
        number)
    """
    frames=w.shape[0]
    points=w.shape[2]
    basis=e.shape[0]
    r=np.empty(2)
    Ps_reshape=Ps.reshape(2*points)
    w_reshape=w.reshape((frames,points*2))
    if (Lambda.size!=0):
         d=np.diag(Lambda[:Lambda.shape[0]-1])

    for i in xrange(check.size):
        c=check[i]
        r[0]=np.sin(c)
        r[1]=np.cos(c)
        grot=camera_r.dot(upgrade_r(r).T)
        rot=grot[:2]
        rot.dot(s0,Ps)
        res[:,:points*2]=w_reshape
        res[:,:points*2]-=Ps_reshape
        proj_e[:,:2*points]=rot.dot(e).transpose(1,0,2).reshape(e.shape[0],2*points)

        if (Lambda.size!=0):
            proj_e[:,2*points:2*points+basis]=d
            res[:,2*points:].fill(0)
            res[:,:points*2]*=Lambda[Lambda.shape[0]-1]
            proj_e[:,:points*2]*=Lambda[Lambda.shape[0]-1]
            proj_e[:,2*points+basis:]=((Lambda[Lambda.shape[0]-1] *
                                        depth_reg)*grot[2]).dot(e)
            res[:,2*points:].fill(0)
            res[:,2*points]=scale_prior
        if weights.size!=0:
            res[:,:points*2]*=weights
            proj_e[:,:points*2]*=weights
        a[i], residue[i], _, _ = scipy.linalg.lstsq(proj_e.T, res.T,
                                                             overwrite_a=True,
                                                             overwrite_b=True)
    #find and return best coresponding solution
    best=np.argmin(residue,0)
    assert(best.shape[0]==frames)
    theta=check[best]
    index=(best,np.arange(frames))
    aa=a.transpose(0,2,1)[index]
    retres=residue[index]
    r=np.empty((2,frames))
    r[0]=np.sin(theta)
    r[1]=np.cos(theta)
    return aa,r,retres

cdef estimate_a_and_r_with_res_weights(np.ndarray[DTYPE_t, ndim=3] w,
                     np.ndarray[DTYPE_t, ndim=3] e,
                     np.ndarray[DTYPE_t, ndim=2] s0,
                     np.ndarray[DTYPE_t, ndim=2] camera_r,
                     np.ndarray[DTYPE_t, ndim=1] Lambda,
                     np.ndarray[DTYPE_t, ndim=1] check,
                     np.ndarray[DTYPE_t, ndim=3] a,
                     np.ndarray[DTYPE_t, ndim=2] weights,
                     np.ndarray[DTYPE_t, ndim=2] res,
                     np.ndarray[DTYPE_t, ndim=2] proj_e,
                     np.ndarray[DTYPE_t, ndim=2] residue,
                     np.ndarray[DTYPE_t, ndim=2] Ps,
                     DTYPE_t depth_reg,
                     DTYPE_t scale_prior
                     ):
    """Rather than perform global optimisation, marginalise over R
    Arguments:

        w is a 3d measurement matrix of form frames*2*points

        e is a 3d set of basis vectors of from basis*3*points

        s0 is the 3d rest shape of form 3*points

        Lambda are the regularisor coefficients on the coefficients of the weights
        typically generated using PPCA

        interval is how far round the circle we should check for break points
        we check every interval*2*pi radians

    Returns:

        a (basis coefficients) and r (representation of rotations as a complex
        number)
    """
    frames=w.shape[0]
    points=w.shape[2]
    basis=e.shape[0]
    r=np.empty(2)
    Ps_reshape=Ps.reshape(2*points)
    w_reshape=w.reshape((frames,points*2))
    p_copy=np.empty_like(proj_e)
    if (Lambda.size!=0):
         d=np.diag(Lambda[:Lambda.shape[0]-1])

    for i in xrange(check.size):
        c=check[i]
        r[0]=np.sin(c)
        r[1]=np.cos(c)
        grot=camera_r.dot(upgrade_r(r).T)
        rot=grot[:2]
        rot.dot(s0,Ps)
        res[:,:points*2]=w_reshape
        res[:,:points*2]-=Ps_reshape
        proj_e[:,:2*points]=rot.dot(e).transpose(1,0,2).reshape(e.shape[0],2*points)

        if (Lambda.size!=0):
            proj_e[:,2*points:2*points+basis]=d
            res[:,2*points:].fill(0)
            res[:,:points*2]*=Lambda[Lambda.shape[0]-1]
            proj_e[:,:points*2]*=Lambda[Lambda.shape[0]-1]
            proj_e[:,2*points+basis:]=((Lambda[Lambda.shape[0]-1] *
                                        depth_reg)*grot[2]).dot(e)
            res[:,2*points:].fill(0)
            res[:,2*points]=scale_prior
        if weights.size!=0:
            res[:,:points*2]*=weights
        for j in xrange(frames):
            p_copy[:]=proj_e
            p_copy[:,:points*2]*=weights[j]
            a[i,:,j], residue[i,j], _, _ = np.linalg.lstsq(p_copy.T, res[j].T)
    #find and return best coresponding solution
    best=np.argmin(residue,0)
    index=(best, np.arange(frames))
    theta=check[best]
    aa=a.transpose(0,2,1)[index]
    retres=residue[index]
    r=np.empty((2,frames))
    r[0]=np.sin(theta)
    r[1]=np.cos(theta)
    return aa,r,retres

cdef upgrade_r(np.ndarray[DTYPE_t, ndim=1] r):
    """Upgrades complex parameterisation of planar rotation to tensor containing
    per frame 3x3 rotation matrices"""
    newr = np.zeros((3, 3))
    newr[:2, 0] = r
    newr[2, 2] = 1
    newr[1::-1, 1] = r
    newr[0, 1] *= -1
    return newr
