#!/usr/bin/env python

import csv

from numpy import array
from scipy import stats
import numpy as np
from scipy.optimize import curve_fit


class UncerQuad():

    def __init__(self,Y,path='',par_list=()):

        if path!='' :
            f = open(path, 'rt')
            reader = csv.reader(f)
            l=list(reader)
            f.close()
        else :
            l=par_list

        self.params, self.std_err, self.p_value, self.cov_mtx, self.rss, self.rmse, self.dof = par_list

        self.c=self.params[2]
        self.alpha=self.params[1]
        self.beta=self.params[0]
        self.varc=self.std_err[2]*self.std_err[2]
        self.vara=self.std_err[1]*self.std_err[1]
        self.varb=self.std_err[0]*self.std_err[0]
        self.vary = 0.0
        self.covab = self.cov_mtx[1][0]
        self.covca = self.cov_mtx[2][1]
        self.covcb = self.cov_mtx[2][0]

        # We have a given yield:
        self.Y = Y
        # We derive the dose from the calibration curve:
        self.D = self.inv(self.Y)
        
    def inv(self, Y) :
        D = ( -self.alpha + np.sqrt( self.alpha*self.alpha + 4.0*self.beta*(Y-self.c) ) )/2.0/self.beta
        return D

    def dYdc(self) :
        return 1.0

    def dYda(self) :
        return self.D

    def dYdb(self) :
        return self.D*self.D

    def dDdc(self) :
        return 4.0 / ( np.sqrt( self.alpha*self.alpha + 4.0*self.beta*(self.Y-self.c) ) )

    def dDda(self) :
        return -2.0/self.beta + 2.0*self.alpha / ( self.beta*np.sqrt( self.alpha*self.alpha + 4.0*self.beta*(self.Y-self.c) ) )

    def dDdb(self) :
        return 4.0*(self.Y-self.c) / ( self.beta*np.sqrt( self.alpha*self.alpha + 4.0*self.beta*(self.Y-self.c) ) ) - 2*(np.sqrt( self.alpha*self.alpha + 4.0*self.beta*(self.Y-self.c))-self.alpha)/self.beta/self.beta

    def dDdY(self) :
        return -4.0 / ( np.sqrt( self.alpha*self.alpha + 4.0*self.beta*(self.Y-self.c) ) )

    def varY(self) :
        if self.dof==0.0: self.dof=50.0
        return self.Y/self.dof

    def ufitY(self) :
        return self.dYdc()*np.sqrt(self.varc) + self.dYda()*np.sqrt(self.vara) + self.dYdb()*np.sqrt(self.varb)

    def varD(self) :
        C = pow(self.dDdc(),2) * self.varc
        A = pow(self.dDda(),2) * self.vara
        B = pow(self.dDdb(),2) * self.varb
        _Y = pow(self.dDdY(),2) * self.vary
        AB = 2 * self.dDda() * self.dDda() * self.covab
        CA = 2 * self.dDdc() * self.dDda() * self.covca
        CB = 2 * self.dDdc() * self.dDdb() * self.covcb
        return C + A + B + _Y + AB + CA + CB

    def method_a(self) :
        DL = self.D - 1.96*np.sqrt(self.varD())
        DU = self.D + 1.96*np.sqrt(self.varD())
        return DL, DU

    def method_b(self) :
        uY = np.sqrt( pow(self.ufitY(),2) + self.varY() )
        YL = self.Y - uY
        YU = self.Y + uY
        DL = self.inv(YL)
        DU = self.inv(YU)
        return DL, DU

    def method_c1(self) :
        uY = np.sqrt( pow(self.ufitY(),2) + self.varY() )
        YL = self.Y - 1.96*np.sqrt(self.varY())
        YU = self.Y + 1.96*np.sqrt(self.varY())
        DL = self.inv(max(0,YL - uY))
        DU = self.inv(YU + uY)
        return DL, DU

    def method_c2(self) :
        YL = self.Y - 1.96*np.sqrt(self.varY())
        YU = self.Y + 1.96*np.sqrt(self.varY())
        DL = self.inv(YL)
        DU = self.inv(YU)
        return DL, DU

class UncerLin():

    def __init__(self,Y,path='',par_list=()):

        if path!='' :
            f = open(path, 'rt')
            reader = csv.reader(f)
            l=list(reader)
            f.close()
        else :
            l=par_list

        self.params, self.std_err, self.p_value, self.cov_mtx, self.rss, self.rmse, self.dof = par_list

        self.c=self.params[1]
        self.alpha=self.params[0]
        self.varc=self.std_err[1]*self.std_err[1]
        self.vara=self.std_err[0]*self.std_err[0]
        self.vary = 0.0
        self.covca = self.cov_mtx[1][0]

        # We have a given yield:
        self.Y = Y
        # We derive the dose from the calibration curve:
        self.D = self.inv(self.Y)
        
    def inv(self, Y) :
        D = (Y - self.c)/self.alpha
        return D

    def dYdc(self) :
        return 1.0

    def dYda(self) :
        return self.D

    def dDdc(self) :
        return -1.0/self.alpha

    def dDda(self) :
        return (self.Y-self.c)*(-1.0)*(1/self.alpha*self.alpha)

    def dDdY(self) :
        return 1./self.alpha

    def varY(self) :
        if self.dof==0.0: self.dof=50.0
        return self.Y/self.dof

    def ufitY(self) :
        return self.dYdc()*np.sqrt(self.varc) + self.dYda()*np.sqrt(self.vara)

    def varD(self) :
        C = pow(self.dDdc(),2) * self.varc
        A = pow(self.dDda(),2) * self.vara
        _Y = pow(self.dDdY(),2) * self.vary
        CA = 2 * self.dDdc() * self.dDda() * self.covca
        return C + A + _Y + CA

    def method_a(self) :
        DL = self.D - 1.96*np.sqrt(self.varD())
        DU = self.D + 1.96*np.sqrt(self.varD())
        return DL, DU

    def method_b(self) :
        uY = np.sqrt( pow(self.ufitY(),2) + self.varY() )
        YL = self.Y - uY
        YU = self.Y + uY
        DL = self.inv(YL)
        DU = self.inv(YU)
        return DL, DU

    def method_c1(self) :
        uY = np.sqrt( pow(self.ufitY(),2) + self.varY() )
        YL = self.Y - 1.96*np.sqrt(self.varY())
        YU = self.Y + 1.96*np.sqrt(self.varY())
        DL = self.inv(max(0,YL - uY))
        DU = self.inv(YU + uY)
        return DL, DU

    def method_c2(self) :
        YL = self.Y - 1.96*np.sqrt(self.varY())
        YU = self.Y + 1.96*np.sqrt(self.varY())
        DL = self.inv(YL)
        DU = self.inv(YU)
        return DL, DU

