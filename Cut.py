import pyqtgraph as pg #graph
import numpy as np #math library
import pandas as pd #pandas data
import os #pathname manipulations
from Plot import *
import numpy.ma as ma # mask for arrays 
def cut(self):
        
    ###### first check to see if cutting############
    if self.lr==[]:
        ######## if no cutting window exists, make one##########
        self.lr = pg.LinearRegionItem()
        self.lr.hide()
        self.p1.addItem(self.lr)
        self.lr.show()
    #### if cut region has been set, cut region and replot remaining data####
    else:
        cutregion = self.lr.getRegion()
        #ma.masked_inside(x, cutregion[0], cutregion[1])
        a=timer()
        mask =  np.logical_not(np.logical_and(self.t > cutregion[0], self.t < cutregion[1]))
        mask[0] = False
        self.data['i1'] = self.data['i1'][mask]
        try:
            self.data['v1'] = self.data['v1'][mask]
        except IndexError:
            pass        
#self.data['i1'] = np.delete(self.data['i1'],np.arange(np.int(cutregion[0]*self.outputsamplerate),np.int(cutregion[1]*self.outputsamplerate)))
        #self.data['v1'] = np.delete(self.data['v1'],np.arange(np.int(cutregion[0]*self.outputsamplerate),np.int(cutregion[1]*self.outputsamplerate)))
        
        try: # execute if exists
            self.data['i2'] = self.data['i2'][mask]
            self.data['v2'] = self.data['v2'][mask]
        except KeyError: 
            pass
        
        self.t = self.t[mask]
        self.p1.clear()
        #self.p1.plot(self.t,self.data['i1'],pen='b', symbol = 'o')
        self.lr=[]
        self.p3.clear()
        #aphy, aphx = np.histogram(self.data, bins=int(np.round(len(self.data) / 1000)))
        #aphhist = pg.PlotCurveItem(aphx, aphy, stepMode=True, fillLevel=0, brush='b')
        #self.p3.addItem(aphhist) 
        self.CuttingApplied = True
        if not self.ui.plotBoth.isChecked():
            if self.ui.ndChannel.isChecked():
                self.sig = 'i2'
                self.sig2 = 'i1'
            else:
                self.sig = 'i1'
                self.sig2 = 'i2'
        
        if not self.ui.actionDon_t_Plot_if_slow.isChecked():
            if self.ui.plotBoth.isChecked():
                DoublePlot(self)
            else:
                PlotSingle(self)  
     
