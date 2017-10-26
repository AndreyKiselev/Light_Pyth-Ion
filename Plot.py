import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from UsefulFunctions import MakePSD
from timeit import default_timer as timer
def PlotSingle(self):
    self.p1.clear() #clear signal plot
    self.p3.clear() #clear event plot
    self.voltagepl.clear() #clear voltage plot
    self.transverseAxis.clear() #clear axis
    self.transverseAxisVoltage.clear() #clear axis
    self.p1.plotItem.hideAxis('right') #clear axis for transvers current (2nd channel)
    self.voltagepl.plotItem.hideAxis('right') #clear axis for transvers voltage (2nd channel)
    #self.m = timer()
    #print('m-b='+str(self.m - self.b) + 's')

    if self.ui.ndChannel.isChecked():
        temp_i = self.data['i2'] #channel2 for current
        temp_v = self.data['v2'] #channel2 for voltage
        self.p1.setLabel('left', text='Transverse Current', units='A') #setting current as in-plane current
        self.voltagepl.setLabel('left', text='Transverse Voltage', units='V') #setting voltage
    else:        
        temp_i = np.array(self.data['i1']) #channel1 for current
        temp_v = self.data['v1']  #channel1 for voltage
        self.p1.setLabel('left', text='Ionic Current', units='A') #setting current as perpendicular to the plane current
        self.voltagepl.setLabel('left', text='Ionic Voltage', units='V') #setting voltage

    self.p1.setLabel('bottom', text='Time', units='s') #setting time axis for current
    self.voltagepl.setLabel('bottom', text='Time', units='s') #setting time axis for voltage
    self.p1.plot(self.t, temp_i, pen='b') #plotting current plot
    

    if self.out['type'] == 'ChimeraRaw':
        self.voltagepl.addLine(y=self.data['v1'], pen='b')
    else:
        self.voltagepl.plot(self.t, temp_v, pen='b')
    if not self.CuttingApplied:
        aphy, aphx = np.histogram(temp_i, bins=int(np.round(len(temp_i) / 1000)))

        aphhist = pg.PlotCurveItem(aphx, aphy, stepMode=True, fillLevel=0, brush='b')
        self.p3.addItem(aphhist)
        self.psdplot.clear()
        MakePSD(temp_i, self.out['samplerate'], self.psdplot)
        siSamplerate = pg.siScale(self.out['samplerate'])
        siSTD = pg.siScale(np.std(temp_i))

        self.ui.SampleRateLabel.setText('Samplerate: ' + str(self.out['samplerate'] * siSamplerate[0]) + siSamplerate[1] + 'Hz')
        self.ui.STDLabel.setText('STD: ' + str(siSTD[0] * np.std(temp_i)) + siSTD[1] + 'A')
    
    self.voltagepl.enableAutoRange(axis='y')
    self.p1.enableAutoRange(axis='y')
    
def DoublePlot(self):
    self.p1.clear()
    self.transverseAxis.clear() #clear axis
    self.transverseAxisVoltage.clear() #clear axis
    p1 = self.p1.plotItem
    p1_v = self.voltagepl.plotItem
    p1.getAxis('left').setLabel(text='Ionic Current', color='#0000FF', units='A')
    p1_v.getAxis('left').setLabel(text='Ionic Voltage', color='#0000FF', units='V')
    ## create a new ViewBox, link the right axis to its coordinate system
    p1.showAxis('right')
    p1_v.showAxis('right')
    p1.scene().addItem(self.transverseAxis)
    p1_v.scene().addItem(self.transverseAxisVoltage)
    p1.getAxis('right').linkToView(self.transverseAxis)
    p1_v.getAxis('right').linkToView(self.transverseAxisVoltage)
    self.transverseAxis.setXLink(p1)
    self.transverseAxisVoltage.setXLink(p1_v)
    self.transverseAxis.show()
    self.transverseAxisVoltage.show()
    p1.getAxis('right').setLabel(text='Transverse Current', color='#FF0000', units='A')
    p1_v.getAxis('right').setLabel(text='Transverse Voltage', color='#FF0000', units='V')

    def updateViews():
        ## view has resized; update auxiliary views to match
        self.transverseAxis.setGeometry(p1.vb.sceneBoundingRect())
        self.transverseAxisVoltage.linkedViewChanged(p1_v.vb, self.transverseAxisVoltage.XAxis)
        self.transverseAxisVoltage.setGeometry(p1_v.vb.sceneBoundingRect())
        self.transverseAxis.linkedViewChanged(p1.vb, self.transverseAxis.XAxis)

    updateViews()
    p1.vb.sigResized.connect(updateViews)
    p1_v.vb.sigResized.connect(updateViews)
    p1.plot(self.t, self.data['i1'], pen='b')
    p1_v.plot(self.t, self.data['v1'], pen='b')
    self.transverseAxis.addItem(pg.PlotCurveItem(self.t, self.data['i2'], pen='r'))
    self.transverseAxisVoltage.addItem(pg.PlotCurveItem(self.t, self.data['v2'], pen='r'))

    #Set Ranges
    #self.transverseAxis.setYRange(np.min(self.data['i2'])-10*np.std(self.data['i2']), np.max(self.data['i2'])+1*np.std(self.data['i2']))
    #self.p1.setYRange(np.min(self.data['i1'])-1*np.std(self.data['i1']), np.max(self.data['i1'])+ 10*np.std(self.data['i1']))
    self.p1.enableAutoRange(axis='x')
    #self.transverseAxisVoltage.setYRange(np.min(self.data['v2']) - 10/100 * np.mean(self.data['v2']),
    #                          np.max(self.data['v2']) + 1/100 * np.mean(self.data['v2']))
    #self.voltagepl.setYRange(np.min(self.data['v1']) - 1/100 * np.mean(self.data['v1']),
    #              np.max(self.data['v1']) + 10/100 * np.mean(self.data['v1']))

