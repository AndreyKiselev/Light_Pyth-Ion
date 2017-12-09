import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import scipy


import pickle as pkl

from PyQt5 import QtGui, QtWidgets
from numpy import linalg as lin #import linear algebra
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import h5py
from timeit import default_timer as timer
import platform
from scipy.optimize import curve_fit #fitting of data to function
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes # zooming for plots
from mpl_toolkits.axes_grid1.inset_locator import mark_inset # zooming for plots








def MakePSD(input, samplerate, fig):
    f, Pxx_den = scipy.signal.periodogram(input, samplerate)
    #f, Pxx_den = scipy.signal.welch(input, samplerate, nperseg=10*256, scaling='spectrum')
    fig.setLabel('left', 'PSD', units='pA^2/Hz')
    fig.setLabel('bottom', 'Frequency', units='Hz')
    f = np.delete(f,0)
    Pxx_den = np.delete(Pxx_den,0)
    fig.setLogMode(x=True, y=True)
    fig.plot(f, Pxx_den*1e24, pen='k')
    return (f,Pxx_den)

def PrintEps(self):

 
    fig, ax = plt.subplots(figsize=(20, 20))
    #plt.title('Signal plot', fontsize = 20)
    #plt.figure(figsize=(15,15))
    ax.plot(self.t[self.Search_start_points:self.Search_end_points:50], 1e9 * self.out['i1'][self.Search_start_points:self.Search_end_points:50])
    ax.set_ylim(-100.0,100.0)
    ax.set_xlabel('Time (s)', size=30)
    ax.set_ylabel('Ionic Current (nA)', size=30)
    axins = zoomed_inset_axes(ax, 20.0, loc=2)
    
    self.control1 = int(self.zero_points[0] -self.Delay_back_trace_points)/ self.out['samplerate']
    self.control2 = int(self.zero_points[0]) / self.out['samplerate']
    self.control3 = (self.zero_points[0] -self.Delay_back_trace_points + self.Safety_region_points)  / self.out['samplerate']
    axins.plot(self.t[self.Search_start_points:int(self.zero_points[0] -self.Delay_back_trace_points + self.Safety_region_points) :10], 1e9 * self.out['i1'][self.Search_start_points:int(self.zero_points[0] -self.Delay_back_trace_points + self.Safety_region_points) :10], 'b')
    axins.plot(self.t[int(self.zero_points[0] -self.Delay_back_trace_points + self.Safety_region_points):int(self.zero_points[0]):10], 1e9 * self.out['i1'][int(self.zero_points[0] -self.Delay_back_trace_points + self.Safety_region_points):int(self.zero_points[0]) :10], 'r')
    plt.axvline(self.t[int(self.zero_points[0] -self.Delay_back_trace_points + self.Safety_region_points)], color='black')
    #axins.plot(self.t[self.Search_start_points:self.Search_start_points+self.Safety_region_points:10], 1e9 * self.out['i1'][self.Search_start_points:self.Search_start_points+self.Safety_region_points:10], 'r')
    x1, x2, y1, y2 = self.control1, 16.59, 3.0, 7.0 # specify the limits
    axins.set_xlim(x1, x2) # apply the x-limits
    axins.set_ylim(y1, y2) # apply the y-limits
    axins.text(16.55, 4.5, 'Delay back trace', fontsize=20)
    axins.annotate('',
            xy=(self.control1, 4.2), xycoords='data',
            xytext=(self.control2, 4.2), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3"), 
            )
    axins.text(16.55, 5.3, 'Safety region', fontsize=20)
    axins.annotate('',
            xy=(self.control1, 5.0), xycoords='data',
            xytext=(self.control3, 5.0), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3"), 
            )
    axins.text(16.58, 5.0, 'Level threshhold', fontsize=20)
    axins.annotate('',
            xy=(16.58, 3.5), xycoords='data',
            xytext=(16.58, 6.0), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3"), 
            )
    
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    fig1 = plt.gcf()
    fig1.savefig('Ionic.eps')
    #plt.show()
    '''
    plt.figure(figsize=(15,15))
    plt.clf()
    plt.title('Signal plot', fontsize = 20)
    plt.xlabel('Time (s)', fontsize = 20)
    plt.ylabel('Ionic Current (nA)', fontsize = 20)
    plt.plot(self.t[self.Search_start_points:self.Search_end_points], 1e9 * self.out['i1'][self.Search_start_points:self.Search_end_points])

    for n in np.arange(0,len(self.zero_points),1): #select regions for future CUSUM analysis
            self.control1 = int(self.zero_points[n] -self.Delay_back_trace_points)
            self.safety_reg = int(self.zero_points[n] -self.Delay_back_trace_points + self.Safety_region_points) 
            self.control2 =  int(self.zero_points[n])
            trace = self.out['i1'][self.control1:self.control2]
            plt.plot(self.t[self.control1:self.safety_reg], 1e9 * self.out['i1'][self.control1:self.safety_reg], 'g')
            plt.plot(self.t[self.safety_reg:self.control2], 1e9 * self.out['i1'][self.safety_reg:self.control2], 'r')
    plt.axis([16.0, 18.0, -50, 50])
    plt.draw()
    plt.show()
    '''
def CutDataIntoVoltageSegments(output, x, y, extractedSegments, delay=0.7, plotSegments = 1):
    sweepedChannel = ''
    if output['type'] == 'ChimeraNotRaw':
        current = output['i1']
        voltage = output['v1']
        samplerate = output['samplerate']
    elif output['type'] == 'Axopatch' and x == 'i1' and y == 'v1':
        current = output['i1']
        voltage = output['v1']
        print('i1,v1')
        samplerate = output['samplerate']
    elif output['type'] == 'Axopatch' and output['graphene'] and x == 'i2' and y == 'v2':
        current = output['i2']
        voltage = output['v2']
        samplerate = output['samplerate']
        print('i2,v2')
    elif output['type'] == 'Axopatch' and output['graphene'] and x == 'i2' and y == 'v1':
        current = output['i2']
        voltage = output['v1']
        samplerate = output['samplerate']
        print('i2,v1')
    elif output['type'] == 'Axopatch' and output['graphene'] and x == 'i1' and y == 'v2':
        current = output['i1']
        voltage = output['v2']
        samplerate = output['samplerate']
        print('i1,v2')
    else:
        print('File doesn''t contain any IV data on the selected channel...')
        return (0, 0)

    time=np.float32(np.arange(0, len(current))/samplerate)
    delayinpoints = int(delay * samplerate)
    diffVoltages = np.diff(voltage)
    VoltageChangeIndexes = diffVoltages
    ChangePoints = np.where(diffVoltages)[0]
    Values = voltage[ChangePoints]
    Values = np.append(Values, voltage[-1])
    print('Cutting into Segments\n{} change points detected...'.format(len(ChangePoints)))
    if len(ChangePoints) is 0:
        print('Can\'t segment the file. It doesn\'t contain any voltage switches')
        return (0,0)

    #   Store All Data
    AllDataList = []
    # First
    Item={}
    Item['Voltage'] = Values[0]
    Item['CurrentTrace'] = current[0:ChangePoints[0]]
    AllDataList.append(Item)
    for i in range(1, len(Values) - 1):
        Item={}
        Item['CurrentTrace'] = current[ChangePoints[i - 1] + delayinpoints:ChangePoints[i]]
        Item['Voltage']=Values[i]
        AllDataList.append(Item)
    # Last
    Item = {}
    Item['CurrentTrace'] = current[ChangePoints[len(ChangePoints) - 1] + delayinpoints:len(current) - 1]
    Item['Voltage']= Values[len(Values) - 1]
    AllDataList.append(Item)
    if plotSegments:

        #extractedSegments = pg.PlotWidget(title="Extracted Parts")
        extractedSegments.plot(time, current, pen='b', linewidth = 2000)
        extractedSegments.setLabel('left', text='Current', units='A')
        extractedSegments.setLabel('bottom', text='Time', units='s')
        # First
        extractedSegments.plot(np.arange(0,ChangePoints[0])/samplerate, current[0:ChangePoints[0]], pen='r')
        #Loop
        for i in range(1, len(Values) - 1):
            extractedSegments.plot(np.arange(ChangePoints[i - 1] + delayinpoints, ChangePoints[i]) / samplerate, current[ChangePoints[i - 1] + delayinpoints:ChangePoints[i]], pen='r')
        #Last
            extractedSegments.plot(np.arange(ChangePoints[len(ChangePoints) - 1] + delayinpoints, len(current) - 1 )/samplerate, current[ChangePoints[len(ChangePoints) - 1] + delayinpoints:len(current) - 1], pen='r')
    else:
        extractedSegments=0
    return (AllDataList, extractedSegments)

def MakeIV(CutData, plot=1):
    l=len(CutData)
    IVData={}
    IVData['Voltage'] = np.zeros(l)
    IVData['Mean'] = np.zeros(l)
    IVData['STD'] = np.zeros(l)
    count=0
    for i in CutData:
        #print('Voltage: ' + str(i['Voltage']) + ', length: ' + str(len(i['CurrentTrace'])))
        IVData['Voltage'][count] = np.float32(i['Voltage'])
        IVData['Mean'][count] = np.mean(i['CurrentTrace'])
        IVData['STD'][count] = np.std(i['CurrentTrace'])
        count+=1
    if plot:
        spacing=np.sort(IVData['Voltage'])
        iv = pg.PlotWidget(title='Current-Voltage Plot')
        err = pg.ErrorBarItem(x=IVData['Voltage'], y=IVData['Mean'], top=IVData['STD'], bottom=IVData['STD'], beam=((spacing[1]-spacing[0]))/2)
        iv.addItem(err)
        iv.plot(IVData['Voltage'], IVData['Mean'], symbol='o', pen=None)
        iv.setLabel('left', text='Current', units='A')
        iv.setLabel('bottom', text='Voltage', units='V')
        print('raboraet')
    else:
        iv=0
    return (IVData, iv)


def ExpFunc(x, a, b, c):
    return a * np.exp(-b * x) + c

def MakeExponentialFit(xdata,ydata):
    try:
        popt, pcov = curve_fit(ExpFunc, xdata, ydata)
        return (popt, pcov)
    except RuntimeError:
        popt = (0,0,0)
        pcov = 0
        return (popt, pcov)

def YorkFit(X, Y, sigma_X, sigma_Y, r=0):
    N_itermax=10 #maximum number of interations
    tol=1e-15 #relative tolerance to stop at
    N = len(X)
    temp = np.matrix([X, np.ones(N)])
    #make initial guess at b using linear squares

    tmp = np.matrix(Y)*lin.pinv(temp)
    b_lse = np.array(tmp)[0][0]
    #a_lse=tmp(2);
    b = b_lse #initial guess
    omega_X = np.true_divide(1,np.power(sigma_X,2))
    omega_Y = np.true_divide(1, np.power(sigma_Y,2))
    alpha=np.sqrt(omega_X*omega_Y)
    b_save = np.zeros(N_itermax+1) #vector to save b iterations in
    b_save[0]=b

    for i in np.arange(N_itermax):
        W=omega_X*omega_Y/(omega_X+b*b*omega_Y-2*b*r*alpha)

        X_bar=np.sum(W*X)/np.sum(W)
        Y_bar=np.sum(W*Y)/np.sum(W)

        U=X-X_bar
        V=Y-Y_bar

        beta=W*(U/omega_Y+b*V/omega_X-(b*U+V)*r/alpha)

        b=sum(W*beta*V)/sum(W*beta*U)
        b_save[i+1]=b
        if np.abs((b_save[i+1]-b_save[i])/b_save[i+1]) < tol:
            break

    a=Y_bar-b*X_bar
    x=X_bar+beta
    y=Y_bar+b*beta
    x_bar=sum(W*x)/sum(W)
    y_bar=sum(W*y)/sum(W)
    u=x-x_bar
    #%v=y-y_bar
    sigma_b=np.sqrt(1/sum(W*u*u))
    sigma_a=np.sqrt(1./sum(W)+x_bar*x_bar*sigma_b*sigma_b)
    return (a, b, sigma_a, sigma_b, b_save)



def FitIV(IVData, plot=1, x='i1', y='v1', iv=0):
    sigma_v = 1e-12*np.ones(len(IVData['Voltage']))
    (a, b, sigma_a, sigma_b, b_save) = YorkFit(IVData['Voltage'], IVData['Mean'], sigma_v, IVData['STD'])
    x_fit = np.linspace(min(IVData['Voltage']), max(IVData['Voltage']), 1000)
    y_fit = scipy.polyval([b,a], x_fit)
    if plot:
        spacing = np.sort(IVData['Voltage'])
        #iv = pg.PlotWidget(title='Current-Voltage Plot', background=None)
        err = pg.ErrorBarItem(x=IVData['Voltage'], y=IVData['Mean'], top=IVData['STD'],
                              bottom=IVData['STD'], pen='b', beam=((spacing[1]-spacing[0]))/2)
        iv.addItem(err)
        iv.plot(IVData['Voltage'], IVData['Mean'], symbol='o', pen=None)
        iv.setLabel('left', text=x + ', Current', units='A')
        iv.setLabel('bottom', text=y + ', Voltage', units='V')
        iv.plot(x_fit, y_fit, pen='r')
        textval = pg.siFormat(1/b, precision=5, suffix='Ohm', space=True, error=None, minVal=1e-25, allowUnicode=True)
        textit = pg.TextItem(text=textval, color=(0, 0, 0))
        textit.setPos(min(IVData['Voltage']), max(IVData['Mean']))
        iv.addItem(textit)

    else:
        iv=0
    YorkFitValues={'x_fit': x_fit, 'y_fit': y_fit, 'Yintercept':a, 'Slope':b, 'Sigma_Yintercept':sigma_a, 'Sigma_Slope':sigma_b, 'Parameter':b_save}
    return (YorkFitValues, iv)



def PlotIV(output, AllData, current = 'i1', unit=1e9, axis = '', WithFit = 1):
    axis.errorbar(AllData[current]['Voltage'], AllData[current]['Mean']*unit, yerr=AllData[current]['STD']*unit, fmt='o', label=str(os.path.split(output['filename'])[1])[:-4])
    axis.set_ylabel('Current ' + current + ' [nA]')
    axis.set_xlabel('Voltage ' + AllData[current]['SweepedChannel'] +' [V]')
    if WithFit:
        axis.set_title('IV Plot with Fit: G={:.2f}nS'.format(AllData[current]['YorkFitValues']['Slope']*unit))
        axis.plot(AllData[current]['YorkFitValues']['x_fit'], AllData[current]['YorkFitValues']['y_fit']*unit, 'r--', label='Linear Fit')
    else:
        axis.set_title('IV Plot')
    return axis







def zoom_factory(ax, base_scale = 2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        plt.draw() # force re-draw

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun

def PlotData(output):
    if output['type'] == 'Axopatch':
        time=np.float32(np.arange(0, len(output['i1']))/output['samplerate'])
        #plot channel 1
        ch1_current = pg.PlotWidget(title="Current vs time Channel 1")
        ch1_current.plot(time, output['i1'])
        ch1_current.setLabel('left', text='Current', units='A')
        ch1_current.setLabel('bottom', text='Time', units='s')

        ch1_voltage = pg.PlotWidget(title="Voltage vs time Channel 1")
        ch1_voltage.plot(time, output['v1'])
        ch1_voltage.setLabel('left', text='Voltage', units='V')
        ch1_voltage.setLabel('bottom', text='Time', units='s')
        #ch1_voltage.setYLink(ch1_current)
        ch1_voltage.setXLink(ch1_current)
        if output['graphene']:
            # plot channel 1
            ch2_current = pg.PlotWidget(title="Current vs time Channel 2")
            ch2_current.plot(time, output['i2'])
            ch2_current.setLabel('left', text='Current', units='A')
            ch2_current.setLabel('bottom', text='Time', units='s')

            ch2_voltage = pg.PlotWidget(title="Voltage vs time Channel 2")
            ch2_voltage.plot(time, output['v2'])
            ch2_voltage.setLabel('left', text='Voltage', units='V')
            ch2_voltage.setLabel('bottom', text='Time', units='s')
            #ch2_voltage.setYLink(ch2_current)
            ch2_voltage.setXLink(ch2_current)

            fig_handles={'Ch1_Voltage': ch1_voltage, 'Ch2_Voltage': ch2_voltage, 'Ch2_Current': ch2_current, 'Ch1_Current': ch1_current}
            return fig_handles
        else:
            fig_handles = {'Ch1_Voltage': ch1_voltage, 'Ch1_Current': ch1_current, 'Ch2_Voltage': 0, 'Ch2_Current': 0}
            return fig_handles

    if output['type'] == 'ChimeraRaw':
        time = np.float32(np.arange(0, len(output['current']))/output['samplerate'])
        figure = plt.figure('Chimera Raw Current @ {} mV'.format(output['voltage']*1e3))
        plt.plot(time, output['current']*1e9)
        plt.ylabel('Current [nA]')
        plt.xlabel('Time [s]')
        figure.show()
        fig_handles = {'Fig1': figure, 'Fig2': 0, 'Zoom1': 0, 'Zoom2': 0}
        return fig_handles

    if output['type'] == 'ChimeraNotRaw':
        time = np.float32(np.arange(0, len(output['current']))/output['samplerate'])
        figure2 = plt.figure('Chimera Not Raw (Display Save Mode)')
        ax3 = plt.subplot(211)
        ax3.plot(time, output['current'] * 1e9)
        plt.ylabel('Current [nA]')
        ax4 = plt.subplot(212, sharex=ax3)
        ax4.plot(time, output['voltage'] * 1e3)
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [mV]')
        f2 = zoom_factory(ax3, 1.5)
        figure2.show()
        fig_handles = {'Fig1': 0, 'Fig2': figure2, 'Zoom1': 0, 'Zoom2': f2}
        return fig_handles




def PlotIV(output, AllData, current = 'i1', unit=1e9, axis = '', WithFit = 1):
    axis.errorbar(AllData[current]['Voltage'], AllData[current]['Mean']*unit, yerr=AllData[current]['STD']*unit, fmt='o', label=str(os.path.split(output['filename'])[1])[:-4])
    axis.set_ylabel('Current ' + current + ' [nA]')
    axis.set_xlabel('Voltage ' + AllData[current]['SweepedChannel'] +' [V]')
    if WithFit:
        axis.set_title('IV Plot with Fit: G={:.2f}nS'.format(AllData[current]['YorkFitValues']['Slope']*unit))
        axis.plot(AllData[current]['YorkFitValues']['x_fit'], AllData[current]['YorkFitValues']['y_fit']*unit, 'r--', label='Linear Fit')
    else:
        axis.set_title('IV Plot')
    return axis





def PlotExtractedPart(output, AllData, current = 'i1', unit=1e9, axis = '', axis2 = ''):
    time = np.arange(0, len(output[current])) / output['samplerate']
    axis.plot(time, output[current] * unit, 'b', label=str(os.path.split(output['filename'])[1])[:-4])
    axis.set_ylabel('Current ' + current + ' [nA]')
    axis.set_title('Time Trace')
    for i in range(0, len(AllData[current]['StartPoint'])):
        axis.plot(time[AllData[current]['StartPoint'][i]:AllData[current]['EndPoint'][i]],
                 output[current][AllData[current]['StartPoint'][i]:AllData[current]['EndPoint'][i]] * unit, 'r')
    axis2.plot(time, output[AllData[current]['SweepedChannel']], 'b', label=str(os.path.split(output['filename'])[1])[:-4])
    axis2.set_ylabel('Voltage ' + AllData[current]['SweepedChannel'] + ' [V]')
    axis2.set_xlabel('Time')
    return (axis, axis2)

def ExportIVData(self):
    f = h5py.File(self.matfilename + '_IVData.hdf5', "w")
    ivdata = f.create_group("IVData")
    for k, l in self.IVData.items():
        ivdata.create_dataset(k, data=l)
    fitdata = f.create_group("FitData")
    for k, l in self.FitValues.items():
        fitdata.create_dataset(k, data=l)
    Data={}
    Data['IVFit']=self.FitValues
    Data['IVData']=self.IVData
    scipy.io.savemat(self.matfilename + '_IVData.mat', Data, appendmat=True)



def SaveFigureList(folder, list):
    filename=os.path.splitext(os.path.basename(folder))[0]
    dirname=os.path.dirname(folder)
    for i in list:
        if list[i]:
            list[i].savefig(dirname+os.sep+filename+'_'+i+'.png', format='png')
    return 0



def MatplotLibIV(self):
    #IV
    if not hasattr(self, 'IVData'):
        return
    x_fit = np.linspace(min(self.IVData['Voltage']), max(self.IVData['Voltage']), 1000)
    y_fit = scipy.polyval([self.FitValues['Slope'], self.FitValues['Yintercept']], x_fit)
    x_si_params=pg.siScale(max(x_fit))
    y_si_params=pg.siScale(max(y_fit))
    plt.figure(1)
    plt.clf()
    plt.errorbar(self.IVData['Voltage']*x_si_params[0], self.IVData['Mean']*y_si_params[0], fmt='ob', yerr=self.IVData['STD']*y_si_params[0], linestyle='None')
    plt.hold(True)
    textval = pg.siFormat(1 / self.FitValues['Slope'], precision=5, suffix='Ohm', space=True, error=None, minVal=1e-25, allowUnicode=True)
    plt.annotate(textval, [min(self.IVData['Voltage']*x_si_params[0]), max(self.IVData['Mean']*y_si_params[0])])
    plt.plot(x_fit*x_si_params[0], y_fit*y_si_params[0], '-r')
    plt.xlabel('Voltage Channel {} [{}V]'.format(self.xaxisIV+1, x_si_params[1]))
    plt.ylabel('Current Channel {} [{}A]'.format(self.yaxisIV+1, y_si_params[1]))
    filename=os.path.splitext(os.path.basename(self.datafilename))[0]
    dirname=os.path.dirname(self.datafilename)
    plt.savefig(dirname+os.sep+filename+'_IV_' + str(self.xaxisIV) + str(self.yaxisIV) + '.eps')
    plt.savefig(dirname+os.sep+filename+'_IV_' + str(self.xaxisIV) + str(self.yaxisIV) + '.png')
    #plt.show()

def SaveDerivatives(self):
    PartToConsider=np.array(self.eventplot.viewRange()[0])
    partinsamples = np.int64(np.round(self.out['samplerate'] * PartToConsider))
    t = self.t[partinsamples[0]:partinsamples[1]]
    i1part = self.out['i1'][partinsamples[0]:partinsamples[1]]
    i2part = self.out['i2'][partinsamples[0]:partinsamples[1]]

    plt.figure(1, figsize=(20,7))
    plt.subplot(2, 1, 1)
    plt.plot(t, i1part, 'b')
    plt.title('i1 vs. i2')
    plt.ylabel('Ionic Current [A]')
    ax = plt.gca()
    ax.set_xticklabels([])

    plt.subplot(2, 1, 2)
    plt.plot(t, i2part, 'r')
    plt.xlabel('time (s)')
    plt.ylabel('Transverse Current [A]')
    self.pp.savefig()

    plt.figure(2, figsize=(20,7))
    plt.subplot(2, 1, 1)
    plt.plot(t, i1part, 'b')
    plt.title('i1 vs. its derivative')
    plt.ylabel('Ionic Current [A]')
    ax = plt.gca()
    ax.set_xticklabels([])

    plt.subplot(2, 1, 2)
    plt.plot(t[:-1], np.diff(i1part), 'y')
    plt.xlabel('time (s)')
    plt.ylabel('d(Ionic Current [A])/dt')
    self.pp.savefig()

    plt.figure(3, figsize=(20,7))
    plt.subplot(2, 1, 1)
    plt.plot(t, i2part, 'r')
    plt.title('i2 vs. its derivative')
    plt.ylabel('Transverse Current [A]')
    ax = plt.gca()
    ax.set_xticklabels([])

    plt.subplot(2, 1, 2)
    plt.plot(t[:-1], np.diff(i2part), 'y')
    plt.xlabel('time (s)')
    plt.ylabel('d(Transverse Current [A])/dt')
    self.pp.savefig()

def MatplotLibCurrentSignal(self):
    if not hasattr(self, 'out'):
        return
    fig, ax1 = plt.subplots(figsize=(20, 7))
    if self.ui.plotBoth.isChecked():
        ax1.plot(self.t, self.out['i1'], 'b-')
        ax1.set_xlim(self.p1.viewRange()[0])
        ax1.set_xlabel('time [s]')
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel('Ionic Current [A]', color='b')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')
        ax2 = ax1.twinx()
        ax2.plot(self.t, self.out['i2'], 'r-')
        ax1.set_xlim(self.p1.viewRange()[0])
        ax1.set_ylim(self.p1.viewRange()[1])
        ax2.set_ylabel('Transverse Current [A]', color='r')
        ax2.set_ylim(self.transverseAxis.viewRange()[1])
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
    elif self.ui.ndChannel.isChecked():
        ax1.plot(self.t, self.out['i2'], 'r-')
        ax1.set_xlim(self.p1.viewRange()[0])
        ax1.set_ylim(self.p1.viewRange()[1])
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel('Transverse Current [A]')
        for tl in ax1.get_yticklabels():
            tl.set_color('r')
    else:
        ax1.plot(self.t, self.out['i1'], 'b-')
        ax1.set_xlim(self.p1.viewRange()[0])
        ax1.set_ylim(self.p1.viewRange()[1])
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel('Ionic Current [A]')

    #self.pp.savefig()

    #SaveDerivatives(self)
    #plt.savefig(dirname+os.sep+filename+'_IV_' + str(self.xaxisIV) + str(self.yaxisIV) + '.eps')
    plt.savefig(self.matfilename + str(self.p1.viewRange()[0][0]) + '_Figure.eps')
    #plt.show()



def SaveToHDF5(self):
    f = h5py.File(self.matfilename + '_OriginalDB.hdf5', "w")
    general = f.create_group("General")
    general.create_dataset('FileName', data=self.data['filename'])
    general.create_dataset('Samplerate', data=self.outputsamplerate)
    general.create_dataset('Machine', data=self.data['type'])
    general.create_dataset('TransverseRecorded', data=self.data['graphene'])
    segmentation_LP = f.create_group("LowPassSegmentation")
    for k,l in self.AnalysisResults.items():
        set1 = segmentation_LP.create_group(k)
        lpset1 = set1.create_group('LowPassSettings')
        for o, p in self.coefficients[k].items():
             lpset1.create_dataset(o, data=p)
        for m, l in self.AnalysisResults[k].items():
             set1.create_dataset(m, data=l)
    # #
    # segmentation_LP.create_dataset('RoughEventLocations', data=self.RoughEventLocations)
    # segmentation_LP.create_dataset('StartPoints', data=self.startpoints)
    # segmentation_LP.create_dataset('EndPoints', data=endpoints)
    # segmentation_LP.create_dataset('LocalBaseline', data=self.localBaseline)
    # segmentation_LP.create_dataset('LocalVariance', data=self.localVariance)
    # segmentation_LP.create_dataset('DeltaI', data=self.deli)
    # segmentation_LP.create_dataset('DwellTime', data=self.dwell)
    # segmentation_LP.create_dataset('FractionalCurrentDrop', data=self.frac)
    # segmentation_LP.create_dataset('Frequency', data=self.dt)
    # segmentation_LP.create_dataset('NumberOfEvents', data=self.numberofevents)



def RecursiveLowPassFast(signal, coeff, self):
    """Recursive Low Pass filter applied to signal to detect abrupt changes in data.
    Parameters
    ----------
    signal : 1D array_like array,
        self.data['i1'] or self.data['i2']. 
    coeff : dictionary, {'a': np.float(a), 'E': np.float(E),
                                   'S': np.float(S),
                                   'eventlengthLimit': np.float(eventlengthLimit)
                                   *self.outputsamplerate}
    self : self.t, 1D array_like array, time (corresponding to current)
        in seconds 
    
    Returns
    -------
    output : 1D array_like [start, end, Running_mean[start], 
        Running_square_variance[start] ], 
        array of lists containing info about starting, ending 
        position of evens in points, Running mean value and 
        Running square variance at the starting point.
    
    Notes
    -----
    Fast algorithm for event detection. Filters signals and finds points which deviate from local Running mean more than coeff['S']*STD.
    References
    ----------
    .. [1] 
    Examples
    """
    # Creates running mean value of the input
    ml = scipy.signal.lfilter([1 - coeff['a'], 0], [1, -coeff['a']], signal) 
    # Plot Running threshold value at the current plot
    self.p1.plot(self.t, ml, pen=pg.mkPen(color=(246, 178, 255), width=3))

    # Creates running square deviation from the mean
    vl = scipy.signal.lfilter([1 - coeff['a'], 0], [1, -coeff['a']], np.square(signal - ml))
    # Creates "threshold line". If current value < sl[i] ->  i belongs to event. 
    sl = ml - coeff['S'] * np.sqrt(vl)
    self.p1.plot(self.t, sl, pen=pg.mkPen(color=(173, 27, 183), width=3))
    # Finds the length of the initial signal
    Ni = len(signal)
    # Finds those points where signal less than "threshold line"
    points = np.array(np.where(signal<=sl)[0])
    to_pop=np.array([]) # Empty supplementary array for finding adjacent points 
    # For loop for finding adjacent points 
    for i in range(1,len(points)):
        if points[i] - points[i - 1] == 1:
            to_pop=np.append(to_pop, i)
    # Points contain only border points of events
    points = np.delete(points, to_pop)
    # Empty list for Event location storage
    RoughEventLocations = []
    NumberOfEvents=0 #Number of events

    # For Loop for finding separating edges of different events and satisfying Event length limits
    for i in points:
        if NumberOfEvents is not 0:
            if i >= RoughEventLocations[NumberOfEvents-1][0] and i <= RoughEventLocations[NumberOfEvents-1][1]:
                continue
        NumberOfEvents += 1
        start = i
        El = ml[i] - coeff['E'] * np.sqrt(vl[i])
        Mm = ml[i]
        Vv = vl[i]
        duration = 0
        while signal[i + 1] < El and i < (Ni - 2) and duration < coeff['eventlengthLimit']:
            duration += 1
            i += 1
        if duration >= coeff['eventlengthLimit'] or i > (Ni - 10):
            NumberOfEvents -= 1
        else:
            k = start
            while signal[k] < Mm and k > 1:
                k -= 1
            start = k - 1
            k2 = i + 1
            while signal[k2] > Mm:
                k2 -= 1
            endp = k2
            if start<0:
                start=0
            RoughEventLocations.append((start, endp, ml[start], vl[start]))

    return np.array(RoughEventLocations)

def RecursiveLowPassFastUp(signal, coeff, self):
    ml = scipy.signal.lfilter([1 - coeff['a'], 0], [1, -coeff['a']], signal)
    vl = scipy.signal.lfilter([1 - coeff['a'], 0], [1, -coeff['a']], np.square(signal - ml))
    sl = ml + coeff['S'] * np.sqrt(vl) #diff
    Ni = len(signal)
    self.p1.plot(self.t, ml, pen=pg.mkPen(color=(173, 27, 183), width=3))
    points = np.array(np.where(signal>=sl)[0]) #diff
    to_pop=np.array([])
    for i in range(1,len(points)):
        if points[i] - points[i - 1] == 1:
            to_pop=np.append(to_pop, i)
    points = np.delete(points, to_pop)

    points =np.delete(points, np.array(np.where(points == 0)[0]))

    RoughEventLocations = []
    NumberOfEvents=0
    for i in points:
        if NumberOfEvents is not 0:
            if i >= RoughEventLocations[NumberOfEvents-1][0] and i <= RoughEventLocations[NumberOfEvents-1][1]:
                continue
        NumberOfEvents += 1
        start = i
        El = ml[i] + coeff['E'] * np.sqrt(vl[i]) #diff
        Mm = ml[i]
        duration = 0
        while signal[i + 1] > El and i < (Ni - 2) and duration < coeff['eventlengthLimit']: #diff
            duration += 1
            i += 1
        if duration >= coeff['eventlengthLimit'] or i > (Ni - 10):
            NumberOfEvents -= 1
        else:
            k = start
            while signal[k] > Mm and k > 2: #diff
                k -= 1
            start = k - 1
            k2 = i + 1
            while signal[k2] > Mm: #starange!!!!!
                k2 -= 1
            endp = k2
            RoughEventLocations.append((start, endp, ml[start], vl[start]))

    return np.array(RoughEventLocations)

def AddInfoAfterRecursive(self):
    """Analyse exsisting event data and returns information about each event in details.
    Parameters
    ----------
    self.sig : 'i1' or 'i2' for ionic and transverse channels respectively

    self.AnalysisResults[self.sig]['RoughEventLocations']: 2D array_like 
        (Number of events, 4) containing   
        [start, end, Running_mean[start], 
        Running_square_variance[start] ]
        for different events
    self.data: 'i1' or 'i2'
    
    Returns (creates variables)
    -------
    output : 1D array_like [start, end, Running_mean[start], 
    self.AnalysisResults[self.sig]['FractionalCurrentDrop']  current drop in series of events / current level at start 
    self.AnalysisResults[self.sig]['DeltaI']  #current drop nA in series of events
    self.AnalysisResults[self.sig]['DwellTime']  #end[i] - start[i] in sec. in series of 
        events
    self.AnalysisResults[self.sig]['Frequency'] # start[i+1] - start[i] in sec. in series of 
        events
    
    Notes
    -----
    Returns information about relative current drop, absolute current drop, length of event and time differences between different events.
    .
    References
    ----------
    .. [1] 
    Examples
    """
    
    print('Info about channel:'  + str(self.sig))
    startpoints = np.uint64(self.AnalysisResults[self.sig]['RoughEventLocations'][:, 0])
    endpoints = np.uint64(self.AnalysisResults[self.sig]['RoughEventLocations'][:, 1])
    localBaseline = self.AnalysisResults[self.sig]['RoughEventLocations'][:, 2]
    localVariance = self.AnalysisResults[self.sig]['RoughEventLocations'][:, 3]
    for (j,k) in enumerate(startpoints): print("%10.7f"% float(startpoints[j]/self.outputsamplerate))
    CusumBaseline=500
    numberofevents = len(startpoints)
    self.AnalysisResults[self.sig]['StartPoints'] = startpoints
    self.AnalysisResults[self.sig]['EndPoints'] = endpoints
    self.AnalysisResults[self.sig]['LocalBaseline'] = localBaseline
    self.AnalysisResults[self.sig]['LocalVariance'] = localVariance
    self.AnalysisResults[self.sig]['NumberOfEvents'] = len(startpoints)

    #### Now we want to move the endpoints to be the last minimum for each ####
    #### event so we find all minimas for each event, and set endpoint to last ####

    deli = np.zeros(numberofevents)
    dwell = np.zeros(numberofevents)
    limit=500e-6*self.outputsamplerate  #0.5 ms
    AllFits={}

    for i in range(numberofevents):
        length = endpoints[i] - startpoints[i]
        if length <= limit and length>3:
            # Impulsion Fit to minimal value
            deli[i] = localBaseline[i] - np.min(self.data[self.sig][int(startpoints[i]+1):int(endpoints[i]-1)]) #current drop cuurrent at starting point - current minimal velue
            dwell[i] = (endpoints[i] - startpoints[i]) / self.outputsamplerate #length of event in seconds
        elif length > limit:
            deli[i] = localBaseline[i] - np.mean(self.data[self.sig][int(startpoints[i]+5):int(endpoints[i]-5)])
            dwell[i] = (endpoints[i] - startpoints[i]) / self.outputsamplerate
            # # Cusum Fit
            # sigma = np.sqrt(localVariance[i])
            # delta = 2e-9
            # h = 1 * delta / sigma
            # (mc, kd, krmv) = CUSUM(self.out[self.sig][startpoints[i]-CusumBaseline:endpoints[i]+CusumBaseline], delta, h)
            # zeroPoint = startpoints[i]-CusumBaseline
            # krmv = krmv+zeroPoint+1
            # AllFits['Event' + str(i)] = {}
            # AllFits['Event' + str(i)]['mc'] = mc
            # AllFits['Event' + str(i)]['krmv'] = krmv
        else:
            deli[i] = localBaseline[i] - np.min(self.data[self.sig][startpoints[i]:endpoints[i]])
            dwell[i] = (endpoints[i] - startpoints[i]) / self.outputsamplerate

    frac = deli / localBaseline #fraction: current drop / current at start
    dt = np.array(0)
    dt = np.append(dt, np.diff(startpoints) / self.outputsamplerate) # differences between starts of different events (Frequency of events)
    numberofevents = len(dt)

    #self.AnalysisResults[self.sig]['CusumFits'] = AllFits
    self.AnalysisResults[self.sig]['FractionalCurrentDrop'] = frac # current drop / current at start 
    self.AnalysisResults[self.sig]['DeltaI'] = deli #current drop in nA
    self.AnalysisResults[self.sig]['DwellTime'] = dwell #end[i] - start[i] in sec.
    self.AnalysisResults[self.sig]['Frequency'] = dt # start[i+1] - start[i] in sec.

def SavingAndPlottingAfterRecursive(self):
    """Plotting the results of analysis done by Low Pass.
    Parameters
    ----------
    For series of events:
    self.AnalysisResults[self.sig]['StartPoints'] : Starting points of events in sec.
    self.AnalysisResults[self.sig]['EndPoints'] : Ending points of events in sec.
    self.AnalysisResults[self.sig]['NumberOfEvents'] : Total number of events
    self.AnalysisResults[self.sig]['DeltaI'] : Current drop in nA
    self.AnalysisResults[self.sig]['DwellTime'] : end[i] - start[i] in sec.
    self.AnalysisResults[self.sig]['FractionalCurrentDrop'] : current drop / current at 
        start 
    self.AnalysisResults[self.sig]['Frequency'] : end[i] - start[i] in sec.
    self.AnalysisResults[self.sig]['LocalBaseline'] : start[i+1] - start[i] in sec.
    
    Returns (creates figures)
    -------
    
    
    Notes
    -----
    .
    .
    References
    ----------
    .. [1] 
    Examples
    """


    #Starting points of events in sec.
    startpoints=self.AnalysisResults[self.sig]['StartPoints']

    #Ending points of events in sec.
    endpoints=self.AnalysisResults[self.sig]['EndPoints']

    #Total number of events
    numberofevents=self.AnalysisResults[self.sig]['NumberOfEvents']

    #Current drop in nA
    deli=self.AnalysisResults[self.sig]['DeltaI']

    #end[i] - start[i] in sec.
    dwell=self.AnalysisResults[self.sig]['DwellTime']

    #current drop / current at start 
    frac=self.AnalysisResults[self.sig]['FractionalCurrentDrop']

    #start[i+1] - start[i] in sec.
    dt=self.AnalysisResults[self.sig]['Frequency']

    #start[i+1] - start[i] in sec.
    localBaseline=self.AnalysisResults[self.sig]['LocalBaseline']

    # If plotting takes too much time: "Don_t_plot..." the graph will be deleted
    if not self.ui.actionDon_t_Plot_if_slow.isChecked():
        #clear signal plot
        self.p1.clear()
        # Event detection plot, Signal Plot
        self.p1.plot(self.t, self.data[self.sig], pen='b')
        # Draw green circles indicating start of event
        self.p1.plot(self.t[startpoints],  self.data[self.sig][startpoints], pen=None, symbol='o', symbolBrush='g', symbolSize=10)
        # Draw red circles indicating end of event
        self.p1.plot(self.t[endpoints],  self.data[self.sig][endpoints], pen=None, symbol='o', symbolBrush='r', symbolSize=10)
        #self.p1.plot(self.t[startpoints-10], localBaseline, pen=None, symbol='x', symbolBrush='y', symbolSize=10)
    
    # choice only data for current file name
    try:
        self.p2.data = self.p2.data[np.where(np.array(self.sdf.fn) != self.matfilename)]
    except:
        IndexError
    #Data frame for event info storage
    self.sdf = self.sdf[self.sdf.fn != self.matfilename]
    
    #Panda series with file name of data loaded (without ending .dat or some other) 
    #repeated numberofevents times
    fn = pd.Series([self.matfilename, ] * numberofevents)
    
    #Same color identification repeated numberofevents
    color = pd.Series([self.cb.color(), ] * numberofevents)
    
    self.sdf = self.sdf.append(pd.DataFrame({'fn': fn, 'color': color, 'deli': deli,
                                             'frac': frac, 'dwell': dwell,
                                             'dt': dt, 'startpoints': startpoints,
                                             'endpoints': endpoints, 'baseline': localBaseline}), ignore_index=True)
    
    # Create Scatter plot with
    # x = log10(dwell)
    # y = current drop / current at start
    #self.p2.addPoints(x=np.log10(dwell), y=frac, symbol='o', brush=(self.cb.color()), pen=None, size=10)
    self.p2.addPoints(x=dwell, y=frac,
                      symbol='o', brush=(self.cb.color()), pen=None, size=10)
    self.w1.addItem(self.p2)
    self.w1.setLogMode(x=False, y=False)
    self.p1.autoRange()
    self.w1.autoRange()
    self.ui.scatterplot.update() # Replot Scatter plot
    # Set y - axis range
    self.w1.setRange(yRange=[0, 1])
    
    # Pandas series of colors. 
    colors = self.sdf.color
    # If we have data from different experiments and different analyte
    # we can change the color for them 
    for i, x in enumerate(colors):

        # Preparation for distribution histogram of fraction. Different color
        # Corresponds to different experiments
        # For better undarstanding see Feng et. al Indentification of single nucliotides
        # in MoS2 nanopores - different nucliotides for different colors 
        # frac = current drop / current at start 
        fracy, fracx = np.histogram(self.sdf.frac[self.sdf.color == x],
                                    bins=np.linspace(0, 1, int(self.ui.fracbins.text())))
        # Create pyqtgraph hist of Fraction data
        hist = pg.PlotCurveItem(fracx, fracy , stepMode = True, 
                                fillLevel=0, brush = x, pen = 'k')       
        #Plot Frac histogram 
        self.w2.addItem(hist) 
        

        # Preparation for distribution histogram of Current drop in nA.
        # Idea of color choice is the same as in histogram above.
        deliy, delix = np.histogram(self.sdf.deli[self.sdf.color == x], 
                                    bins=np.linspace(float(self.ui.delirange0.text()) *1e-9, 
                                    float(self.ui.delirange1.text()) * 1e-9, 
                                    int(self.ui.delibins.text())))
        # Create pyqtgraph hist of Deli data
        #hist = pg.BarGraphItem(height=deliy, x0=delix[:-1], x1=delix[1:], brush=x)
        hist = pg.PlotCurveItem(delix, deliy , stepMode = True, 
                                fillLevel=0, brush = x, pen = 'k')
        self.w3.addItem(hist) #Deli histogram plot
        self.w3.setRange(xRange=[float(self.ui.delirange0.text()) * 10 ** -9,
                         float(self.ui.delirange1.text()) * 10 ** -9])
        
        # Preparation for distribution histogram of length of events expressed as
        # end[i] - start[i] in sec..
        # Idea of color choice is the same as in histogram above.
        #linspace for bins
        print('dwell = ' + str(self.sdf.dwell))
        bins_dwell = np.linspace(float(self.ui.dwellrange0.text()) * 1e-6, 
                                 float(self.ui.dwellrange1.text()) * 1e-6, 
                                 int(self.ui.dwellbins.text()))

        dwelly, dwellx = np.histogram((self.sdf.dwell[self.sdf.color == x]),
                                      bins=bins_dwell,range=(bins_dwell.min(),
                                      bins_dwell.max()))
        hist = pg.PlotCurveItem(dwellx, dwelly , stepMode = True, 
                                fillLevel=0, brush = x, pen = 'k')
        self.w4.addItem(hist)

       
        # Preparation for distribution histogram of start[i+1] - start[i] in sec. 
        # "Frequency" of events expressed as
        # Idea of color choice is the same as in histogram above.

        dty, dtx = np.histogram(self.sdf.dt[self.sdf.color == x],
                                bins=np.linspace(float(self.ui.dtrange0.text()), 
                                float(self.ui.dtrange1.text()),
                                int(self.ui.dtbins.text())))
        hist = pg.PlotCurveItem(dtx, dty , stepMode = True, 
                                fillLevel=0, brush = x, pen = 'k')
        self.w5.addItem(hist) #Dt histogram plot


def save(self):
    np.savetxt(self.matfilename + 'DB.txt', np.column_stack((self.deli, self.frac, self.dwell, self.dt)),
               delimiter='\t')

def PlotEventSingle(self, clicked=[]):
    f = h5py.File(self.matfilename + '_OriginalDB.hdf5', "r")
    sig='i1'

    startpoints=self.AnalysisResults[sig]['StartPoints']
    endpoints=self.AnalysisResults[sig]['EndPoints']
    localBaseline=self.AnalysisResults[sig]['LocalBaseline']

    # Reset plot
    self.p3.setLabel('bottom', text='Time', units='s')
    self.p3.setLabel('left', text='Current', units='A')
    self.p3.clear()
    eventnumber = np.int(self.ui.eventnumberentry.text())
    eventbuffer = np.int(self.ui.eventbufferentry.value())
    
    # plot event trace
    self.p3.plot(self.t[int(startpoints[eventnumber] - eventbuffer):int(endpoints[eventnumber] + eventbuffer)],
                 self.data[sig][int(startpoints[eventnumber] - eventbuffer):int(endpoints[eventnumber] + eventbuffer)],
                 pen='b')

    # plot event fit
    self.p3.plot(self.t[int(startpoints[eventnumber] - eventbuffer):int(endpoints[eventnumber] + eventbuffer)], np.concatenate((
        np.repeat(np.array([localBaseline[eventnumber]]), eventbuffer),
        np.repeat(np.array([localBaseline[eventnumber] - self.AnalysisResults['i1']['DeltaI'][eventnumber
        ]]), endpoints[eventnumber] - startpoints[eventnumber]),
        np.repeat(np.array([localBaseline[eventnumber]]), eventbuffer)), 0),
                 pen=pg.mkPen(color=(173, 27, 183), width=3))

    self.p3.autoRange()

def PlotEventSingle_CUSUM(self, clicked=[]):
    #f = h5py.File(self.matfilename + '_OriginalDB.hdf5', "r")
    sig='i1'

    startpoints=self.cusum['i1']['Real_Start']
    endpoints=self.cusum['i1']['Real_End']
    localBaseline=self.cusum['i1']['Real_Depth']
    self.NumberOfEvents = len(startpoints)
    # Reset plot
    self.p3.setLabel('bottom', text='Time', units='s')
    self.p3.setLabel('left', text='Current', units='A')
    self.p3.clear()
    eventnumber = np.int(self.ui.eventnumberentry.text())
    eventbuffer = np.int(self.ui.eventbufferentry.value())
    print('Start Points = ' + str(startpoints[eventnumber]))
    print('Start Points = ' + str(endpoints[eventnumber]))
    # plot event trace
    self.p3.plot(self.t[int(startpoints[eventnumber] - eventbuffer):int(endpoints[eventnumber] + eventbuffer)],
                 self.data['i1'][int(startpoints[eventnumber] - eventbuffer):int(endpoints[eventnumber] + eventbuffer)],
                 pen='b')
    
    # plot event fit
    self.p3.plot(self.t[int(startpoints[eventnumber] - eventbuffer):int(endpoints[eventnumber] + eventbuffer)], np.concatenate((
        np.repeat(np.mean(self.data['i1'][int(startpoints[eventnumber] - eventbuffer):int(startpoints[eventnumber])]), eventbuffer),
        np.repeat(localBaseline[eventnumber], endpoints[eventnumber] - startpoints[eventnumber]),
        np.repeat(np.mean(self.data['i1'][int(endpoints[eventnumber]):int(endpoints[eventnumber] + eventbuffer)]), eventbuffer)), 0),
                 pen=pg.mkPen(color=(173, 27, 183), width=3))
    
    self.p3.autoRange()

def PlotEventDouble(self, clicked=[]):
    f = h5py.File(self.matfilename + '_OriginalDB.hdf5', "r")

    if self.ui.actionPlot_i1_detected_only.isChecked():
        indexes = f['LowPassSegmentation/i1/OnlyIndex']
        i = f['LowPassSegmentation/i1/']
        sig = 'i1'
        sig2 = 'i2'
        leftlabel = "Ionic Current"
        rightlabel = "Transverse Current"
    if self.ui.actionPlot_i2_detected_only.isChecked():
        indexes = f['LowPassSegmentation/i2/OnlyIndex']
        i = f['LowPassSegmentation/i2/']
        sig = 'i2'
        sig2 = 'i1'
        rightlabel = "Ionic Current"
        leftlabel = "Transverse Current"
    self.p3.clear()
    self.transverseAxisEvent.clear()

    p1 = self.p3.plotItem
    p1.getAxis('left').setLabel(text=leftlabel, color='#0000FF', units='A')
    ## create a new ViewBox, link the right axis to its coordinate system
    p1.showAxis('right')
    p1.scene().addItem(self.transverseAxisEvent)
    p1.getAxis('right').linkToView(self.transverseAxisEvent)
    self.transverseAxisEvent.setXLink(p1)
    self.transverseAxisEvent.show()
    p1.getAxis('right').setLabel(text=rightlabel, color='#FF0000', units='A')

    def updateViews():
        ## view has resized; update auxiliary views to match
        self.transverseAxisEvent.setGeometry(p1.vb.sceneBoundingRect())
        self.transverseAxisEvent.linkedViewChanged(p1.vb, self.transverseAxisEvent.XAxis)

    updateViews()
    p1.vb.sigResized.connect(updateViews)

    # Correct for user error if non-extistent number is entered
    eventbuffer = np.int(self.ui.eventbufferentry.value())
    maxEvents=self.NumberOfEvents

    eventnumber = np.int(self.ui.eventnumberentry.text())
    if eventnumber >= maxEvents:
        eventnumber=0
        self.ui.eventnumberentry.setText(str(eventnumber))
    elif eventnumber < 0:
        eventnumber=maxEvents
        self.ui.eventnumberentry.setText(str(eventnumber))

    # plot event trace
    parttoplot=np.arange(i['StartPoints'][indexes[eventnumber]] - eventbuffer, i['EndPoints'][indexes[eventnumber]] + eventbuffer,1, dtype=np.uint64)

    p1.plot(self.t[parttoplot], self.out[sig][parttoplot], pen='b')

    # plot event fit
    p1.plot(self.t[parttoplot],
                 np.concatenate((
                     np.repeat(np.array([i['LocalBaseline'][indexes[eventnumber]]]), eventbuffer),
                     np.repeat(np.array([i['LocalBaseline'][indexes[eventnumber]] - i['DeltaI'][indexes[eventnumber]
                     ]]), i['EndPoints'][indexes[eventnumber]] - i['StartPoints'][indexes[eventnumber]]),
                     np.repeat(np.array([i['LocalBaseline'][indexes[eventnumber]]]), eventbuffer)), 0),
                 pen=pg.mkPen(color=(173, 27, 183), width=3))

    # plot 2nd Channel
    if self.Derivative == 'i2':
        self.transverseAxisEvent.addItem(pg.PlotCurveItem(self.t[parttoplot][:-1], np.diff(self.out[sig][parttoplot]), pen='r'))
        p1.getAxis('right').setLabel(text='Derivative of i2', color='#FF0000', units='A')
        print('In if...')
        #plt.plot(t[:-1], np.diff(i1part), 'y')
    else:
        self.transverseAxisEvent.addItem(pg.PlotCurveItem(self.t[parttoplot], self.out[sig2][parttoplot], pen='r'))

    min1 = np.min(self.out[sig][parttoplot])
    max1 = np.max(self.out[sig][parttoplot])
    self.p3.setYRange(min1-(max1-min1), max1)
    self.p3.enableAutoRange(axis='x')
    min2 = np.min(self.out[sig2][parttoplot])
    max2 = np.max(self.out[sig2][parttoplot])
    self.transverseAxisEvent.setYRange(min2, max2+(max2-min2))

    # Mark event start and end points
    p1.plot([self.t[i['StartPoints'][indexes[eventnumber]]], self.t[i['StartPoints'][indexes[eventnumber]]]],
                 [self.out[sig][i['StartPoints'][indexes[eventnumber]]], self.out[sig][i['StartPoints'][indexes[eventnumber]]]],
                 pen=None,
                 symbol='o', symbolBrush='g', symbolSize=12)
    p1.plot([self.t[i['EndPoints'][indexes[eventnumber]]], self.t[i['EndPoints'][indexes[eventnumber]]]],
                 [self.out[sig][i['EndPoints'][indexes[eventnumber]]], self.out[sig][i['EndPoints'][indexes[eventnumber]]]],
                 pen=None,
                 symbol='o', symbolBrush='r', symbolSize=12)

    dtime=pg.siFormat(i['DwellTime'][indexes[eventnumber]], precision=5, suffix='s', space=True, error=None, minVal=1e-25, allowUnicode=True)
    dI=pg.siFormat(i['DwellTime'][indexes[eventnumber]], precision=5, suffix='A', space=True, error=None, minVal=1e-25, allowUnicode=True)
    self.ui.eventinfolabel.setText(leftlabel + ': Dwell Time=' + dtime + ', Deli=' + dI)

def PlotEventDoubleFit(self, clicked=[]):
    f = h5py.File(self.matfilename + '_OriginalDB.hdf5', "r")
    i1_indexes=f['LowPassSegmentation/i1/CommonIndex']
    i2_indexes=f['LowPassSegmentation/i2/CommonIndex']
    i1=f['LowPassSegmentation/i1/']
    i2=f['LowPassSegmentation/i2/']

    self.p3.clear()
    self.transverseAxisEvent.clear()

    leftlabel="Ionic Current"
    rightlabel="Transverse Current"

    p1 = self.p3.plotItem
    p1.getAxis('left').setLabel(text=leftlabel, color='#0000FF', units='A')
    ## create a new ViewBox, link the right axis to its coordinate system
    p1.showAxis('right')
    p1.scene().addItem(self.transverseAxisEvent)
    p1.getAxis('right').linkToView(self.transverseAxisEvent)
    self.transverseAxisEvent.setXLink(p1)
    self.transverseAxisEvent.show()
    p1.getAxis('right').setLabel(text=rightlabel, color='#FF0000', units='A')

    def updateViews():
        ## view has resized; update auxiliary views to match
        self.transverseAxisEvent.setGeometry(p1.vb.sceneBoundingRect())
        self.transverseAxisEvent.linkedViewChanged(p1.vb, self.transverseAxisEvent.XAxis)

    updateViews()
    p1.vb.sigResized.connect(updateViews)

    # Correct for user error if non-extistent number is entered
    eventbuffer = np.int(self.ui.eventbufferentry.value())
    maxEvents=len(i1_indexes)

    eventnumber = np.int(self.ui.eventnumberentry.text())
    if eventnumber >= maxEvents:
        eventnumber=0
        self.ui.eventnumberentry.setText(str(eventnumber))
    elif eventnumber < 0:
        eventnumber=maxEvents
        self.ui.eventnumberentry.setText(str(eventnumber))

    # plot event trace
    parttoplot=np.arange(i1['StartPoints'][i1_indexes[eventnumber]] - eventbuffer, i1['EndPoints'][i1_indexes[eventnumber]] + eventbuffer,1, dtype=np.uint64)
    parttoplot2=np.arange(i2['StartPoints'][i2_indexes[eventnumber]] - eventbuffer, i2['EndPoints'][i2_indexes[eventnumber]] + eventbuffer,1, dtype=np.uint64)

    p1.plot(self.t[parttoplot], self.out['i1'][parttoplot], pen='b')

    # plot event fit
    p1.plot(self.t[parttoplot],
                 np.concatenate((
                     np.repeat(np.array([i1['LocalBaseline'][i1_indexes[eventnumber]]]), eventbuffer),
                     np.repeat(np.array([i1['LocalBaseline'][i1_indexes[eventnumber]] - i1['DeltaI'][i1_indexes[eventnumber]
                     ]]), i1['EndPoints'][i1_indexes[eventnumber]] - i1['StartPoints'][i1_indexes[eventnumber]]),
                     np.repeat(np.array([i1['LocalBaseline'][i1_indexes[eventnumber]]]), eventbuffer)), 0),
                 pen=pg.mkPen(color=(173, 27, 183), width=3))

    # plot 2nd Channel
    self.transverseAxisEvent.addItem(pg.PlotCurveItem(self.t[parttoplot2], self.out['i2'][parttoplot2], pen='r'))
    self.transverseAxisEvent.addItem(pg.PlotCurveItem(self.t[parttoplot2], np.concatenate((np.repeat(
        np.array([i2['LocalBaseline'][i2_indexes[eventnumber]]]), eventbuffer), np.repeat(
        np.array([i2['LocalBaseline'][i2_indexes[eventnumber]] - i2['DeltaI'][i2_indexes[eventnumber]]]),
        i2['EndPoints'][i2_indexes[eventnumber]] - i2['StartPoints'][i2_indexes[eventnumber]]), np.repeat(
        np.array([i2['LocalBaseline'][i2_indexes[eventnumber]]]), eventbuffer)), 0), pen=pg.mkPen(color=(173, 27, 183), width=3)))

    min1 = np.min(self.out['i1'][parttoplot])
    max1 = np.max(self.out['i1'][parttoplot])
    self.p3.setYRange(min1-(max1-min1), max1)
    self.p3.enableAutoRange(axis='x')
    min2 = np.min(self.out['i2'][parttoplot2])
    max2 = np.max(self.out['i2'][parttoplot2])
    self.transverseAxisEvent.setYRange(min2, max2+(max2-min2))

    # Mark event start and end points
    p1.plot([self.t[i1['StartPoints'][i1_indexes[eventnumber]]], self.t[i1['StartPoints'][i1_indexes[eventnumber]]]],
                 [self.out['i1'][i1['StartPoints'][i1_indexes[eventnumber]]], self.out['i1'][i1['StartPoints'][i1_indexes[eventnumber]]]],
                 pen=None,
                 symbol='o', symbolBrush='g', symbolSize=12)
    p1.plot([self.t[i1['EndPoints'][i1_indexes[eventnumber]]], self.t[i1['EndPoints'][i1_indexes[eventnumber]]]],
                 [self.out['i1'][i1['EndPoints'][i1_indexes[eventnumber]]], self.out['i1'][i1['EndPoints'][i1_indexes[eventnumber]]]],
                 pen=None,
                 symbol='o', symbolBrush='r', symbolSize=12)

    self.transverseAxisEvent.addItem(pg.PlotCurveItem([self.t[i2['StartPoints'][i2_indexes[eventnumber]]], self.t[i2['StartPoints'][i2_indexes[eventnumber]]]],
                 [self.out['i2'][i2['StartPoints'][i2_indexes[eventnumber]]], self.out['i2'][i1['StartPoints'][i2_indexes[eventnumber]]]],
                 pen=None,
                 symbol='o', symbolBrush='g', symbolSize=12))
    self.transverseAxisEvent.addItem(pg.PlotCurveItem([self.t[i2['EndPoints'][i2_indexes[eventnumber]]], self.t[i2['EndPoints'][i2_indexes[eventnumber]]]],
                 [self.out['i2'][i1['EndPoints'][i2_indexes[eventnumber]]], self.out['i2'][i1['EndPoints'][i2_indexes[eventnumber]]]],
                 pen=None,
                 symbol='o', symbolBrush='r', symbolSize=12))

    dtime=pg.siFormat(i1['DwellTime'][i1_indexes[eventnumber]], precision=5, suffix='s', space=True, error=None, minVal=1e-25, allowUnicode=True)
    dI=pg.siFormat(i1['DwellTime'][i1_indexes[eventnumber]], precision=5, suffix='A', space=True, error=None, minVal=1e-25, allowUnicode=True)
    dtime2=pg.siFormat(i2['DwellTime'][i2_indexes[eventnumber]], precision=5, suffix='s', space=True, error=None, minVal=1e-25, allowUnicode=True)
    dI2=pg.siFormat(i2['DwellTime'][i2_indexes[eventnumber]], precision=5, suffix='A', space=True, error=None, minVal=1e-25, allowUnicode=True)

    self.ui.eventinfolabel.setText('Ionic Dwell Time=' + dtime + ',   Ionic Deli=' + dI + ', ' 'Trans Dwell Time=' + dtime2 + ',   Trans Deli=' + dI2)

def SaveEventPlotMatplot(self):
    eventbuffer = np.int(self.ui.eventbufferentry.value())
    eventnumber = np.int(self.ui.eventnumberentry.text())

    parttoplot = np.arange(i['StartPoints'][indexes[eventnumber]][eventnumber] - eventbuffer, i['EndPoints'][indexes[eventnumber]][eventnumber] + eventbuffer, 1,
                           dtype=np.uint64)

    t=np.arange(0,len(parttoplot))
    t=t/self.out['samplerate']*1e3

    fig1=plt.figure(1, figsize=(20,7))
    plt.subplot(2, 1, 1)
    plt.cla
    plt.plot(t, self.out['i1'][parttoplot]*1e9, 'b')
    plt.ylabel('Ionic Current [nA]')
    ax = plt.gca()
    ax.set_xticklabels([])

    plt.subplot(2, 1, 2)
    plt.cla
    plt.plot(t, self.out['i2'][parttoplot]*1e9, 'r')
    plt.ylabel('Transverse Current [nA]')
    plt.xlabel('time (ms)')
    self.pp.savefig()
    fig1.clear()

def CombineTheTwoChannels(file):
    f = h5py.File(file, 'a')
    i1 = f['LowPassSegmentation/i1/']
    i2 = f['LowPassSegmentation/i2/']
    i1StartP = i1['StartPoints'][:]
    i2StartP = i2['StartPoints'][:]

    # Common Events
    # Take Longer
    CommonEventsi1Index = np.array([], dtype=np.uint64)
    CommonEventsi2Index = np.array([], dtype=np.uint64)
    DelayLimit = 10e-3*f['General/Samplerate/'].value

    for k in i1StartP:
        val = i2StartP[(i2StartP > k - DelayLimit) & (i2StartP < k + DelayLimit)]
        if len(val)==1:
            CommonEventsi2Index = np.append(CommonEventsi2Index, np.where(i2StartP == val)[0])
            CommonEventsi1Index = np.append(CommonEventsi1Index, np.where(i1StartP == k)[0])
        if len(val) > 1:
            diff=np.absolute(val-k)
            minIndex=np.where(diff == np.min(diff))
            CommonEventsi2Index = np.append(CommonEventsi2Index, np.where(i2StartP == val[minIndex])[0])
            CommonEventsi1Index = np.append(CommonEventsi1Index, np.where(i1StartP == k)[0])


    # for i in range(len(i1StartP)):
    #     for j in range(len(i2StartP)):
    #         if np.absolute(i1StartP[i] - i2StartP[j]) < DelayLimit:
    #             CommonEventsi1Index = np.append(CommonEventsi1Index, i)
    #             CommonEventsi2Index = np.append(CommonEventsi2Index, j)

    # Only i1
    Onlyi1Indexes = np.delete(range(len(i1StartP)), CommonEventsi1Index)
    # Only i2
    Onlyi2Indexes = np.delete(range(len(i2StartP)), CommonEventsi2Index)

    e = "CommonIndex" in i1
    if e:
        del i1['CommonIndex']
        i1.create_dataset('CommonIndex', data=CommonEventsi1Index)
        del i2['CommonIndex']
        i2.create_dataset('CommonIndex', data=CommonEventsi2Index)
        del i1['OnlyIndex']
        i1.create_dataset('OnlyIndex', data=Onlyi1Indexes)
        del i2['OnlyIndex']
        i2.create_dataset('OnlyIndex', data=Onlyi2Indexes)
    else:
        i1.create_dataset('CommonIndex', data=CommonEventsi1Index)
        i2.create_dataset('CommonIndex', data=CommonEventsi2Index)
        i1.create_dataset('OnlyIndex', data=Onlyi1Indexes)
        i2.create_dataset('OnlyIndex', data=Onlyi2Indexes)

    CommonIndexes={}
    CommonIndexes['i1']=CommonEventsi1Index
    CommonIndexes['i2']=CommonEventsi2Index
    OnlyIndexes={}
    OnlyIndexes['i1'] = Onlyi1Indexes
    OnlyIndexes['i2'] = Onlyi2Indexes
    return (CommonIndexes, OnlyIndexes)

def PlotEvent(t1, t2, i1, i2, fit1 = np.array([]), fit2 = np.array([])):
    fig1 = plt.figure(1, figsize=(20, 7))
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212, sharex=ax1)
    ax1.plot(t1, i1*1e9, 'b')
    if len(fit1) is not 0:
        ax1.plot(t1, fit1*1e9, 'y')
    ax2.plot(t2, i2*1e9, 'r')
    if len(fit2) is not 0:
        ax2.plot(t2, fit2*1e9, 'y')
    ax1.set_ylabel('Ionic Current [nA]')
    #ax1.set_xticklabels([])
    ax2.set_ylabel('Transverse Current [nA]')
    ax2.set_xlabel('Time [ms]')
    ax2.ticklabel_format(useOffset=False)
    ax2.ticklabel_format(useOffset=False)
    ax1.ticklabel_format(useOffset=False)
    ax1.ticklabel_format(useOffset=False)

    return fig1

def EditInfoText(self):
    text2='ionic: {} events, trans: {} events\n'.format(str(self.AnalysisResults['i1']['RoughEventLocations'].shape[0]), str(self.AnalysisResults['i2']['RoughEventLocations'].shape[0]))
    text1='The file contains:\n{} Common Events\n{} Ionic Only Events\n{} Transverse Only Events'.format(len(self.CommonIndexes['i1']), len(self.OnlyIndexes['i1']), len(self.OnlyIndexes['i2']))
    print(text2)
    print(text1)
    self.ui.InfoTexts.setText(text2+text1)

def creation_date(path_to_file):
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_mtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime



def CombineEventDatabases(filename, DBfiles):
    f = h5py.File(filename, "w")

    general = f.create_group("RawDB")
    general.create_dataset('FileName', data=self.out['filename'])
    general.create_dataset('Samplerate', data=self.out['samplerate'])
    general.create_dataset('Machine', data=self.out['type'])
    general.create_dataset('TransverseRecorded', data=self.out['graphene'])
    segmentation_LP = f.create_group("LowPassSegmentation")
    for k, l in self.AnalysisResults.items():
        set1 = segmentation_LP.create_group(k)
        lpset1 = set1.create_group('LowPassSettings')
        for o, p in self.coefficients[k].items():
            lpset1.create_dataset(o, data=p)
        for m, l in self.AnalysisResults[k].items():
            set1.create_dataset(m, data=l)
