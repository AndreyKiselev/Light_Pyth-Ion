import numpy as np
from scipy import io #scipy input and output function
import os # operation system
from PyQt5 import QtGui
import scipy
from scipy import signal
from timeit import default_timer as timer #for time measurements
def getfile(self): #load button pressed
    self.CuttingApplied = False #when we cut some data, PSD and histograms are too complicated and not plotted
    self.direc = os.getcwd()
    try:
        ######## attempt to open dialog from most recent directory########
        datafilenametemp = QtGui.QFileDialog.getOpenFileName(parent=self, caption='Open file', directory=str(self.direc), filter="Amplifier Files(*.dat *.log  *.mat)") #creates turple with two values: the adress of the selected file and the line of Amplifier Filters
        if datafilenametemp[0] != '': #if you select some file the param will be not 0
            self.datafilename=datafilenametemp[0] #full path to the loaded file
            self.direc=os.path.dirname(self.datafilename) #Return the directory name
            Load(self, loadandplot = True) #Load module
    except IOError:
        #### if user cancels during file selection, exit loop#############
        pass


def Load(self, loadandplot = True): #file loader works with getfile()

    print('File Adress: {}'.format(self.datafilename)) #print chosen file adress on command line
    self.data = {} #initial dictionary for current and voltage
    self.var = {} #initial dictionary for variation (different channels)

    #####Event plot cutting off Pyth-ion logo, inserting Event plot and its parameters
    self.p3.clear()
    self.p3.setLabel('top', text='Event plot')
    self.p3.setLabel('bottom', text='Current', units='A', unitprefix = 'n')
    self.p3.setLabel('left', text='', units = 'Counts')
    self.p3.setAspectLocked(False)

    #####establish initial parameters for new file  
    self.ui.eventinfolabel.clear() # clearing Dwell time from previous
    self.ui.eventnumberentry.setText(str(0)) #nulify Event number     
    self.hasbaselinebeenset=0 #no baseline calc done for new file
    self.ui.AxopatchGroup.setVisible(0) #two channels not available at start


    self.ui.filelabel.setText('...' + str(self.datafilename[-45:])) #print path to downloaded file
    self.LPfiltercutoff = np.float64(self.ui.LPentry.text())*1000 #filter cutoff from input (Hz)
    self.Samplerate_aftr_LPF = np.float64(self.ui.Samplerate_aftr_LPF.text())*1000 #filter cutoff from input (Hz)
        
        

    ########Importing Axopatch Data######################
    if str(os.path.splitext(self.datafilename)[1])=='.dat':
        print('Loading Axopatch Data')
        ds_factor = 1 #Scaling factor for time
        self.out=ImportAxopatchData(self.datafilename)
        self.matfilename = str(os.path.splitext(self.datafilename)[0])
        self.outputsamplerate=self.out['samplerate']
        print('samplrt='+ str(self.outputsamplerate))
        self.ui.outputsamplerateentry.setText(str(self.out['samplerate']/1000))
        if self.out['graphene']:
            self.ui.AxopatchGroup.setVisible(1)
        else:
            self.ui.AxopatchGroup.setVisible(0)


    if str(os.path.splitext(self.datafilename)[1]) == '.log':
        print('Loading Chimera File')
        self.out = ImportChimeraData(self.datafilename)
        self.ui.outputsamplerateentry.setText(str(self.out['samplerate'] / 1000))
        self.matfilename = str(os.path.splitext(self.datafilename)[0])
        if self.out['type']  == 'ChimeraNotRaw':
            ds_factor = 1 #Scaling factor for time
            self.data['i1'] = self.out['i1']
            print('chimera not raw')
            print('Amount of points before filtering:' + str(self.data['i1'].shape)+ ' Points')
            print('Samplerate before filtering:' + str(self.out['samplerate'])+ 'Hz')
            self.vdata = self.out['v1']
            self.ui.outputsamplerateentry.setText(str(self.out['samplerate']/1000))
        else:
            self.ui.outputsamplerateentry.setText(str(self.out['samplerate'] / 1000))
            print('chimera raw')
            print('Amount of points before filtering:' + str(self.out['i1raw'].shape))
            print('Samplerate before filtering:' + str(self.out['samplerate']))
            s=timer()
            ds_factor = self.out['samplerate']/self.Samplerate_aftr_LPF  #down sampling factor
            if ds_factor>1:
                Wn = round(2*self.LPfiltercutoff/(self.out['samplerate']), 4)   # [0,1] nyquist frequency
                b, a = signal.bessel(4, Wn, btype='low', analog=False) #4-th order digital filter
                Filt_sig = signal.filtfilt(b, a, self.out['i1raw'])
                print('signal filtered')
                    
                self.out['i1'] = scipy.signal.resample(Filt_sig, int(len(self.out['i1raw'])/ds_factor) )    
                print('signal resampled')
                self.out['samplerate'] = self.out['samplerate'] / ds_factor
                print('Samplerate after filtering:' + str(self.out['samplerate']))   
            else:
                print('Warning!!! Low Pass Filter value was inserted incorrect! LPF must be less than Output Samplerate')
                    

            self.data['i1'] = self.out['i1']
            self.vdata = np.ones(len(self.data)) * self.out['v1']
            e=timer()
            print('Chimera Loading: ' + 'time reqired: ' + str(e-s) + ' sec.')
    self.data['i1'] = self.out['i1']
    self.data['v1'] = self.out['v1']
    self.var['i1']=np.std(self.data['i1'])
    if str(os.path.splitext(self.datafilename)[1]) != '.log' and self.out['graphene']:
        self.data['i2'] = self.out['i2']
        self.data['v2'] = self.out['v2']
        self.var['i2']=np.std(self.data['i2'])
    self.outputsamplerate=self.out['samplerate']
    self.t = np.arange(len(self.out['i1']))  # Setting up time series
    print('Number of points = '+ str(len(self.t)))
    print('Length of experiment = '+str(len(self.out['i1'])/self.out['samplerate'] )+ ' s')
    self.t = self.t/(self.out['samplerate'])
    if loadandplot:
        self.Plot()


def ImportAxopatchData(datafilename):
    x=np.fromfile(datafilename, np.dtype('>f4'))
    f=open(datafilename, 'rb')
    graphene=0
    for i in range(0, 10):
        a=str(f.readline())
        #print(a)
        if 'Acquisition' in a or 'Sample Rate' in a:
            samplerate=int(''.join(i for i in a if i.isdigit()))/1000
        if 'FEMTO preamp Bandwidth' in a:
            femtoLP=int(''.join(i for i in a if i.isdigit()))
        if 'I_Graphene' in a:
            graphene=1
            print('This File Has a Graphene Channel!')
    end = len(x)
    if graphene:
        #pore current
        i1 = x[250:end-3:4]
        #graphene current
        i2 = x[251:end-2:4]
        #pore voltage
        v1 = x[252:end-1:4]
        #graphene voltage
        v2 = x[253:end:4]
        print('The femto was set to : {} Hz, if this value was correctly entered in the LabView!'.format(str(femtoLP)))
        output={'FemtoLowPass': femtoLP, 'type': 'Axopatch', 'graphene': 1, 'samplerate': samplerate, 'i1': i1, 'v1': v1, 'i2': i2, 'v2': v2, 'filename': datafilename}
    else:
        i1 = np.array(x[250:end-1:2])
        v1 = np.array(x[251:end:2])
        output={'type': 'Axopatch', 'graphene': 0, 'samplerate': samplerate, 'i1': i1, 'v1': v1, 'filename': datafilename}
    return output

def ImportChimeraRaw(datafilename):
    matfile=io.loadmat(str(os.path.splitext(datafilename)[0]))
    samplerate = np.float64(matfile['ADCSAMPLERATE'])
    data = np.fromfile(datafilename, np.dtype('<u2'))
    #buffersize=matfile['DisplayBuffer']
    
    TIAgain = np.int32(matfile['SETUP_TIAgain'])
    preADCgain = np.float64(matfile['SETUP_preADCgain'])
    currentoffset = np.float64(matfile['SETUP_pAoffset'])
    ADCvref = np.float64(matfile['SETUP_ADCVREF'])
    ADCbits = np.int32(matfile['SETUP_ADCBITS'])

    closedloop_gain = TIAgain * preADCgain
    bitmask = (2 ** 16 - 1) - (2 ** (16 - ADCbits) - 1)
    data = -ADCvref + (2 * ADCvref) * (data & bitmask) / 2 ** 16
    data = (data / closedloop_gain + currentoffset)
    data.shape = [data.shape[1], ]
    output = {'matfilename': str(os.path.splitext(datafilename)[0]),'i1raw': data, 'v1': np.float64(matfile['SETUP_mVoffset']), 'samplerate': np.int64(samplerate), 'type': 'ChimeraRaw', 'filename': datafilename} #final data representation
    return output

def ImportChimeraData(datafilename):
    matfile = io.loadmat(str(os.path.splitext(datafilename)[0]))
    samplerate = matfile['ADCSAMPLERATE']
    if samplerate<4e6:
        data = np.fromfile(datafilename, np.dtype('float64'))
        buffersize = matfile['DisplayBuffer']
        out = Reshape1DTo2D(data, buffersize)
        output = {'i1': out['i1'], 'v1': out['v1'], 'samplerate':float(samplerate), 'type': 'ChimeraNotRaw', 'filename': datafilename} #final data representation
    else:
        output = ImportChimeraRaw(datafilename)
    return output


def Reshape1DTo2D(inputarray, buffersize):  # For chimera data download
    npieces = np.uint16(len(inputarray)/buffersize)
    voltages = np.array([], dtype=np.float64)
    currents = np.array([], dtype=np.float64)
    #print(npieces)

    for i in range(1, int(npieces+1)):
        if i % 2 == 1:
            currents = np.append(currents, inputarray[int((i-1)*buffersize):int(i*buffersize-1)], axis=0)
            #print('Length Currents: {}'.format(len(currents)))
        else:
            voltages = np.append(voltages, inputarray[int((i-1)*buffersize):int(i*buffersize-1)], axis=0)
            #print('Length Voltages: {}'.format(len(voltages)))

    v1 = np.ones((len(voltages)), dtype=np.float64)
    i1 = np.ones((len(currents)), dtype=np.float64)
    v1[:]=voltages
    i1[:]=currents

    out = {'v1': v1, 'i1': i1}
    print('Currents:' + str(v1.shape))
    print('Voltages:' + str(i1.shape))
    return out

