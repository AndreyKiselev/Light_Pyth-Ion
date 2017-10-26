from PyQt5 import QtGui, QtWidgets
import os #pathname manipulations


def load_and_merge(self):
    print('load and merge')
    self.ui.Text_Chimera.setText('Filename: ')
    try:
        ######## attempt to open dialog from most recent directory########
        datafilenametemp = QtGui.QFileDialog.getOpenFileName(parent=self, caption='Open file', directory=str(self.direc), filter="Amplifier Files(*.dat *.log  *.mat)") #creates turple with two values: the adress of the selected file and the line of Amplifier Filters
        if datafilenametemp[0] != '': #if you select some file the param will be not 0
            self.datafilename=datafilenametemp[0] #full path to the loaded file
            self.direc=os.path.dirname(self.datafilename) #Return the directory name
            self.Load() #Load module
    except IOError:#### if user cancels during file selection, exit loop#############
        pass
    
