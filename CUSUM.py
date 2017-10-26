import numpy as np
import pyqtgraph as pg


def detection(self, data,  dt, threshhold , minlength , maxstates ):


    logp = 0 #instantaneous log-likelihood for positive jumps
    logn = 0 #instantaneous log-likelihood for negative jumps
    cpos = np.zeros(len(data), dtype='float64') #cumulative log-likelihood function for positive jumps
    cneg = np.zeros(len(data), dtype='float64') #cumulative log-likelihood function for negative jumps
    gpos = np.zeros(len(data), dtype='float64') #decision function for positive jumps
    gneg = np.zeros(len(data), dtype='float64') #decision function for negative jumps
    edges = np.array([0], dtype='int64') #initialize an array with the position of the first subevent - the start of the event
    anchor = 0 #the last detected change
    length = len(data)
    mean = data[0]
    self.var = np.std(data)
    h = threshhold / self.var
    k = 1000
    nStates = 0
    varM = data[0]
    varS = 0
    mean = data[0]
    print('length data =' + str(length))
    v = np.zeros(length, dtype='float64')
    while k < length-100:
            k += 1     
            if nStates == 0: variance = np.var(data[anchor:k])
            mean = np.mean(data[anchor:k])
            if variance ==0: break
            logp = threshhold/variance * (data[k] - mean - threshhold/2.) #instantaneous log-likelihood for current sample assuming local baseline has jumped in the positive direction
            logn = -threshhold/variance * (data[k] - mean + threshhold/2.) #instantaneous log-likelihood for current sample assuming local baseline has jumped in the negative direction
            cpos[k] = cpos[k-1] + logp #accumulate positive log-likelihoods
            cneg[k] = cneg[k-1] + logn #accumulate negative log-likelihoods
            gpos[k] = max(gpos[k-1] + logp, 0) #accumulate or reset positive decision function
            gneg[k] = max(gneg[k-1] + logn, 0) #accumulate or reset negative decision function
            if (gpos[k] > h or gneg[k] > h):
                    
                    if (gpos[k] > h): #significant positive jump detected
                            jump = anchor + np.argmin(cpos[anchor:k+1]) #find the location of the start of the jump
                            if jump - edges[nStates] > minlength and np.abs(data[jump+minlength]-data[jump]) >threshhold/4:
                                    edges = np.append(edges, jump)
                                    nStates += 1
                                    print('EVENT!!!!! at ='+str(self.t[jump]))
                                    anchor = k# no data meaning at bad points!
                                     # away from bad point more! 
                                    cpos[0:len(cpos)] = 0 #reset all decision arrays
                                    cneg[0:len(cneg)] = 0
                                    gpos[0:len(gpos)] = 0
                                    gneg[0:len(gneg)] = 0
                    if (gneg[k] > h): #significant negative jump detected
                            jump = anchor + np.argmin(cneg[anchor:k+1])
                            if jump - edges[nStates] > minlength and np.abs(data[jump+minlength]-data[jump]) >threshhold/4:
                                    edges = np.append(edges, jump)
                                    nStates += 1
                                    print('EVENT!!!!! at ='+str(self.t[jump] ))
                                    anchor = k # no data meaning at bad points!
                                     # away from bad point more! 
                                    cpos[0:len(cpos)] = 0 #reset all decision arrays
                                    cneg[0:len(cneg)] = 0
                                    gpos[0:len(gpos)] = 0
                                    gneg[0:len(gneg)] = 0
                                 
                    
                    
                   
            if maxstates > 0:
                if nStates > maxstates:
                    print('too sensitive')
                    nStates = 0
                    k = 0
                    threshhold = threshhold*1.1
                    h = h*1.1
                    logp = 0 #instantaneous log-likelihood for positive jumps
                    logn = 0 #instantaneous log-likelihood for negative jumps
                    cpos = np.zeros(len(data), dtype='float64') #cumulative log-likelihood function for positive jumps
                    cneg = np.zeros(len(data), dtype='float64') #cumulative log-likelihood function for negative jumps
                    gpos = np.zeros(len(data), dtype='float64') #decision function for positive jumps
                    gneg = np.zeros(len(data), dtype='float64') #decision function for negative jumps
                    edges = np.array([0], dtype='int64') #initialize an array with the position of the first subevent - the start of the event
                    anchor = 0 #the last detected change
                    length = len(data)
                    mean = data[0]
                    nStates = 0
                    mean = data[0]
    edges = np.append(edges, len(data)-1) #mark the end of the event as an edge
    nStates += 1


    cusum = dict()
    cusum['CurrentLevels'] = [np.average(data[edges[i]+minlength:edges[i+1]]) for i in range(nStates)] #detect current levels during detected sub-event
    print('Length of time = ' + str(len(self.t)))
    print('Edges[-1] = ' + str(edges[-1]))
    cusum['EventDelay'] = edges  #locations of sub-events in the data
    cusum['Threshold'] = threshhold #record the threshold used
    print('Event = '+str( cusum['EventDelay']))
    cusum['jumps'] = np.diff(cusum['CurrentLevels'])
    #self.__recordevent(cusum)

    return cusum

def print_fitting(self, cusum):
    self.p1.plot(self.t[0: cusum['EventDelay'][1]], np.repeat(np.array([cusum['CurrentLevels'][0]]), len(self.t[0:cusum['EventDelay'][1]])), pen=pg.mkPen(color=(173, 27, 183), width=3))
    for number_cusum in np.arange(1,len(cusum['EventDelay'])-1 ,1): 
        self.p1.plot(self.t[cusum['EventDelay'][number_cusum]:cusum['EventDelay'][number_cusum+1]], np.repeat(np.array([cusum['CurrentLevels'][number_cusum]]), len(self.t[cusum['EventDelay'][number_cusum]:cusum['EventDelay'][number_cusum+1]])), pen=pg.mkPen(color=(173, 27, 183), width=3))
        self.p1.plot(np.linspace(self.t[cusum['EventDelay'][number_cusum]],self.t[cusum['EventDelay'][number_cusum]] + 0.00001 ,100),np.linspace(cusum['CurrentLevels'][number_cusum-1],cusum['CurrentLevels'][number_cusum],100), pen=pg.mkPen(color=(173, 27, 183), width=3))
    self.p1.plot(self.t[cusum['EventDelay'][-2]:-1], np.repeat(np.array([cusum['CurrentLevels'][-1]]), len(self.t[cusum['EventDelay'][-2]:-1])), pen=pg.mkPen(color=(173, 27, 183), width=3))
    self.p1.autoRange()




