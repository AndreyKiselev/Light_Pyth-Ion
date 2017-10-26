import numpy as np
import pyqtgraph as pg


def detect_cusum(self, data,  dt, threshhold , minlength , maxstates ):


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
    h = threshhold / self.var
    k = self.safety_reg - self.control1
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
                                    print('EVENT!!!!! at ='+str((self.control1 + jump) / self.out['samplerate'] ))
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
                                    print('EVENT!!!!! at ='+str((self.control1 + jump) / self.out['samplerate'] ))
                                    anchor = k # no data meaning at bad points!
                                     # away from bad point more! 
                                    cpos[0:len(cpos)] = 0 #reset all decision arrays
                                    cneg[0:len(cneg)] = 0
                                    gpos[0:len(gpos)] = 0
                                    gneg[0:len(gneg)] = 0
                                 
                    
                    
                   
            if maxstates > 0:
                if nStates > 10:
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
    edges = np.append(edges, len(data)) #mark the end of the event as an edge
    nStates += 1


    cusum = dict()
    cusum['CurrentLevels'] = [np.average(data[edges[i]+minlength:edges[i+1]]) for i in range(nStates)] #detect current levels during detected sub-events
    cusum['EventDelay'] = edges  #locations of sub-events in the data
    cusum['Threshold'] = threshhold #record the threshold used
    print('Event = '+str((self.control1 ) +cusum['EventDelay']))
    cusum['jumps'] = np.diff(cusum['CurrentLevels'])
    #self.__recordevent(cusum)

    return cusum
