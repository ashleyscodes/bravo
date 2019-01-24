import os
import numpy as np
import matplotlib.pyplot as plt

#import all data
filenames = os.listdir('.')

#sort out files we don't need
numfiles = 0
count = 0
for filename in filenames:  
    
    #find ONLY the rp data files
    if not filename.endswith('.txt'): continue
    waveforms = np.array( np.loadtxt(filename, skiprows = 9, unpack = True) )
    numfiles += 1 
    
    #separte filename into element, isotope, energy
    element, isotope, test = filename.split('_')
    test, _ = test.split('.')
    
    #narrow down pulses and only plot a few
    def selectPulses(pulses):
        resonable_pulses = pulses[0:len(pulses),0:1]
        return resonable_pulses
    
    #Plot a few wave forms
    waveforms = selectPulses(waveforms)
    plt.figure()
    plt.plot(waveforms, 'k-')
    plt.title('Digitized Pulses')
    plt.xlabel('Trace Length')
    plt.ylabel('Pulse Height [Channel]')

print "Пoka" 
