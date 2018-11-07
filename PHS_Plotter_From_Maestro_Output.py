import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns

#Let's import all the data in this folder
filenames = os.listdir('.')

#Let's sort out files we don't need
numfiles = 0
count = 0
for filename in filenames:  
    
    #Let's find ONLY the text data files
    if not filename.endswith('.txt'): continue
    phs_data = np.array( np.loadtxt(filename, skiprows = 12, unpack = True) )
    counts = phs_data
    numfiles += 1 
    
    #Let's separte filename into element, isotope, time of test, and test number
    element, isotope, time, test, _ = filename.split('.')  
    elementup = element.upper()
    element = elementup[0] + element[1]
    channels = np.asarray(range(0,len(counts)))
    
    #Let's sort out the photopeaks for Ba-133, Cs-137, Co-60, and Na-22
    counts = np.asarray( phs_data )
    if element == 'Ba':
        counts_peak_1 = counts[60:120]
        channels_1 = np.arange(60,120)
        counts_peak_2 = counts[340:410]
        channels_2 = np.arange(340,410)
        tup_1 = channels_1, counts_peak_1
        tup_2 = channels_2, counts_peak_2
        peak_1 = '81'
        peak_2 = '356'
        channels_counts_1 = np.column_stack(tup_1)
        channels_counts_2 = np.column_stack(tup_2)
    elif element == 'Co':
        counts_peak_1 = counts[1075:1275]
        channels_1 = np.arange(1075,1275)
        counts_peak_2 = counts[1275:1430]
        channels_2 = np.arange(1275,1430)
        peak_1 = '1173'
        peak_2 = '1332'
        tup_1 = channels_1, counts_peak_1
        tup_2 = channels_2, counts_peak_2
        channels_counts_1 = np.column_stack(tup_1)
        channels_counts_2 = np.column_stack(tup_2)
    elif element == 'Cs':
        counts_peak_1 = counts[620:760]
        channels_1 = np.arange(620,760)
        counts_peak_2 = counts[620:760]
        channels_2 = np.arange(620,760)
        peak_1 = '662'
        peak_2 = '662'
        tup_1 = channels_1, counts_peak_1
        channels_counts_1 = np.column_stack(tup_1)
        channels_counts_2 = np.column_stack(tup_1)
    elif element == 'Na':
        counts_peak_1 = counts[475:600]
        channels_1 = np.arange(475,600)
        counts_peak_2 = counts[1200:1375]
        channels_2 = np.arange(1200,1375)
        peak_1 = '551'
        peak_2 = '1275'
        tup_1 = channels_1, counts_peak_1
        tup_2 = channels_2, counts_peak_2
        channels_counts_1 = np.column_stack(tup_1)
        channels_counts_2 = np.column_stack(tup_2)
        
    #Let's define line + guassian model functions
    def line(x, slope, intercept): 
        return intercept + slope * x

    def guassian(x, centroid, sigma, area):
        return (area / (np.sqrt(2 * np.pi) * sigma) ) * np.exp( ( - (x - centroid)**2 / (2 * (sigma**2) ) ) ) 

    def model(x, slope, intercept, centroid, sigma, area):
        return line(x, slope, intercept) + guassian(x, centroid, sigma, area)

    def power(x, coefficient, exponant):
        return coefficient * ( x**exponant )
    
    for elements in range(0,2):
    
        #Let's declare a few lists
        numfiles = 0
        cent = []
        resolution = []
        peakEnergy = []
        xs = []
        
        #Let's pick a peak
        if elements == 0:
            channel = channels_1
            count = counts_peak_1
            peak = peak_1
        elif elements == 1:
            channel = channels_2
            count = counts_peak_2
            peak = peak_2
            
        #Let's plot the initial data collected by the detector
        sns.set_context('poster')
        plt.figure()
        plt.plot(channel, count, 'k.', label = 'Collected Data', alpha = 0.4)
        plt.xlabel('Channel Number')
        plt.ylabel('Count Number')
    
        #Let's define initial guesses for the parameters
        A = np.sum(count)
        Xbar = channel[ np.argmax(count) ] * 1.0
        slp = (count[-1]-count[0])/(channel[-1] - channel[0])
        inter = count[0] - (slp*channel[0])
        sig = 0.07 * Xbar/2.35
        
        #Let's subtract the area under the line from the total area under the curve
        base = channel[-1] - channel[0]
        floor = count[-1]
        ceil = count[0]
        height = ceil - floor
        A -= 0.5 * base * height + base * floor
        
        #Let's use curve_fit to optimize the guassian + line model
        inGuess = [slp, inter, Xbar, sig, A]  
        params, covarience = opt.curve_fit(model, channel, count, p0 = inGuess)
        stdevs = np.sqrt( np.diag(covarience) )
        
        #Let's plot the new model, the initial guess, and detector data
        modelA = model(channel, params[0], params[1], params[2], params[3], params[4])
        res = params[3] * 2.35 / params[2]   
        res_val = str(res)
        plt.plot(channel, modelA, 'r-', label = 'Optimized Model', alpha = 1, linewidth = 2)
        plt.legend()
        titleplot = peak + 'keV for ' + element + '-' + isotope + ' with a resolution of ' + res_val
        plt.title(titleplot)
        plt.savefig( filename.replace('.txt', '.png') )

        #Let's calibrate pulse areas to energy values
        def convertChannelToEnergy(pulse_channel):
            slope = 1
            intercept = -23.7
            m = np.full(pulse_channel.shape, slope)
            b = np.full(pulse_channel.shape, intercept)
            pulse_energies = np.add( np.multiply(pulse_channel, m), b)
            return np.abs(pulse_energies)
    
    #Let's convert the channels now
    energies = convertChannelToEnergy(channels)
    
    #Let's plot the PHS
    plt.figure()
    plt.plot(energies, counts,'k-')
    plt.title('Pulse Height Spectrum for ' + element + '-' + isotope)
    plt.xlabel('Energy [keV]')
    plt.ylabel('Counts')
    plt.legend()

print "Fin."
