import h5py
import glob
import os
from tables import *
import numpy as np
import pandas as pd
import ROOT
from ROOT import gROOT 
from ROOT import gStyle
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import json
#%jsroot on

ROOT.EnableImplicitMT()
gStyle.SetOptStat("0000000")

class Calibration:

    def __init__(self, data_pdf):
        self.data_pdf = data_pdf
        self.SP3 = None
        self.calibratedspectra = None
        self.Bi609 = 0
        self.Bi1120 = 0
        self.K1460 = 0
        self.Bi1764 = 0
        self.Tl2614 = 0

    def gaussian(self, x, kk, sigma, mu):
        return kk * (1/(sigma*math.sqrt(2*np.pi))) * np.exp(-0.5 * ((x-mu)/sigma)**2)

    def smooth(self, incrementalbinning=False):

        Time = self.data_pdf['tstart']
        Max = self.data_pdf['peak']

        ####################
        # EVALUATE SPECTRA #
        ####################
        
        SPECTRUM_MAX = []
        
        for j in range(len(Time)):
            if Max[j] >= 0:
                SPECTRUM_MAX.append(Max[j]) 
        
        SPECTRUM_MAX = np.array(SPECTRUM_MAX)
        
        #############################
        # 1 STEP: REBINNING DEL MAX #
        #############################
        
        BIN = 1
        N_BIN = int((np.max(SPECTRUM_MAX)-np.min(SPECTRUM_MAX))/BIN)
        BS = np.arange(np.min(SPECTRUM_MAX), np.max(SPECTRUM_MAX), BIN)
        hist, bin_edges = np.histogram(SPECTRUM_MAX, bins=BS)
        SP1 = np.zeros((N_BIN, 2))
        
        for cc in range(N_BIN-1):
            SP1[cc]=[BS[cc+1], hist[cc]]
        
        ################
        # PLOT SPECTRA #
        ################
        
        fig = plt.figure(figsize=(20,10))
        axsp0 = fig.add_subplot(1,1,1)
        plt.title('SPETTRO PRELIMINARE\n', size=24)
        axsp0.step(SP1[:,0], SP1[:,1], markersize=5, c='k', linewidth=1)
        plt.setp(axsp0.get_xticklabels(), fontsize=18)
        plt.setp(axsp0.get_yticklabels(), fontsize=18)
        axsp0.set_xlim(0,1500)
        plt.yscale('log')
        axsp0.set_ylabel('RPG101 counts', fontsize=24)
        plt.gcf().subplots_adjust(bottom=0.15)
        axsp0.set_xlabel('channel', fontsize=24)
        #plt.savefig("%s/02_SP_array.png" % self.description, facecolor='w', transparent=False)
        
        ######################################
        # 2 STEP: SMOOTHING DEI PICCHI SPURI #
        ######################################
        
        SP2 = []
        
        for j in range(0,2):
            SP2.append([SP1[j][0], SP1[j][1]])
        
        for j in range(2,len(SP1)):
            if SP1[j-1][1] <= 1.25*SP1[j-2][1] and SP1[j-1][1] <= 1.25*SP1[j][1]:
                SP2.append([SP1[j-1][0], SP1[j-1][1]])
            else:
                SP2.append([SP1[j-1][0], (SP1[j-2][1]+SP1[j][1])/2.0])
        
        SP2 = np.array(SP2)
        
        fig = plt.figure(figsize=(20,10))
        axsp0 = fig.add_subplot(1,1,1)
        plt.title('SPETTRO CON SMOOTHING DEI PICCHI SPURI\n', size=24)
        axsp0.step(SP2[:,0], SP2[:,1], markersize=5, c='k', linewidth=1)
        plt.setp(axsp0.get_xticklabels(), fontsize=18)
        plt.setp(axsp0.get_yticklabels(), fontsize=18)
        axsp0.set_xlim(0,1500)
        axsp0.set_ylim(5,8e4)
        plt.yscale('log')
        axsp0.set_ylabel('RPG101 counts', fontsize=24)
        plt.gcf().subplots_adjust(bottom=0.15)
        axsp0.set_xlabel('channel', fontsize=24)
        #plt.savefig("%s/03_SP_smooth.png" % self.description, facecolor='w', transparent=False)
        
        if incrementalbinning == True:
        
            ##################################
            # 3 STEP: REBINNING INCREMENTALE #
            ##################################
        
            # inizio e indice di rebinning
            s = 220
            r = 1.3
        
            SP3 = []
        
            for i in range(500):
                SP3.append([ np.mean([s+i**r, s+(i+1)**r]), np.sum(SP2[np.where((SP2[:,0]>=s+i**r) & (SP2[:,0]<s+(i+1)**r)),1]) ])
        
            SP3 = np.array(SP3)
        
            fig = plt.figure(figsize=(20,10))
            axsp0 = fig.add_subplot(1,1,1)
            plt.title('REBINNING INCREMENTALE\n', size=24)
            axsp0.step(SP3[:,0], SP3[:,1], markersize=5, c='k', linewidth=1)
            plt.setp(axsp0.get_xticklabels(), fontsize=18)
            plt.setp(axsp0.get_yticklabels(), fontsize=18)
            axsp0.set_xlim(0,1500)
            plt.yscale('log')
            axsp0.set_ylabel('counts', fontsize=24)
            plt.gcf().subplots_adjust(bottom=0.15)
            axsp0.set_xlabel('channel', fontsize=24)
            #plt.savefig("%s/04_SP_rebin.png" % self.description, facecolor='w', transparent=False)
        
        SP3 = SP2
        
        #np.savetxt("%s/spectra.txt" % self.description, SP3, delimiter=',')
        np.savetxt("spectraSP3.txt", SP3, delimiter=',')
        self.SP3 = SP3
        return self.SP3

    def peakfinder(self, jsonconfigfile):

        # Carica i dati dal file JSON
        with open(jsonconfigfile, 'r') as file:
            data = json.load(file)

        # Organizza i dati secondo la tua struttura
        i1_Bi609 = data["Bi609"]["i1"]
        i2_Bi609 = data["Bi609"]["i2"]
        
        i1_Bi1120 = data["Bi1120"]["i1"]
        i2_Bi1120 = data["Bi1120"]["i2"]
        
        i1_K1460 = data["K1460"]["i1"]
        i2_K1460 = data["K1460"]["i2"]
        
        i1_Bi1764 = data["Bi1764"]["i1"]
        i2_Bi1764 = data["Bi1764"]["i2"]
        
        i1_Tl2614 = data["Tl2614"]["i1"]
        i2_Tl2614 = data["Tl2614"]["i2"]
        
        # Stampa i risultati
        print(f"i1_Bi609 = {i1_Bi609}")
        print(f"i2_Bi609 = {i2_Bi609}")
        
        print(f"i1_Bi1120 = {i1_Bi1120}")
        print(f"i2_Bi1120 = {i2_Bi1120}")
        
        print(f"i1_K1460 = {i1_K1460}")
        print(f"i2_K1460 = {i2_K1460}")
        
        print(f"i1_Bi1764 = {i1_Bi1764}")
        print(f"i2_Bi1764 = {i2_Bi1764}")
        
        print(f"i1_Tl2614 = {i1_Tl2614}")
        print(f"i2_Tl2614 = {i2_Tl2614}")
 
        SP3 = self.SP3
        
        # 1 - 214Bi - 609 keV - tra 330 e 380
        SP3_PZ = SP3[np.where((SP3[:,0]>i1_Bi609) & (SP3[:,0]<i2_Bi609))]
        p0 = [1, 1., np.mean(SP3_PZ[:,0])]
        popt, pcov = curve_fit(self.gaussian, SP3_PZ[:,0], SP3_PZ[:,1], p0=p0)
        FAKEBi609 = []
        X = np.arange(i1_Bi609,i2_Bi609,0.5)
        for j in range(len(X)):
            FAKEBi609.append([X[j], popt[0]*(1/(popt[1]*math.sqrt(2*np.pi))) * np.exp(-0.5 * ( (X[j]-popt[2])/popt[1] )**2) ])
        FAKEBi609 = np.array(FAKEBi609)
        self.Bi609 = popt[2]
        print("Bi609 ", self.Bi609)
        
        # 2 - 214Bi - 1120 keV - tra 550 e 620
        SP3_PZ = SP3[np.where((SP3[:,0]>i1_Bi1120) & (SP3[:,0]<i2_Bi1120))]
        p0 = [1, 1., np.mean(SP3_PZ[:,0])]
        popt, pcov = curve_fit(self.gaussian, SP3_PZ[:,0], SP3_PZ[:,1], p0=p0)
        FAKEBi1120 = []
        X = np.arange(i1_Bi1120,i2_Bi1120,0.5)
        for j in range(len(X)):
            FAKEBi1120.append([X[j], popt[0]*(1/(popt[1]*math.sqrt(2*np.pi))) * np.exp(-0.5 * ( (X[j]-popt[2])/popt[1] )**2) ])
        FAKEBi1120 = np.array(FAKEBi1120)
        self.Bi1120 = popt[2]
        print("Bi1120 ", self.Bi1120)
        
        
        # 3 - 40K - 1460 keV - tra 700 e 850
        SP3_PZ = SP3[np.where((SP3[:,0]>i1_K1460) & (SP3[:,0]<i2_K1460))]
        p0 = [1, 1., np.mean(SP3_PZ[:,0])]
        popt, pcov = curve_fit(self.gaussian, SP3_PZ[:,0], SP3_PZ[:,1], p0=p0)
        FAKEK1460 = []
        X = np.arange(i1_K1460,i2_K1460,0.5)
        for j in range(len(X)):
            FAKEK1460.append([X[j], popt[0]*(1/(popt[1]*math.sqrt(2*np.pi))) * np.exp(-0.5 * ( (X[j]-popt[2])/popt[1] )**2) ])
        FAKEK1460 = np.array(FAKEK1460)
        self.K1460 = popt[2]
        print("K1460 ", self.K1460)
        
        # 4 - 214Bi - 1764 keV - tra 830 e 930
        SP3_PZ = SP3[np.where((SP3[:,0]>i1_Bi1764) & (SP3[:,0]<i2_Bi1764))]
        p0 = [1, 1., np.mean(SP3_PZ[:,0])]
        popt, pcov = curve_fit(self.gaussian, SP3_PZ[:,0], SP3_PZ[:,1], p0=p0)
        FAKEBi1764 = []
        X = np.arange(i1_Bi1764,i2_Bi1764,0.5)
        for j in range(len(X)):
            FAKEBi1764.append([X[j], popt[0]*(1/(popt[1]*math.sqrt(2*np.pi))) * np.exp(-0.5 * ( (X[j]-popt[2])/popt[1] )**2) ])
        FAKEBi1764 = np.array(FAKEBi1764)
        self.Bi1764 = popt[2]
        print("Bi1764 ", self.Bi1764)
        
        # 5 - 208Tl - 2614 keV - tra 1250 e 1350
        SP3_PZ = SP3[np.where((SP3[:,0]>i1_Tl2614) & (SP3[:,0]<i2_Tl2614))]
        p0 = [1, 1., np.mean(SP3_PZ[:,0])]
        popt, pcov = curve_fit(self.gaussian, SP3_PZ[:,0], SP3_PZ[:,1], p0=p0)
        FAKETl2614 = []
        X = np.arange(i1_Tl2614,i2_Tl2614,0.5)
        for j in range(len(X)):
            FAKETl2614.append([X[j], popt[0]*(1/(popt[1]*math.sqrt(2*np.pi))) * np.exp(-0.5 * ( (X[j]-popt[2])/popt[1] )**2) ])
        FAKETl2614 = np.array(FAKETl2614)
        self.Tl2614 = popt[2]
        print("Tl2614 ", self.Tl2614)
        
        fig = plt.figure(figsize=(20,10))
        axsp0 = fig.add_subplot(1,1,1)
        plt.title('SPETTRO CON IDENTIFICAZIONE RIGHE DI EMISSIONE\n', size=24)
        axsp0.step(SP3[:,0], SP3[:,1], markersize=5, c='k', linewidth=1)
        axsp0.step(FAKEBi609[:,0], FAKEBi609[:,1], markersize=5, c='r', linewidth=4, alpha=0.5)
        axsp0.step(FAKEBi1120[:,0], FAKEBi1120[:,1], markersize=5, c='r', linewidth=4, alpha=0.5)
        axsp0.step(FAKEK1460[:,0], FAKEK1460[:,1], markersize=5, c='r', linewidth=4, alpha=0.5)
        axsp0.step(FAKEBi1764[:,0], FAKEBi1764[:,1], markersize=5, c='r', linewidth=4, alpha=0.5)
        axsp0.step(FAKETl2614[:,0], FAKETl2614[:,1], markersize=5, c='r', linewidth=4, alpha=0.5)
        plt.setp(axsp0.get_xticklabels(), fontsize=18)
        plt.setp(axsp0.get_yticklabels(), fontsize=18)
        axsp0.set_xlim(0,1500)
        axsp0.set_ylim(5,8e4)
        plt.axvline(x=self.Bi609, color='red')
        plt.axvline(x=self.Bi1120, color='indigo')
        plt.axvline(x=self.K1460, color='orange')
        plt.axvline(x=self.Bi1764, color='blue')
        plt.axvline(x=self.Tl2614, color='green')
        plt.yscale('log')
        axsp0.set_ylabel('RPG101 counts', fontsize=24)
        plt.gcf().subplots_adjust(bottom=0.15)
        axsp0.set_xlabel('channel', fontsize=24)
        #plt.savefig("%s/04_SP_lines.png" % TEST, facecolor='w', transparent=False)

    #fittype = 0 linear
    #fittype = 1 quadratic
    #fittype = 2 cubic
    #return calibrated spectra
    def calibrate(self, fittype=0):
        SP3 = self.SP3
        ######################################
        # CALIBRAZIONE SULLA BASE DEI PICCHI #
        ######################################
        
        LINES_CHN = np.array([self.Bi609, self.Bi1120, self.K1460, self.Bi1764, self.Tl2614])
        LINES_ERG = np.array([609, 1120, 1460, 1764, 2614])
        
        # FIT LINEARE
        if fittype == 0:
            m,b = np.polyfit(LINES_CHN, LINES_ERG, 1)
        
            fit_equation = "m = %f, q = %f" % (m,b)
        
            SP4 = []
            print(len(SP3))
            for i in range(len(SP3)):
                SP4.append([m*SP3[i][0]+b, SP3[i][1]])
            SP4 = np.array(SP4)
        
        # FIT QUADRATICO
        if fittype == 1:
            def parabola(x, a, b, c):
                return a*x**2 + b*x + c
        
            p0 = [-1, 2, -150]
            popt, pcov = curve_fit(parabola, LINES_CHN, LINES_ERG, p0=p0)
        
            a = float(popt[0])
            b = float(popt[1])
            c = float(popt[2])
        
            fit_equation = "a = %f, b = %f, c = %f" % (a,b,c)
        
            SP4 = []
            for i in range(len(SP3)):
                #SP4.append([a*SP3[i][0]**2 + b*SP3[i][0] + c, SP3[i][1]])
                SP4.append([parabola(SP3[i][0], a, b, c), SP3[i][1]])
            SP4 = np.array(SP4)
        
        # FIT CUBICO
        if fittype == 2:
            def cub3(x, a, b, c, d):
                return a*x**3 + b*x**2 + c*x + d
        
            p0 = [10, 20, -150, -1700]
            popt, pcov = curve_fit(cub3, LINES_CHN, LINES_ERG, p0=p0)
        
            a = float(popt[0])
            b = float(popt[1])
            c = float(popt[2])
            d = float(popt[3])
        
            fit_equation = "a = %f, b = %f, c = %f, d = %f" % (a,b,c,d)
        
            SP4 = []
            for i in range(len(SP3)):
                SP4.append([cub3(SP3[i][0], a, b, c, d), SP3[i][1]])
            SP4 = np.array(SP4)
        
        self.calibratedspectra = SP4
        
        fig = plt.figure(figsize=(20,10))
        axsp0 = fig.add_subplot(1,1,1)
        plt.title('SPETTRO CALIBRATO (fit equation %s)\n' % fit_equation, size=24)
        axsp0.step(SP4[:,0], SP4[:,1], markersize=5, c='k', linewidth=1)
        plt.setp(axsp0.get_xticklabels(), fontsize=18)
        plt.setp(axsp0.get_yticklabels(), fontsize=18)
        axsp0.set_xlim(0,3000)
        axsp0.set_ylim(5,8e4)
        plt.axvline(x=609, color='red')
        plt.axvline(x=1120, color='indigo')
        plt.axvline(x=1460, color='orange')
        plt.axvline(x=1764, color='blue')
        plt.axvline(x=2614, color='green')
        plt.text(609, 3e4, "  214Bi\n  609 keV", color='red', size=15)
        plt.text(1120, 3e4, "  214Bi\n  1120 keV", color='indigo', size=15)
        plt.text(1460, 3e4, "  40K\n  1460 keV", color='orange', size=15)
        plt.text(1764, 3e4, "  214Bi\n  1764 keV", color='blue', size=15)
        plt.text(2614, 3e4, "  208Tl\n  2614 keV", color='green', size=15)
        plt.yscale('log')
        axsp0.set_ylabel('RPG101 counts', fontsize=24)
        plt.gcf().subplots_adjust(bottom=0.15)
        axsp0.set_xlabel('energy [keV]', fontsize=24)
        #plt.savefig("%s/05_SP_calibrated.png" % TEST, facecolor='w', transparent=False)

        #print(type(self.calibratedspectra))
        #print(np.histogram_bin_edges(self.calibratedspectra))
        return self.calibratedspectra
    

class ProcessDL2:

    def __init__(self, patternfile, description=None, json_config=None):
        self.pattern = patternfile
        
        self.filter = ""
        self.spectrapeak = None
        self.spectraintegral1 = None
        self.tstart = 0
        self.tend = 0
        self.nevents = 0
        self.data_rdf = None #Root data frame
        self.data_pdf = None #Pandas data frame
        self.fittype = -1
        if json_config is not None:
            self.read_config(json_config)
        if description is not None:
            self.description  = description

    def read_config(self, json_file_path):
        try:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)
    
                # Estrai i valori dal JSON
                self.RP = data.get("RP")
                self.description = data.get("description")
                self.fittype = data.get("fittype")
                self.temperature = data.get("temperature")
                self.parameters = data.get("parameters")
    
                # Stampa i valori estratti
                print("RP:", self.RP)
                print("Description:", self.description)
                print("Fittype:", self.fittype)
                print("Temperature:", self.temperature)
                print("Parameters:", self.parameters)
    
                # Estrai parametri specifici in base al fittype
                if self.fittype == 0 and self.parameters:
                    m = self.parameters.get("m")
                    b = self.parameters.get("b")
                    print("Fittype 0 Parameters: m={}, b={}".format(m, b))
                elif self.fittype == 1 and self.parameters:
                    a = self.parameters.get("a")
                    b = self.parameters.get("b")
                    c = self.parameters.get("c")
                    print("Fittype 1 Parameters: a={}, b={}, c={}".format(a, b, c))
                elif self.fittype == 2 and self.parameters:
                    a = self.parameters.get("a")
                    b = self.parameters.get("b")
                    c = self.parameters.get("c")
                    d = self.parameters.get("d")
                    print("Fittype 2 Parameters: a={}, b={}, c={}, d={}".format(a, b, c, d))
                else:
                    print("Invalid fittype or missing parameters.")

        except FileNotFoundError:
            print("File not found:", json_file_path)
        except json.JSONDecodeError:
            print("Invalid JSON format in:", json_file_path)

    
    #h5 or csv
    def load(self, filetype="h5", nmax=800):
        files = sorted(glob.glob(self.pattern))
        print(len(files))
        result=[]
        n=0
        
        if filetype == "csv":
            for n, txtfile in enumerate(files):
                #print(n)
                df1 = pd.read_csv(txtfile, sep="\t", nrows=30000000)
                if n == 0:
                    frames = [df1]
                else:
                    frames = [df1, result]
                result = pd.concat(frames)
                if n>nmax:
                    break          
        elif filetype == "h5":
            h5file = open_file(files[0], mode="r")
            group = h5file.get_node("/dl2/eventlist")
            for n, h5file in enumerate(h5_files):
                #print(n)
                df1 = pd.read_hdf(h5file, key="/dl2/eventlist")
                if n == 0:
                    frames = [df1]
                else:
                    frames = [df1, result]
                result = pd.concat(frames)
                #print(n, h5file)
                if n>nmax:
                    break
        else:
            print("not supported file type")
            
        print(len(result))
        print(n)
        result=result.sort_values("tstart", ascending=True).reset_index(drop=True)
        #extract data
        halflife=result['halflife'].to_numpy()
        temp = result['temp'].to_numpy()
        peak=result['peak'].to_numpy()
        tstart=result['tstart'].to_numpy()
        index_peak=result['index_peak'].to_numpy()
        mult=result['mult'].to_numpy()
        try:
            n_waveform=result['n_waveform'].to_numpy()
        except:
            n_waveform=result['N_Waveform'].to_numpy()
        integral1=result['integral1'].to_numpy()
        integral2=result['integral2'].to_numpy()
        integral3=result['integral3'].to_numpy()
    
        tdiff = np.diff(tstart)
        tdiff = np.append(tdiff, 0)

        # Estrai parametri specifici in base al fittype
        if self.fittype == 0 and self.parameters:
            m = self.parameters.get("m")
            b = self.parameters.get("b")
            print("Fittype 0 Parameters: m={}, b={}".format(m, b))
            energy = []
            for i in range(len(peak)):
                energy.append( m*peak[i]+b )
            energy = np.array(energy)

        elif self.fittype == 1 and self.parameters:
            a = self.parameters.get("a")
            b = self.parameters.get("b")
            c = self.parameters.get("c")
            print("Fittype 1 Parameters: a={}, b={}, c={}".format(a, b, c))
        elif self.fittype == 2 and self.parameters:
            a = self.parameters.get("a")
            b = self.parameters.get("b")
            c = self.parameters.get("c")
            d = self.parameters.get("d")
            print("Fittype 2 Parameters: a={}, b={}, c={}, d={}".format(a, b, c, d))
        else:
            print("Invalid fittype or missing parameters.")
        

        self.tstart = result.tstart[0]
        self.tend = result.tstart[len(tstart)-1]
        self.nevents = len(tstart)
        
        print('nsecs ' + str(result.tstart[len(tstart)-1] - result.tstart[0]))
    
        rdf = ROOT.RDF.MakeNumpyDataFrame({'tstart': tstart, 'tdiff': tdiff, 'halflife': halflife, 'index_peak': index_peak, 'n_waveform': n_waveform,  'temp': temp, 'peak': peak, 'mult': mult, 'integral1':integral1, 'integral2':integral2, 'integral3':integral3, 'energy':energy})
        #df.Describe()

        pandas_df = rdf.AsNumpy()
    
        #colonne_ordine = [
        #"n_waveform", "mult", "tstart", "index_peak", "peak",
        #"integral1", "integral2", "integral3", "halflife", "temp"
        #]
        #result = result[colonne_ordine]
        #result["n_waveform"] = result["n_waveform"].astype(int)
        #result["mult"] = result["mult"].astype(int)
        #result["index_peak"] = result["index_peak"].astype(int)
        #result["peak"] = result["peak"].astype(int)
        #result["halflife"] = result["halflife"].astype(int)
        #result = result

        #root data frame
        self.data_rdf = rdf
        #pandas data frame
        self.data_pdf = pandas_df
        return self.data_rdf

    #filter the already loaded data frame
    def apply_filter(self, filter):
        self.data_rdf = self.data_rdf.Filter(filter)
        self.filter = filter
        self.description = self.description + "-" + filter
        self.data_pdf = self.data_rdf.AsNumpy()
        self.spectrapeak = None
        self.spectraintegral1 = None
        self.tstart = self.data_pdf['tstart'][0]
        self.nevents = len(self.data_pdf['tstart'])
        self.tend = self.data_pdf['tstart'][self.nevents-1]
        
    def histo_halflife(self):
        hist1=self.data_rdf.Histo1D(("halflife/"+self.description, "halflife_"+self.description, 100, 0, 200), "halflife")
        hist1.Scale(1/hist1.Integral())
        return hist1

    def histo_index_peak(self):
        hist1=self.data_rdf.Histo1D(("index_peak/"+self.description, "index_peak/"+self.description, 180, 0, 18000), "index_peak")
        hist1.Scale(1/hist1.Integral())
        return hist1

    def histo_mult(self):
        hist1=self.data_rdf.Histo1D(("mult/"+self.description, "mult/"+self.description, 20, 0, 20), "mult")
        hist1.Scale(1/hist1.Integral())
        return hist1

    def histo_n_waveform(self):
        hist1=self.data_rdf.Histo1D(("n_waveform/"+self.description, "n_waveform/"+self.description, 20, 0, 2000), "n_waveform")
        hist1.Scale(1/hist1.Integral())
        return hist1
        
    def spectra_peak(self, xmin=0, xmax=2000, ndivbin=4):
        nbins=int((xmax-xmin)/ndivbin)
        spectrapeak=self.data_rdf.Histo1D(("spectrapeak/"+self.description, "spectrapeak/"+self.description, nbins, xmin, xmax), "peak")
        spectrapeak.Scale(1/spectrapeak.Integral())
        self.spectrapeak = spectrapeak
        return spectrapeak

    def spectra_energy(self, xmin=0, xmax=8000, ndivbin=4):
        nbins=int((xmax-xmin)/ndivbin)
        spectraenergy=self.data_rdf.Histo1D(("spectraenergy/"+self.description, "spectraenergy/"+self.description, nbins, xmin, xmax), "energy")
        spectraenergy.Scale(1/spectraenergy.Integral())
        self.spectraenergy = spectraenergy
        return spectraenergy

    def spectra_integral1(self, xmin=0, xmax=200000, ndivbin=10):
        nbins=int((xmax-xmin)/ndivbin)
        spectraintegral1=self.data_rdf.Histo1D(("integral1/"+self.description, "integral1/"+self.description, nbins, xmin, xmax), "integral1")
        spectraintegral1.Scale(1/spectraintegral1.Integral())
        self.spectraintegral1 = spectraintegral1
        return spectraintegral1

    def lc(self, binsizesecs = 10, tmin=0, tmax=0):
        if tmin==0:
            tmin = self.tstart
        if tmax==0:
            tmax = self.tend
        nbins = (int(tmax) - int(tmin) + 1) / binsizesecs
        lc = self.data_rdf.Histo1D(("lc", str(binsizesecs)+" s - "+str(int(tmin))+"-"+str(int(tmax))+"", int(nbins), tmin, tmax), "tstart")
        return lc

    def calibration(self, jsonconfigfile, fittype=0):
        cal = Calibration(self.data_pdf)
        cal.smooth()
        cal.peakfinder(jsonconfigfile)
        return cal.calibrate(fittype)
     
    def _merge_csv(self, directory):
        out = 'merged.csv'

        try:
            os.system("rm " + directory + "/" + out)
        except:
            print(outfilename + " not present")
            
        file_list = sorted([f for f in os.listdir(directory) if f.endswith('.txt')])
    
        if not file_list:
            print("No text files found in the directory.")
            return
        
        merged_content = ""
        
        for i, file_name in enumerate(file_list):
            with open(os.path.join(directory, file_name), 'r') as file:
                content = file.read()
                if i == 0:
                    merged_content += content
                else:
                    # Remove the first line from all files except the first one
                    _, _, rest_of_content = content.partition('\n')
                    merged_content += rest_of_content
        
        output_file_path = os.path.join(directory, out)
        
        with open(output_file_path, 'w') as output_file:
            output_file.write(merged_content)
        
        print(f"Files merged successfully. Merged content saved to {output_file_path}")
 
    def _drawH(self, listahisto, legend=False, nameCanvas="c1", title="", xtitle="", xrange=-1):
        c = ROOT.TCanvas(nameCanvas, nameCanvas, 1024, 768)
        c.SetLogy()
        
        indexcolor=1
        for histo in listahisto:
            histo.SetLineColor(indexcolor)
            if indexcolor == 1:
                histo.SetTitle(title)
                histo.GetXaxis().SetTitle(xtitle)
                if xrange != -1:
                    histo.GetXaxis().SetRangeUser(xrange[0], xrange[1])
                histo.Draw("HIST")

            else:
                histo.Draw("SAMEHIST")
            indexcolor = indexcolor + 1
        if legend:
            c.BuildLegend()
        return c


    def _drawEnergyBars(self):
        energies=[511, 609, 1120, 1460, 1764, 2614]
        le=[]
        for energy in energies:
            print(energy)
            la = ROOT.TLine(energy, 0, energy, 1)
            la.SetLineStyle(7)
            la.SetLineColor(2)
            la.Draw()
            le.append(la)

    def _drawLC(self, listalc, legend=False, nameCanvas="clc1", title=""):
        c = ROOT.TCanvas(nameCanvas, nameCanvas, 1024, 768)
        indexcolor=1
        for histo in listalc:
            histo.SetLineColor(indexcolor)
            if indexcolor == 1:
                histo.SetTitle(title)
                histo.Draw("HIST")
            else:
                histo.Draw("SAMEHIST")
            indexcolor = indexcolor + 1
        if legend:
            c.BuildLegend()
        return c

    def _sum(self, hist, hist2, normalize=False):
        hist3=hist.Clone()
        for j in range(0, hist.GetNbinsX()):
            hist3.SetBinContent(j, hist.GetBinContent(j)+hist2.GetBinContent(j))
        if normalize == True:
            hist3.Scale(1/hist3.Integral())
        return hist3
        
    def _smooth(self, hist, iterations=3):
        hist2=hist.Clone()
        for j in range(2, hist2.GetNbinsX() - 1):
            hist2.SetBinContent(j, hist.GetBinContent(j))
            if hist.GetBinContent(j) == 0 and hist.GetBinContent(j-1) != 0:
                hist2.SetBinContent(j, hist.GetBinContent(j-1))
            else:
                if hist.GetBinContent(j) == 0 and hist.GetBinContent(j-2) != 0:
                    hist2.SetBinContent(j, hist.GetBinContent(j-2))    
        for k in range(1, iterations):
            for j in range(2, hist2.GetNbinsX() - 1):
                if hist2.GetBinContent(j-1) > 1.25*hist2.GetBinContent(j):
                    hist2.SetBinContent(j-1, hist2.GetBinContent(j))
                #if hist2.GetBinContent(j-1) <= 1.25*hist2.GetBinContent(j-2) and hist2.GetBinContent(j-1) <= 1.25*hist2.GetBinContent(j):
                #    hist2.SetBinContent(j-1, hist2.GetBinContent(j-1))
                #else:
                #    val = float((hist2.GetBinContent(j-2)+hist2.GetBinContent(j))/2.0)
                #    hist2.SetBinContent(j-1, val)
        return hist2
