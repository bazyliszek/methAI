'''
Created on 8 Sep 2019

@team: marcin&jan
'''
import pickle
import os
from numpy import mean, std

'''
class TissueDataset():
    
    NAValue = -1
    def __init__(self, cancerFolderName, healthyFolderName, matchToFolderName):
        
        #convert genome to latest
        genomeMap = self.createMap(matchToFolderName)
        print("Loaded genome map successfully. Found", len(genomeMap), "locations")
        
        #read betavalues    
        cancerBetaValues = self.loadBetaValues(cancerFolderName, genomeMap)
        healthyBetaValues = self.loadBetaValues(healthyFolderName, genomeMap)
        print("Loaded beta values successfully. Found",len(cancerBetaValues) + len(healthyBetaValues), "patients with", len(cancerBetaValues[0]), "locations")
        cancerBetaValues = self.throwOutNAPatients(cancerBetaValues, 0.15)
        healthyBetaValues = self.throwOutNAPatients(healthyBetaValues, 0.15)
        print("After removing NA Patients still have",len(cancerBetaValues) + len(healthyBetaValues), "patients with", len(cancerBetaValues[0]), "locations")
        self.cancerBetaValues, self.healthyBetaValues = self.throwOutNAPositions(cancerBetaValues, healthyBetaValues)
        print("After removing NA Positions still have",len(cancerBetaValues) + len(healthyBetaValues), "patients with", len(cancerBetaValues[0]), "locations")
        
        
            
    def createMap(self, folder):
        genomeMap = {}
        directory = os.listdir(folder)[0]
        files = os.listdir(os.path.join(folder, directory))
        for currentFileName in files:
            if(currentFileName[:3] == "jhu" and "HumanMethylation450" in currentFileName and currentFileName[-4:] == ".txt"):
                with open(os.path.join(folder, directory, currentFileName)) as currentFile:
                    for line in currentFile:
                        separated = line.split("\t")
                        genomeMap[separated[0].strip()] = separated[2].strip() + ':' + separated[3].strip()
                break
        return genomeMap
    
    
    
    def loadBetaValues(self, folder, genomeMap):
        subdirs = os.listdir(folder)
        
        patients = []
        #maxFiles = 5
        i = 0
        for currentDirectory in subdirs:
            i += 1
            #if(i >= maxFiles):
            #    break
            
            #try:
            betaDict = {}
            try:
                files = os.listdir(os.path.join(folder, currentDirectory))
            except:
                print("skipping directory", currentDirectory,"as it could not be opened")
                continue
                
            print("reading directory", currentDirectory, "Found", len(files), "files")
            
            for currentFileName in files:
                if(currentFileName[:3] == "jhu" and "HumanMethylation450" in currentFileName and currentFileName[-4:] == ".txt"):
                    with open(os.path.join(folder, currentDirectory, currentFileName)) as currentFile:
                        #print("dir", i, " file is open. Reading lines")
                        j = 0
                        for line in currentFile:
                            #print("reading line", j, "of file in dir", i)
                            try:
                                separated = line.split("\t")
                                #print("file contains",len(separated),"columns")
                                probe = separated[0].strip()
                                location = genomeMap[probe]
                                try:
                                    betaDict[location] = float(separated[1].strip())
                                except ValueError:
                                    #any value that can not be interpreted as a float is set to NA
                                    betaDict[location] = self.NAValue
                            except KeyError as e:
                                print("exception line", j, "of file in dir", i)
                                print(str(e))
                                print("checking for", probe)
                                pass
                            j+=1
                                #skip any lines that can't be matched (presumably only the header
            
            if(len(betaDict) > 0):#skip the folders that don't use 450k probes
                patients.append(betaDict)
                print("adding patient to dictionary. Current length =", len(patients))
            #except:
            #    pass
            #    #if it finds something that isn't a directory then just ignore it
            #    #there shouldn't be anything else in a freshly downloaded dataset but
            #    #we stuck various other files in there at different points
        return patients
    
    
    def throwOutNAPatients(self, group, minPercent):
        print("Throwing out NA Patients")
        newGroup = []
        for patient in range(0, len(group)):
            #print("working on patient", patient)
            NAs = 0
            betas = 0
            for key in group[patient].keys():
                #print("working on patient", patient, "key", key)
                if(group[patient][key] == self.NAValue):
                    NAs += 1
                else:
                    betas += 1
            print("found", betas,"betas and", NAs, "NAs in patient", patient)
            try:
                if(betas / float(betas+NAs) >= minPercent):
                    newGroup.append(group[patient])
            except:
                print("empty file at", len(newGroup), self.name)
        return newGroup
    
    
    def throwOutNAPositions(self, cancerGroup, healthyGroup):
        print("Throwing out NA Positions")        
        keysToKeep = []
        discarded = 0
        
        #First find which positions have the required amount of values. Store these in keys to keep. Also throw out anything with an * in the key
        for key in cancerGroup[0].keys():
            
            if(not '*' in key):
                discard = False
                for patient in cancerGroup:
                    if(patient[key] == self.NAValue):
                        discard = True
                        break
                for patient in healthyGroup:
                    if(patient[key] == self.NAValue):
                        discard = True
                        break
                if(not discard):
                    keysToKeep.append(key)
                else:
                    discarded += 1
                    
        #then remake each group only keeping the selected keys
        newCancerGroup = []
        for patient in cancerGroup:
            newPatient = {}
            for key in keysToKeep:
                newPatient[key] = patient[key]
            newCancerGroup.append(newPatient)
        newHealthyGroup = []
        for patient in healthyGroup:
            newPatient = {}
            for key in keysToKeep:
                newPatient[key] = patient[key]
            newHealthyGroup.append(newPatient)
        
        print("Threw out", discarded, "positions")
        
        return newCancerGroup, newHealthyGroup
    

cancerFilesLocation = "/home/fruitdragon/workspace/CancerClassifier/data/raw/cancer/Bladder"
healthyFilesLocation = "/home/fruitdragon/workspace/CancerClassifier/data/raw/healthy/Bladder"
matchName = cancerFilesLocation #genome version to be converted to.
td = TissueDataset(cancerFilesLocation, healthyFilesLocation, matchName)

cancer = []
for patient in td.cancerBetaValues:
    for value in patient:
        cancer.append(patient[value])
healthy = []
for patient in td.healthyBetaValues:
    for value in patient:
        healthy.append(patient[value])

with open("rawCancerPat1.pkl", 'wb') as saveTo:
    pickle.dump(cancer, saveTo)
with open("rawCancerPat2.pkl", 'wb') as saveTo:
    pickle.dump(healthy, saveTo)

#'''

with open("rawCancerPat1.pkl", 'rb') as loadFrom:
    cancer =pickle.load(loadFrom)

with open("rawCancerPat2.pkl", 'rb') as loadFrom:
    healthy =pickle.load(loadFrom)
#'''



def plotHistogram():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    fig = plt.figure()
    fig.suptitle('Cancer vs Healthy Histogram')
    
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Beta Value', fontsize = 15)
    ax.set_ylabel('Prob Density', fontsize = 15)
        
    
    plt.hist(cancer, 100, density=1, facecolor='red', alpha=0.8)
    plt.hist(healthy, 100, density=1, facecolor='blue', alpha=0.35)
    
    
    #plt.show()
    
    
    with PdfPages("HealthyVsCancerHistogram.pdf") as pdf:
        pdf.savefig()
    
    plt.close()


def calculateMoments():
    import numpy as np
    from scipy.stats import kurtosis
    from scipy.stats import skew
    
    npCancer = np.array(cancer)
    npHealthy = np.array(healthy)
    
    cancerMoments = []
    healthyMoments = []
    
    cancerMoments.append(mean(npCancer))
    cancerMoments.append(std(npCancer))
    cancerMoments.append(skew(npCancer))
    cancerMoments.append(kurtosis(npCancer))
    
    healthyMoments.append(mean(npHealthy))
    healthyMoments.append(std(npHealthy))
    healthyMoments.append(skew(npHealthy))
    healthyMoments.append(kurtosis(npHealthy))
    
    with open ("GroupMoments.txt", 'w') as saveTo:
        saveTo.write("Cancer Moments  = {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}\n".format(cancerMoments[0],cancerMoments[1],cancerMoments[2],cancerMoments[3]))
        saveTo.write("Healthy Moments = {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}\n".format(healthyMoments[0],healthyMoments[1],healthyMoments[2],healthyMoments[3]))
        saveTo.flush()
    #print(cancerMoments)
    #print(healthyMoments)



plotHistogram()
calculateMoments()

print("Finished")













