'''
Created on Mar 10, 2018

@team: marcin&jan
'''


import matplotlib
import os
matplotlib.use('Agg')
#from pylab import *
from pylab import exp, plt, hist, sqrt, diag, plot, legend, concatenate, normal
from scipy.optimize import curve_fit
import pickle
from matplotlib.backends.backend_pdf import PdfPages

#'''
#Marcin's version
#dataSetLocations = ["/work/projects/nn9328k/nnetwork/bladderDataSet.pickle"]

#dataSetLocations = ["/work/projects/nn9328k/nnetwork/bladderDataSet.pickle",
#                    "/work/projects/nn9328k/nnetwork/kidneyDataSet.pickle",
#                    "/work/projects/nn9328k/nnetwork/prostateDataSet.pickle"]


#dataSetLocations = ["/home/fruitdragon/workspace/CancerClassifier/data/raw/cancer/BladderCancer",
#                    "/home/fruitdragon/workspace/CancerClassifier/data/raw/cancer/BladderHealthy"]
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
        '''Get rid of any patients that have less than a set ratio of BetaValues to NAs'''
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
        '''Get rid of any positions that have any NAs'''
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
    

'''


cancerFilesLocation = "/home/fruitdragon/workspace/CancerClassifier/data/raw/cancer/Bladder"
healthyFilesLocation = cancerFilesLocation
matchName = cancerFilesLocation #genome version to be converted to.
td = TissueDataset(cancerFilesLocation, healthyFilesLocation, matchName)

cancer = []
for patient in td.cancerBetaValues:
    for value in patient:
        cancer.append(patient[value])
        #if(len(cancer)>1000):
        #    break
healthy = []
for patient in td.healthyBetaValues:
    for value in patient:
        healthy.append(patient[value])
        #if(len(healthy)>5000):
        #    break

'''



'''
filePath = "/home/fruitdragon/workspace/NeuralClassifier/old/data"
cancerAddress = "bladderCancer_full.pkl"
healtyAddress = "bladderHealthy_full.pkl"


with open(os.path.join(filePath, cancerAddress), 'rb') as cancerFile:
    dataset = pickle.load(cancerFile)
    cancer = []
    trainingSet = dataset["training"]
    for patient in trainingSet:
        for value in patient:
            cancer.append(value)
    validSet = dataset["validation"]
    for patient in trainingSet:
        for value in patient:
            cancer.append(value)
    testSet = dataset["testing"]
    for patient in trainingSet:
        for value in patient:
            cancer.append(value)
            

with open(os.path.join(filePath, healtyAddress), 'rb') as healthyFile:
    dataset = pickle.load(healthyFile)
    healthy = []
    trainingSet = dataset["training"]
    for patient in trainingSet:
        for value in patient:
            healthy.append(value)
    validSet = dataset["validation"]
    for patient in trainingSet:
        for value in patient:
            healthy.append(value)
    testSet = dataset["testing"]
    for patient in trainingSet:
        for value in patient:
            healthy.append(value)
    
'''


















#patienMax used for quick testing
patientMax = None
min_y = 0
max_y = 2.2
bin = 200

print("If this doesn't work check that it is running in python3")


def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)





#set up a new blank sheet and title it
fig = plt.figure()
fig.suptitle('Histogram of Beta Values for ')

#the first of two columns on the sheet is for the cancer plot. Set up the axes and title it 'cancer'.
ax = plt.subplot(121)
ax.grid(True)
plt.title("Cancer")
#print(params,'\n',sigma) 



#y,x,_=hist(cancer,bin,alpha=.3,label='data')
#y,x,_=hist(healthy,bin,alpha=.3,label='data')






data=concatenate((normal(1,.2,5000),normal(2,.2,2500)))
y,x,_=hist(data,100,alpha=.3,label='data')
#x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
#print("data x =", x)
#print("data y =", y)
#print(cancer)


#y,x,_=hist(cancer,bin,alpha=.3,label='data')
x=(x[1:]+x[:-1])/2 # for len(x)==len(y)


expected=(1.4,0.2,250,2,0.2,125)
params,cov=curve_fit(bimodal,x,y,expected)
sigma=sqrt(diag(cov))
plot(x,bimodal(x,*params),color='red',lw=3,label='model')
legend()



#the second of two columns on the sheet is for the healthy plot. Set up the axes and title it 'healthy'.
ax = plt.subplot(122)
ax.grid(True)
plt.title("Healthy")



data=concatenate((normal(1,.2,5000),normal(2,.2,2500)))
y,x,_=hist(data,10,alpha=.3,label='data')
#y,x,_=hist(healthy,bin,alpha=.3,label='data')
x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
#print("healthy x =", x)
#print("healthy y =", y)


params,cov=curve_fit(bimodal,x,y,expected)
sigma=sqrt(diag(cov))
plot(x,bimodal(x,*params),color='red',lw=3,label='model')
legend()


#fig.show()
#input()


with PdfPages("TestCombinedHistogram.pdf") as pdf:
    pdf.savefig()

plt.close()

print("FINISHED")
