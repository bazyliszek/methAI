'''
Created on 8 Jun 2019

@author: fruitdragon
'''
import os
import pickle
from random import seed, shuffle
import numpy as np
from sklearn.decomposition import PCA
from decisionTreeDimReduction import DecisionTree

#NEEDS TO BE ABLE TO CONVERT OTHER POINTS INTO THE SAME PCA REFERENCE. THIS REQUIRES IT TO THROW OUT THE SAME NAS
#AND TO STORE THE PCA WEIGHTS.
#NEEDS TO STORE WHICH COLUMNS ARE DELETED DUE TO NAS AND REPLACE ANY OTHER NAS WITH AVERAGE VALUES

#NEEDS TO LET THE USER DECREASE THE DIMENSIONALITY BY EITHER PCA OR DECISION TREE. BOTH SHOULD CREATE 
#RESULTS AS SIMILAR AS POSSIBLE AND BOTH SHOULD BE REPLICATABLE.

#MAYBE INCLUDE A LINEAR MODEL AS A THIRD WAY OF REDUCING DIMENTIONALITY?



import argparse
import sys
parser = argparse.ArgumentParser(description='combines files into a pickle object, processes the data, and divides into training sets')
parser.add_argument('-i', dest='folder', default="/rds/projects/2018/colboujk-wojewodzic/nnetwork/newdata/bladder/", type=str,
                    help='Folder containing the files to be processed. (Marcin can enter b, p, or k for defaults)')
parser.add_argument('-s', dest='sets', default="0.7:0.2", type=str,
                    help='Controls how the training, validation, and testing sets are divided. Expects text in the form a:b:c or a:b where a is the training set, b is validation and c is testing. If testing is omitted then it is assumed to be 1-(a+b)')
parser.add_argument('-r', dest='seed', default=None, type=int,
                    help='Set the random seed')
parser.add_argument('-p', dest='param_copy', action="store_true",
                    help='If provided then the program will load the parameters from the file provided by -n. Overrides -d, -t, -a, and -b')
parser.add_argument('-t', dest='useTree', action="store_true",
                    help='If provided then the program will decrease the dimensions using the decision tree instead of PCA.')
parser.add_argument('-d', dest='dimensions', default=2, type=int,
                    help='Controls how many dimensions to keep after PCA or decisionTree.')
parser.add_argument('-a', dest='minPatientNA', default=0.15, type=float,
                    help='Sets the maximum proportion of NA values a patient can have before being discarded. Overridden if -p is set')
parser.add_argument('-b', dest='minPositionNA', default=0.15, type=float,
                    help='Sets the maximum proportion of NA values a cpg site can have before being discarded. Overridden if -p is set')
parser.add_argument('-o', dest='patientsFile', default=None, type=str,
                    help='Specify the location to output the patient file to. If none will create it in the same folder as the input with a generated name.')
parser.add_argument('-n', dest='paramsFile', default=None, type=str,
                    help='Specify the location to output the parameter file to. If none will create it in the same folder as the input with a generated name.')
args = parser.parse_args()

#-i b -d 2 -r 0
#-i k -d 2 -r 0
#-i p -d 2 -r 0



#-i b -t -d 10 -r 0
#-i k -t -d 10 -r 0
#-i p -t -d 10 -r 0

#-i b -p "/rds/projects/2018/colboujk-wojewodzic/nnetwork/newdata/kidney_ProcessingParams.pkl" -o "BladderKidneyMixed.pkl"
#-i b -p "/rds/projects/2018/colboujk-wojewodzic/nnetwork/newdata/prostate_ProcessingParams.pkl" -o "BladderProstateMixed.pkl"
#-i k -p "/rds/projects/2018/colboujk-wojewodzic/nnetwork/newdata/bladder_ProcessingParams.pkl" -o "KidneyBladderMixed.pkl"
#-i k -p "/rds/projects/2018/colboujk-wojewodzic/nnetwork/newdata/prostate_ProcessingParams.pkl" -o "KidneyProstateMixed.pkl"
#-i p -p "/rds/projects/2018/colboujk-wojewodzic/nnetwork/newdata/bladder_ProcessingParams.pkl" -o "ProstateBladderMixed.pkl"
#-i p -p "/rds/projects/2018/colboujk-wojewodzic/nnetwork/newdata/kidney_ProcessingParams.pkl" -o "ProstateKidneyMixed.pkl"

inputFolder = args.folder
if(inputFolder == "b"):
    inputFolder = "/rds/projects/2018/colboujk-wojewodzic/nnetwork/newdata/bladder/"
elif(inputFolder == "p"):
    inputFolder = "/rds/projects/2018/colboujk-wojewodzic/nnetwork/newdata/prostate/"
elif(inputFolder == "k"):
    inputFolder = "/rds/projects/2018/colboujk-wojewodzic/nnetwork/newdata/kidney/"
if(inputFolder[-1] != "/"):
    inputFolder += "/"
if(not os.path.exists(inputFolder)):
    print("Error! No folder exists at ", inputFolder)
    print("Exiting")
    sys.exit(1)
folderHead, folderTail = os.path.split(inputFolder[:-1])

sets = args.sets.split(":")
trainingRatio = float(sets[0])
validationRatio = float(sets[1])
if(len(sets)>2):
    ratio_sum = trainingRatio + validationRatio + float(sets[2])
    trainingRatio = trainingRatio/ratio_sum
    validationRatio = validationRatio/ratio_sum
testingRatio = 1 - trainingRatio - validationRatio

seed(args.seed)

recreatePCA = not args.param_copy
if(recreatePCA):
    pcaDimensions = args.dimensions
    minPatientNA = args.minPatientNA
    minPositionNA = args.minPositionNA

patientsFile = args.patientsFile
if(patientsFile == None):
    if(recreatePCA):
        if(pcaDimensions > 0):
            if args.useTree:
                patientsFile = os.path.join(folderHead, folderTail+"_TOP"+str(pcaDimensions)+".pkl")
            else:
                patientsFile = os.path.join(folderHead, folderTail+"_PCA"+str(pcaDimensions)+".pkl")
        else:
            patientsFile = os.path.join(folderHead, folderTail+"_NoPCA.pkl")
    else:
        patientsFile = os.path.join(folderHead, folderTail+"_PCAcopied.pkl")
        
paramsFile = args.paramsFile
if(paramsFile == None):
    paramsFile = os.path.join(folderHead, folderTail+"_ProcessingParams.pkl")
if(recreatePCA and not os.path.exists(paramsFile)):
    print("ERROR! Could not find a param file at ", paramsFile)
    print("As -p was supplied a param file is required. Exiting")
    sys.exit(1)



class TissueDataset():
    
    NAValue = -1
    def __init__(self, patients, trainingRatio, validationRatio, seedValue, pca_params):
        self.trainingSet, self.validationSet, self.testingSet = self.divideIntoSets(patients, trainingRatio, validationRatio, seedValue)
        self.pca_params = pca_params
        print("trainingSet has", len(self.trainingSet), "patients with", len(self.trainingSet[0]), "locations")
    
    def divideIntoSets(self, group, trainingRatio, validationRatio, seedValue):
        print("Dividing the data into sets")
        seed(seedValue)    
        shuffle(group)
        
        trainingLength = int(round(len(group) * trainingRatio))
        validateLength = int(round(len(group) * validationRatio))
        
        trainSet = group[:trainingLength]
        validSet = group[trainingLength:][:validateLength]
        testSet = group[trainingLength + validateLength:]
        return trainSet, validSet, testSet
    
    
    def save(self, pickleLocation, processingParamsLocation):
        pickleDic = {"training" : self.trainingSet,
                     "validation" : self.validationSet,
                     "testing" : self.testingSet}
        
        with open(pickleLocation, 'wb') as saveTo:
            pickle.dump(pickleDic, saveTo)
        with open(processingParamsLocation, 'wb') as saveTo:
            pickle.dump(self.pca_params, saveTo)
    
    
def throwOutNAPatients(group, minPercent):
    '''Get rid of any patients that have less than a set ratio of BetaValues to NAs'''
    print("Throwing out NA Patients")
    newGroup = []
    for patient in range(0, len(group)):
        if(not "Cancer" in group[patient].keys()):
            print("WEIRD Patient", patient,"is missing cancer key at start")
        #print("working on patient", patient)
        NAs = 0
        betas = 0
        for key in group[patient].keys():
            #print("working on patient", patient, "key", key)
            if(group[patient][key] == TissueDataset.NAValue):
                NAs += 1
            else:
                betas += 1
        #print("found", betas,"betas and", NAs, "NAs in patient", patient)
        try:
            if(betas / float(betas+NAs) >= minPercent):
                
                if(not "Cancer" in group[patient].keys()):
                    print("ALSO WEIRD Patient", patient,"is missing cancer key at end")
                newGroup.append(group[patient])
        except:
            pass
    
    for patientNum in range(0, len(newGroup)):
        for key in newGroup[patientNum - 1].keys():
            if(not key in newGroup[patientNum].keys()):
                print("4: ERROR patient", patientNum - 1, "has key", key, "but patient", patientNum, "does not!!!")
    
    
    return newGroup


def throwOutNAPositions(patients, minPercent):
    '''Get rid of any positions that have too many NAs'''
    print("Throwing out NA Positions")        
    keysToKeep = []
    averages = {}
    discarded = 0
    
    
    #First find which positions have the required amount of values. Store these in keys to keep. Also throw out anything with an * in the key
    for key in patients[0].keys():
        NAs = 0
        betas = 0
        average = 0
        if(not '*' in key):
            for patient in patients:
                if(patient[key] == TissueDataset.NAValue):
                    NAs +=1
                else:
                    betas += 1
                    average += patient[key]
            #check if it has a high enough ratio to be kept
            if(betas / float(betas+NAs) >= minPercent):
                keysToKeep.append(key)
                average /= betas
                averages[key] = average
            else:
                discarded += 1
                
    #then remake each group only keeping the selected keys and averaging any NA values that remain
    newPatients = []
    for patient in patients:
        newPatient = {}
        for key in keysToKeep:
            if(patient[key] == TissueDataset.NAValue):
                newPatient[key] = averages[key]
            else:
                newPatient[key] = patient[key]
        newPatients.append(newPatient)
    
    print("Threw out", discarded, "positions")
    
    return newPatients, keysToKeep, averages


def treeReduce(patients, dimensions):
    '''uses the decision tree to select the best positions to be kept.'''
    
    locations = []
    bestPositions = []
    while(len(bestPositions) < dimensions):
        classifier = DecisionTree(patients, depth = 2, forbiddenQuestions=bestPositions)

        allResults = classifier.getAllQuestions()
        for i in range(len(bestPositions), dimensions):
            bestPositions.append(allResults.pop(0))

    print("AFTER TRAINING THE MOST INTERESTING LOCATIONS WERE")
    for i in range(0, dimensions):
        locations.append(bestPositions[i])
        print(bestPositions[i])
    
    locations = bestPositions
    
    
    
    newPatients = []
    for patient in patients:
        newPatient = {}
        for key in locations:
            newPatient[key] = patient[key]
        newPatient["Cancer"] = patient["Cancer"]
        newPatients.append(newPatient)
    return newPatients, locations
    

def pca(patients, locations, dimensions = None, pca = None):
    '''performs pca on the data to reduce it down to a specified number of dimensions.'''
    
    if(dimensions == None):
        if(pca == None):
            print("either dimensions or pca_params must be given")
            return None
        else:
            dimensions = pca.get_params(False)["n_components"]
    
    
    #First converts the data into a numpy array
    listOfLists = []
    for current in patients:
        currentList = []
        for key in locations:
            if(key != "Cancer"):
                currentList.append(current[key])
        listOfLists.append(currentList)
    np_patients = np.array(listOfLists)
    
    names = []
    for i in range(0, dimensions):
        names.append("PCA" + str(i))
    
    #then passes that to the PCA module
    if(pca != None):
        data = pca.transform(np_patients)
    else:
        pca = PCA(n_components=dimensions)
        data=pca.fit_transform(np_patients)
    
    newPatients = []
    for patientNum in range(0, len(data)):
        newPatient = {}
        for i in range(0, len(names)):
            newPatient[names[i]]=data[patientNum][i]
        newPatient["Cancer"] = patients[patientNum]["Cancer"]
        newPatients.append(newPatient)
            
    return newPatients, pca

    
    
    
    
    
def createPCAReference(patients, minPatientNA, minPositionNA, dimensions):
    
    
    patients = throwOutNAPatients(patients, minPatientNA)
    print("After removing NA Patients still have",len(patients), "patients with", len(patients[0]), "locations")
    
    
    patients, locations, locationAverages = throwOutNAPositions(patients, minPositionNA)
    print("After removing NA Positions still have", len(patients[0]), "locations")
    
    if(dimensions > 0):
        if(args.useTree):
            patients, locations = treeReduce(patients, dimensions)
            return patients, (locations, locationAverages, None, "Tree")
        else:
            patients, pca_params = pca(patients, locations, dimensions)
            return patients, (locations, locationAverages, pca_params, "PCA")
    else:
        pca_params = None
        return patients, (locations, locationAverages, None, "None")
    
        

def convertToPCAReference(patients, pcaFormat):
    locations, locationAverages, pca_params, processType = pcaFormat
    newPatients = []
    for patient in patients:
        newPatient = {}
        missingValues = 0
        for location in locations:
            try:
                newPatient[location] = patient[location]
                if(newPatient[location] == TissueDataset.NAValue):
                    newPatient[location] = locationAverages[location]
                    missingValues += 1
                    
            except:
                newPatient[location] = locationAverages[location]
                missingValues += 1
        print("replaced ", missingValues,"/", len(locations), "positions")
        newPatients.append(newPatient)
    
    if(processType == "Tree"):
        for i in range(0, len(patients)):
            newPatients[i]["Cancer"] = patient[i]["Cancer"]
    else:
        if(pca_params != None):
            newPatients, params = pca(newPatients, locations, dimensions = None, pca = pca_params)
        
    
    return newPatients


def loadBetaValues(folder):
    subdirs = os.listdir(folder)
    
    patients = []
    #maxFiles = 6
    i = 0
    for currentDirectory in subdirs:
        i += 1
        #if(i >= maxFiles):
        #    patients[0]["Cancer"] = True
        #    patients[1]["Cancer"] = True
        #    patients[2]["Cancer"] = True
        #    patients[3]["Cancer"] = False
        #    patients[4]["Cancer"] = False
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
                            location = separated[2].strip() + ':' + separated[3].strip()
                            try:
                                betaDict[location] = float(separated[1].strip())
                            except ValueError:
                                #any value that can not be interpreted as a float is set to NA
                                betaDict[location] = TissueDataset.NAValue
                        except KeyError as e:
                            print("exception line", j, "of file in dir", i)
                            print(str(e))
                            print("checking for", separated[0].strip())
                            pass
                        j+=1
                            #skip any lines that can't be matched (presumably only the header
            
            #jhu-usc.edu_PRAD.HumanMethylation450.18.lvl-3.TCGA-ZG-A9LZ-01A-11D-A41L-05.gdc_hg38.txt
            #                                                           ^^ tissue type identifier
            if(len(betaDict) > 0):#skip the folders that don't use 450k probes
                try:
                    if(currentFileName.split("-")[5][:2] == "01"):
                        betaDict["Cancer"] = True
                        patients.append(betaDict)
                    elif(currentFileName.split("-")[5][:2] == "11"):
                        betaDict["Cancer"] = False
                        patients.append(betaDict)
                
                    #print("adding new patient. Current length =", len(patients))
                except IndexError:
                    pass
    
    
    #Catch weird cancer key error
    for patientNum in range(0, len(patients)):
        for key in patients[patientNum - 1].keys():
            if(not key in patients[patientNum].keys()):
                print("1: ERROR patient", patientNum - 1, "has key", key, "but patient", patientNum, "does not!!!")
    
    
    return patients


if( __name__ == "__main__"):    
        
    unprocessedPatientData = loadBetaValues(inputFolder)
    if(recreatePCA):
        patients, pcaFormat = createPCAReference(unprocessedPatientData, minPatientNA, minPositionNA, pcaDimensions)
    else:
        with open(paramsFile, 'rb') as loadFrom:
            pcaFormat = pickle.load(loadFrom)
        patients = convertToPCAReference(unprocessedPatientData, pcaFormat)
    
    td = TissueDataset(patients, trainingRatio, validationRatio, args.seed, pcaFormat)
    td.save(patientsFile, paramsFile)
    
    print("FINISHED")
    