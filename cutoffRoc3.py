import os
import pickle
from neuralPCAClassifier import Classifier
import random

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from random import randint
import sys




        





def getRandomRocPoints(patients):
    answers = []
    for i in range(0, len(patients["validation"])):
        answers.append(random.random())
    
    allCutoffs = [-1,2]
    allCutoffs.extend(answers)
    allCutoffs.sort()
    
    sensitivities = []
    negSpecificities = []
    
    for i in range(0, len(allCutoffs)):
        falseNeg = 0
        falsePos = 0
        trueNeg = 0
        truePos = 0
        for j in range(0, len(patients["validation"])):
            if(patients["validation"][j]["Cancer"]):
                if(answers[j]>allCutoffs[i]):
                    truePos+=1
                else:
                    falseNeg+=1
            else:
                if(answers[j]>allCutoffs[i]):
                    falsePos+=1
                else:
                    trueNeg+=1
            
        sensitivities.append(truePos/(falseNeg+truePos))
        negSpecificities.append(1-(trueNeg/(falsePos+trueNeg)))
    
    return (sensitivities, negSpecificities)

def getRocPoints(patients, nodes, iterations, name = None, useBoth = False, test = False):
    classifier = Classifier(patients["training"], hiddenLayerSize = nodes)
    classifier.train(maxIterations = iterations)
    
    if(test):
        answers = classifier.classify(patients["testing"])
    else:
        if(useBoth):
            combinedSet = []
            combinedSet.extend(patients["training"])
            combinedSet.extend(patients["validation"])
            answers = classifier.classify(combinedSet)
        else:
            answers = classifier.classify(patients["validation"])
    allCutoffs = [[-1],[2]]
    allCutoffs.extend(answers)
    allCutoffs.sort()
    
    sensitivities = []
    negSpecificities = []
    
    for i in range(0, len(allCutoffs)):
        falseNeg = 0
        falsePos = 0
        trueNeg = 0
        truePos = 0
        if(test):
            for j in range(0, len(patients["testing"])):
                if(patients["testing"][j]["Cancer"]):
                    if(answers[j]>allCutoffs[i]):
                        truePos+=1
                    else:
                        falseNeg+=1
                else:
                    if(answers[j]>allCutoffs[i]):
                        falsePos+=1
                    else:
                        trueNeg+=1
        else:
            for j in range(0, len(patients["validation"])):
                if(patients["validation"][j]["Cancer"]):
                    if(answers[j]>allCutoffs[i]):
                        truePos+=1
                    else:
                        falseNeg+=1
                else:
                    if(answers[j]>allCutoffs[i]):
                        falsePos+=1
                    else:
                        trueNeg+=1
            
        sensitivities.append(truePos/(falseNeg+truePos))
        negSpecificities.append(1-(trueNeg/(falsePos+trueNeg)))
    
    if(name != None):
        classifier.save(name+"_nn_weights.pkl")
    return (sensitivities, negSpecificities)



def areaUnderCurve(sens, negSpec):
    AUC = 0
    for i in range(1, len(sens)):
        AUC += abs(negSpec[i] - negSpec[i-1])*(sens[i-1] + sens[i])/2    

    return AUC









#def plot(sens1, negSpec1, sens2, negSpec2, sens3, negSpec3, sens4, negSpec4, AUC1, AUC2, AUC3, AUC4):
def plot(lines, saveTo, title):
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('1-Specificity', fontsize = 15)
    ax.set_ylabel('Sensitivity', fontsize = 15)
    
    if(includeTraining):
        if(seed == None):
            ax.set_title('Approx RocCurve ' + title, fontsize = 20)
        else:
            ax.set_title('Approx RocCurve ' + title + ' (Seed={0})'.format(seed), fontsize = 20)
    else:
        if(seed == None):
            ax.set_title('RocCurve ' + title, fontsize = 20)
        else:
            ax.set_title('RocCurve ' + title, fontsize = 20)
            #ax.set_title('RocCurve ' + title + ' (Seed={0})'.format(seed), fontsize = 20)
        
    for i in range(0, len(lines)):
        ax.scatter(lines[i].negSpec, lines[i].sens, c=colorList[i], s=5)
        ax.plot(lines[i].negSpec, lines[i].sens, c=colorList[i])
    
    ax.grid()
    
    legends = []
    for i in range(0, len(lines)):
        legends.append(lines[i].name + " (AUC = {0:.3f})".format(lines[i].AUC))
    
    ax.legend(legends, bbox_to_anchor=(0.85,0.35), bbox_transform=plt.gcf().transFigure)
    
    #ax.legend([str([7,4])+" (AUC = {0:.3f}".format(AUC1) +")",str([10,10])+" (AUC = {0:.3f}".format(AUC2) +")", str([5,4,3])+" (AUC = {0:.3f}".format(AUC3) +")", "RANDOM (AUC = {0:.3f}".format(AUC4) +")"],
    #           bbox_to_anchor=(0.7,0.3),
    #           bbox_transform=plt.gcf().transFigure)
    
    if(saveTo == None):
        plt.show()
    else:
        with PdfPages(saveTo) as pdf:
            pdf.savefig()



class RocLine():
    def __init__(self, name, sens, negSpec, AUC):
        self.name = name
        self.sens = sens
        self.negSpec = negSpec
        self.AUC = AUC



#TEST = True
TEST = False
if(TEST):
    colorList = ['#D55E00','#CC79A7','#0072B2','#F0E442','#009E73']
else:
    colorList = ['#009E73','#D55E00','#CC79A7','#0072B2','#F0E442']

seed = None
seed = 1
if(seed != None):
    random.seed(seed)
includeTraining = False
#includeTraining = True
netIterations = 20
cutoff= 0.3
#netIterations = 1


#neuralNets = [[7,4], [10,10], [5,4,3]]
#neuralNets = [[7,4]]
neuralNets = [[7,4]]
#neuralNets = [[2,2], [1,2], [2,1], None]

#pickleFile = "/home/fruitdragon/workspace/NeuralClassifier/old/data/prostate_top7.pkl"
#pickleFile = "/home/fruitdragon/workspace/NeuralClassifier/old/data/141119/kidney_PCA2.pkl"
#pickleFile = "/home/fruitdragon/workspace/NeuralClassifier/old/data/prostate_Random10.pkl"

#for cancerType in ["Bladder","Kidney","Prostate"]:
for cancerType in ["bladder","kidney","prostate"]:
    lines = []
    for boost in range(0, 5):
        pickleFile = "/home/fruitdragon/workspace/NeuralClassifier/old/data/boosted/"+cancerType+"_TOP10_boost_"+str(boost)+".pkl"
        if(TEST):
            saveFile = "/home/fruitdragon/"+cancerType+"_Test_ROC.pdf"
        else:
            saveFile = "/home/fruitdragon/"+cancerType+"_Valid_ROC.pdf"
            
        saveFile = None
            
        #title ="Random"
        title = None
        
        with open(pickleFile, 'rb') as cancerFile:
            patients = pickle.load(cancerFile)
        
        #print("Starting with " + str(len(patients["training"][0].keys())))
        #print(patients["training"][0].keys())
    
        #Toss out any with the same median between both cancer and normal groups
        badKeys = set()
        for key in patients["training"][0].keys():
            cancerCountHigh = 0
            cancerCountLow = 0
            normalCountHigh = 0
            normalCountLow = 0
            
            for setName in["training", "validation"]:
                for patient in patients[setName]:
                    if(patient["Cancer"]):
                        if(patient[key] > 0.3):
                            cancerCountHigh += 1
                        else:
                            cancerCountLow += 1
                    else:
                        if(patient[key] > 0.3):
                            normalCountHigh += 1
                        else:
                            normalCountLow += 1
            if((cancerCountHigh<cancerCountLow) == (normalCountHigh<normalCountLow)):
                badKeys.add(key)
        
        
        
        #count = 0
        #for key in patients["training"][0].keys():
        #    if(count > 3):
        #        if(key != "Cancer"):
        #            badKeys.add(key)
        #    count += 1
            
                
        print("in " +cancerType +" throwing out " + str(badKeys))
            
        for key in badKeys:
            for setName in["training", "validation", "testing"]:
                for patient in patients[setName]:
                    del patient[key]
            print("Thrown out " + key)
            print(patients["training"][0].keys())
                        
        
        if(title == None):
            name = pickleFile.rsplit(sep="/", maxsplit=1)[1]
            name = name.split(sep="_", maxsplit=1)
            title = name[0] + " " + str(len(patients["training"][0].keys())-1) + "CpG Sites "
            if(TEST):
                title += "(Test Set)"
            else:
                title += "(Validation)"
    
        
        if(cutoff != None):
            for setName in ["training", "validation", "testing"]:
                for patient in patients[setName]:
                    for key in patient.keys():
                        if(patient[key]>cutoff):
                            patient[key] = 1
                        else:
                            patient[key] = 0
        
        #RANDOMIZE RESULTS !!! REMOVE BEFORE RUNNING
        #print("WARNING RANDOMIZING RESULTS")
        #for key in patients.keys():
        #    for current in patients[key]:
        #        current["Cancer"] = randint(1,2) == 1
        
        
        #lines = []
        for neuralNet in neuralNets:
            if(neuralNet == None):
                sens, negSpec = getRandomRocPoints(patients)
                name = "Random"
            else:
                weightsPickleName = cancerType+"_nn" +str(neuralNet) + "_weights.pkl"
                sens, negSpec = getRocPoints(patients, neuralNet, netIterations, weightsPickleName, includeTraining, test=TEST)
                name = "NN(" +str(neuralNet) + ") BOOST " + str(boost) 
                    
            AUC = areaUnderCurve(sens, negSpec)
            lines.append(RocLine(name, sens, negSpec, AUC))
        
        
    plot(lines, saveFile, title)
                
        
print("Finished")
