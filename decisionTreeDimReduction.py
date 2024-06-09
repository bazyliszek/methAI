from math import log
import pickle
import os
from random import shuffle, randint, seed
import time
import sys

METHYLATION_CUTOFF = 0.3

class DTNode():
    def __init__(self, cancerDataSet, healthyDataSet, locations, remainingQuestions, forbiddenQuestions, healthyRatio, NA_VALUE = -1):
        myPrint("Starting analysis of node with", len(cancerDataSet), "cancer points and", len(healthyDataSet), "healthy points")
        self.NA_VALUE = NA_VALUE
        self.yesNode = None
        self.noNode = None
        self.divideLocation = None
        myPrint("categorizing node")
        #dataset contains cancerTainingSet and healthyTrainingSet, both of which are 2d arrays divided patient then betaLocation.
        if(len(cancerDataSet) == 0):
            myPrint("cancer group empty")
            if(len(healthyDataSet) == 0):
                #Node has no data points at all and therefore cannot be classified
                myPrint("Node is empty")
                self.entropy = 0
                self.confidence = 0
                self.counts = 0
                self.answer = "Empty"
            else:
                #node has only healthy data points
                myPrint("Node contains only healthy data")
                self.entropy = 0
                self.confidence = 1
                self.counts = len(healthyDataSet)
                self.answer = "Healthy"
        else:
            myPrint("cancer group not empty")
            if(len(healthyDataSet) == 0):
                #node has only cancer data points
                myPrint("node contains only cancer data")
                self.entropy = 0
                self.confidence = 1
                self.counts = len(cancerDataSet)
                self.answer = "Cancer"
            else:
                myPrint("node contains a mix of data")
                #node has a mix of healthy and cancer data points
                healthyCount = len(healthyDataSet)
                cancerCount = len(cancerDataSet)
                self.counts = healthyCount + cancerCount
                self.weightedCount = healthyCount*healthyRatio + cancerCount
                
                
                healthyFraction = healthyCount*healthyRatio/float(self.weightedCount)
                self.entropy = -healthyFraction*log(healthyFraction,2)- (1-healthyFraction)*log((1-healthyFraction),2)
                
                if(healthyCount*healthyRatio > cancerCount):
                    self.answer = "Healthy"
                    self.confidence = healthyCount*healthyRatio/float(self.weightedCount)
                else:
                    self.answer = "Cancer"
                    self.confidence = cancerCount/float(self.weightedCount)
                    
        
                self.yesNode = None
                self.noNode = None
                self.divideValue = 0
                if(remainingQuestions > 0 and self.entropy > 0):
                    self.__divideData(cancerDataSet, healthyDataSet, locations, remainingQuestions, forbiddenQuestions, healthyRatio)
                else:
                    myPrint("Stopping with", remainingQuestions, "remaining questions and", self.entropy, "entropy")
                    
    
    
    def __divideData(self, cancerDataSet, healthyDataSet, locations, remainingQuestions, forbiddenQuestions, healthyRatio):
        myPrint("Remaining questions =", remainingQuestions, ": Looking for best location to divide on")
        self.divideLocation = self.findBestQuestion(cancerDataSet, healthyDataSet, locations, forbiddenQuestions, healthyRatio)

        cancerYesGroup = []
        healthyYesGroup = []
        cancerNoGroup = []
        healthyNoGroup = []
        
        for currentPatient in cancerDataSet:
            #check if the current beta value is greater than the cutoff
            if(self.divideLocation not in currentPatient.keys() or currentPatient[self.divideLocation] == self.NA_VALUE):
                cancerYesGroup.append(currentPatient)
                cancerNoGroup.append(currentPatient)
            elif(currentPatient[self.divideLocation] > METHYLATION_CUTOFF):
                cancerYesGroup.append(currentPatient)
            else:
                cancerNoGroup.append(currentPatient)
                
        for currentPatient in healthyDataSet:
            #check if the current beta value is greater than the cutoff
            if(self.divideLocation not in currentPatient.keys() or currentPatient[self.divideLocation] == self.NA_VALUE):
                healthyYesGroup.append(currentPatient)
                healthyNoGroup.append(currentPatient)
            elif(currentPatient[self.divideLocation] > METHYLATION_CUTOFF):
                healthyYesGroup.append(currentPatient)
            else:
                healthyNoGroup.append(currentPatient)


        myPrint("node divided at position", self.divideLocation, ". Going into yes group now")
        self.yesNode = DTNode(cancerYesGroup, healthyYesGroup, locations, remainingQuestions - 1, forbiddenQuestions, healthyRatio)
        myPrint("node divided at position", self.divideLocation, ". Going into no group now")
        self.noNode = DTNode(cancerNoGroup, healthyNoGroup, locations, remainingQuestions - 1, forbiddenQuestions, healthyRatio)
    
    def findBestQuestion(self, cancerDataSet, healthyDataSet, locations, forbiddenQuestions, healthyRatio):
        #check how the entropy changes when the data is divided along a given position
        entropyPerQuestion = []
        
        
        '''
        print("RUN IN DEBUG MODE!!!")
        #DEBUGGING ONLY. TOGGLE COMMENT TRAP BEFORE RUNNING FOR REAL
        for i in range(0, 100):
            question = locations[i]
            '''
        for question in locations:
            #'''
            
            #any position listed in forbidden has infinite entropy
            if question in forbiddenQuestions:
                entropyPerQuestion.append(float("inf"))
                continue
            
            cancerYesGroup = []
            healthyYesGroup = []
            cancerNoGroup = []
            healthyNoGroup = []
            
            for currentPatient in cancerDataSet:
                #check if the current beta value is greater than the cutoff
                if(question not in currentPatient.keys() or currentPatient[question] == self.NA_VALUE):
                    cancerYesGroup.append(currentPatient)
                    cancerNoGroup.append(currentPatient)
                elif(currentPatient[question] > METHYLATION_CUTOFF):
                    cancerYesGroup.append(currentPatient)
                else:
                    cancerNoGroup.append(currentPatient)
                    
            for currentPatient in healthyDataSet:
                #check if the current beta value is greater than the cutoff
                if(question not in currentPatient.keys() or currentPatient[question] == self.NA_VALUE):
                    healthyYesGroup.append(currentPatient)
                    healthyNoGroup.append(currentPatient)
                elif(currentPatient[question] > METHYLATION_CUTOFF):
                    healthyYesGroup.append(currentPatient)
                else:
                    healthyNoGroup.append(currentPatient)
                    
                    
            #Calculate the entropy for each group.
            if(len(healthyYesGroup)>0):
                healthyYesFraction = len(healthyYesGroup)*healthyRatio/float(len(healthyYesGroup)*healthyRatio+len(cancerYesGroup))
                if(healthyYesFraction == 1 or healthyYesFraction == 0):
                    yesEntropy = 0
                else:
                    yesEntropy = -healthyYesFraction*log(healthyYesFraction,2)- (1-healthyYesFraction)*log((1-healthyYesFraction),2)
            else:
                yesEntropy = 0
            
            
            if(len(healthyNoGroup) > 0):
                healthyNoFraction = len(healthyNoGroup)*healthyRatio/float(len(healthyNoGroup)*healthyRatio+len(cancerNoGroup))
                if(healthyNoFraction == 1 or healthyNoFraction == 0):
                    noEntropy = 0
                else:
                    noEntropy = -healthyNoFraction*log(healthyNoFraction,2)- (1-healthyNoFraction)*log((1-healthyNoFraction),2)
            else:
                noEntropy = 0
            
            #CONSIDER WEIGHTING THE ENTROPY BY THE NUMBER OF NODES IN EACH GROUP
            entropyPerQuestion.append(yesEntropy + noEntropy)
        
        #select the question which has the lowest entropy
        bestEntropy = entropyPerQuestion[0]
        bestIndex = 0
        ties = 0
        for i in range(1, len(entropyPerQuestion)):
            if(entropyPerQuestion[i]<bestEntropy):
                bestEntropy = entropyPerQuestion[i]
                bestIndex = i
                ties = 0
            if(entropyPerQuestion[i]==bestEntropy):
                ties += 1
            
        myPrint("best entropy gain is in position", bestIndex, "which reduces entropy from", self.entropy, "to", bestEntropy)
        myPrint("There were", ties, "other positions that would have achieved the same improvement so the first found was used")
        return locations[bestIndex]
        
        
        
class DecisionTree():
    def __init__(self, dataset, depth = 4, forbiddenQuestions = []):
        
        cancerData = []
        normalData = []
        for current in dataset:
            patient = {}
            for key in current:
                if key == "Cancer":
                    continue
                else:
                    patient[key] = current[key]
            if current["Cancer"]:
                cancerData.append(patient)
            else:
                normalData.append(patient)
        
        self.healthyRatio = len(cancerData)/len(normalData)
        
        #dataSet is a 3d array divided by classification, then patient, then beta location.
        self.rootNode = DTNode(cancerData, normalData, list(cancerData[0].keys()), depth, forbiddenQuestions, self.healthyRatio, -1)
        
        myPrint("initial entropy is", self.rootNode.entropy)
        myPrint("best question for the root node is", self.rootNode.divideLocation)
    
                
                
    def getAllQuestions(self, node = None):
        if(node == None):
            node = self.rootNode

        if(node.divideLocation != None):
            locations = [node.divideLocation]
            locations.extend(self.getAllQuestions(node.yesNode))
            locations.extend(self.getAllQuestions(node.noNode))
            return locations
        else:
            return []
        
    def save(self, saveFile):
        pickle.dump(self.rootNode, open(saveFile, "wb"))
    
    def getFinalEntropy(self, node=None):
        if(node == None):
            node = self.rootNode
        totalEntropy = 0
        if(node.divideLocation != None):
            totalEntropy = self.getFinalEntropy(node.yesNode)+self.getFinalEntropy(node.noNode)
        else:
            totalEntropy = node.entropy
        return totalEntropy
    
    def printTree(self, destinationFile = None, cancerValidationSet = [], healthyValidationSet = []):
        depth = 0
        lines = self._printNode(depth, self.rootNode)
        if(len(cancerValidationSet) > 0 and len(healthyValidationSet) > 0):
            tn, tp, fn, fp = self.roc(cancerValidationSet, healthyValidationSet)
            sence = tp/float(tp+fn)
            spec = tn/float(tn+fp)
            lines += os.linesep
            lines += "Sensitivity = " + str(sence) + os.linesep + "Specificity = "+ str(spec) + os.linesep
            lines += "(based on "+str(len(healthyValidationSet))+" healthy validation patients and "+str(len(cancerValidationSet))+")" + os.linesep
        lines += ("Total entropy = " + str(self.getFinalEntropy(self.rootNode)))
        
            
        if destinationFile == None:
            print(lines)
        else:
            with open(destinationFile, 'w') as toWrite:
                toWrite.write(lines)



    def test(self, cancerTestingSet, healthyTestingSet):
        tn, tp, fn, fp = self.roc(cancerTestingSet, healthyTestingSet)
        sence = tp/float(tp+fn)
        spec = tn/float(tn+fp)
        
        print("TP =", tp,": TN =",tn,": FP =",fp,": FN =",fn)
        print("sensitivity =", sence, ": specificity =", spec)

    def validate(self, cancerValidationSet, healthyValidationSet):
        tn, tp, fn, fp = self.roc(cancerValidationSet, healthyValidationSet)
        try:
            sence = tp/float(tp+fn)
        except:
            sence = 0
            
        try:
            spec = tn/float(tn+fp)
        except:
            spec = 0
        
        return sence, spec
        
    def roc(self, cancerData, healthyData):
        tn, tp, fn, fp = 0,0,0,0
        for i in range(0, len(healthyData)):
            allClassifications = self.classify(healthyData[i])
            
            confidence = 0
            answer = None
            for key in allClassifications.keys():
                if(allClassifications[key] > confidence):
                    confidence = allClassifications[key]
                    answer = key
            
            if(answer == "Healthy"):
                #print(i, "TRUE NEGATIVE (CONFINDENCE", confidence, ")")
                tn += 1
            else:
                #print(i, "FALSE POSITIVE (CONFINDENCE", confidence, ")")
                fp+=1
        for i in range(0, len(cancerData)):
            allClassifications = self.classify(cancerData[i])
                        
            confidence = 0
            answer = None
            for key in allClassifications.keys():
                if(allClassifications[key] > confidence):
                    confidence = allClassifications[key]
                    answer = key
            
            if(answer == "Healthy"):
                #print(i, "FALSE NEGATIVE (CONFINDENCE", confidence, ")")
                fn+=1
            else:
                #print(i, "TRUE POSITIVE (CONFINDENCE", confidence, ")")
                tp += 1
        
        return tn, tp, fn, fp
        
        
        
    def classify(self, data, currentNode = None):
        if(currentNode == None):
            currentNode = self.rootNode
        if(currentNode.divideLocation != None):
            if(data[currentNode.divideLocation] == -1):
                yesClassifications = self.classify(data, currentNode.yesNode)
                noClassifications = self.classify(data, currentNode.noNode)
                finalClassifications = {}
                for key1 in yesClassifications.keys():
                    finalClassifications[key1] = yesClassifications[key1]/2
                for key2 in noClassifications.keys():
                    if(key2 in finalClassifications.keys):
                        finalClassifications[key2] += noClassifications[key2]/2
                    else:
                        finalClassifications[key2] = noClassifications[key2]/2
                return finalClassifications
            if(data[currentNode.divideLocation] > 0.3):
                return self.classify(data, currentNode.yesNode)
            else:
                return self.classify(data, currentNode.noNode)
        else:
            return {currentNode.answer : currentNode.confidence}
    
        
    def _printNode(self, depth, currentNode, nodeName = "rootNode"):
        line = ""
        for _i in range(0, depth):
            line += "    "
        
        if(currentNode.entropy == 0):
            line += nodeName + " (" + str(currentNode.counts) + ") is classified as " + str(currentNode.answer) + " and has an entropy of 0."+os.linesep
            return line
        elif(currentNode.divideLocation != None):
            line += nodeName + " (" + str(currentNode.counts) + ") is classified as "+ currentNode.answer + " but still has an entropy of {0:1g}".format(currentNode.entropy) + " and is divided at point " + str(currentNode.divideLocation)+ os.linesep
            lines =line + self._printNode(depth+1, currentNode.yesNode, "YesNode") + self._printNode(depth+1, currentNode.noNode, "NoNode")
            return lines
        else:
            line += nodeName + " (" + str(currentNode.counts) + ") is classified as " + str(currentNode.answer) + " but still has an entropy of " + str(currentNode.entropy) + " however the max depth has been reached so no further divisions are possible."+os.linesep
            return line

def myPrint(*data):
    #with open(OUTPUTFILE_NAME, 'a') as outputFile:
    #    outputFile.write(str(data))
    #    outputFile.write(os.linesep)
    print(data)
    
def clear():     
    #with open(OUTPUTFILE_NAME, 'w') as outputFile:
    #    outputFile.write("Starting")
    #    outputFile.write(os.linesep)
    pass
    
