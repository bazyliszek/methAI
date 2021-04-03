'''
Created on 14 Nov 2016

@author: fruitdragon
'''

import nn
import pickle
from random import shuffle

'''neural net currently throws out all but 10 of the datapoints as the extra noise from 400000 of them was preventing
it from training in a reasonable time.

Solution - strip out all data that does not have a high variance?
'''


class Classifier():
    def __init__(self, inputPatients, hiddenLayerSize = [20, 20], loadNetFrom = None):
        '''TrainingFiles should be a list of filenames. Each file contains the data for a different caner type.
        Each line in each file contains the beta values for a given datapoint.'''
        
        inputCancerPatients = []
        inputNormalPatients = []
        for i in range(0, len(inputPatients)):
            if(inputPatients[i]["Cancer"]):
                inputCancerPatients.append(inputPatients[i])
            else:
                inputNormalPatients.append(inputPatients[i])
                
        maxLen = max(len(inputCancerPatients), len(inputNormalPatients))
        
        cancerPatients = []
        normalPatients = []
        for i in range(0, maxLen):
            cancerPatients.append(inputCancerPatients[i%len(inputCancerPatients)])
        for i in range(0, maxLen):
            normalPatients.append(inputNormalPatients[i%len(inputNormalPatients)])
        
        
        dataAndAnswer =[]
        self.keys = list(inputPatients[0].keys())
        self.keys.remove("Cancer")
        for i in cancerPatients:
            newPatient = []
            for key in self.keys:
                newPatient.append(i[key])
            dataAndAnswer.append((newPatient, [1]))
        for i in normalPatients:
            newPatient = []
            for key in self.keys:
                newPatient.append(i[key])
            dataAndAnswer.append((newPatient, [0]))
        shuffle(dataAndAnswer)
        
        self.dataSets = []
        self.answers = []
        for i in dataAndAnswer:
            self.dataSets.append(i[0])
            self.answers.append(i[1])
        
        
            
                
        #Load the net if a net is given, else build a new one.
        if (loadNetFrom == None):
            #build a new neural net based on the training file
            layersList = []
            layersList.append(len(self.dataSets[0]))
            for i in hiddenLayerSize:
                layersList.append(i)
            layersList.append(1)
            self.net = nn.NN(layersList)
        else:
            self.net = pickle.load(open(loadNetFrom, "rb")) 


    def train(self, acceptableSSE = 0.01, maxIterations = 50000):
        '''Calls the training function on the neural net. Training will continue until the average sum squared error is met, or until the a given number of iterations has been exceeded.'''
        print("training with", self.answers)
        sse = self.net.train(self.dataSets, self.answers, acceptableSSE,maxIterations)
        print("achieved an sse of", round(sse, 3))
        
    def save(self, saveFile):
        '''saves the current weights of the network to a file'''
        pickle.dump(self.net, open(saveFile, "wb"))

    def classify(self, patients):
        '''reads data from an input file, runs it through the neural net and writes the results to an output file.'''
        data =[]
        for i in patients:
            newPatient = []
            for key in self.keys:
                newPatient.append(i[key])
            data.append(newPatient)
            
        results = self.net.multiProcess(data)

        return results
                
                
    def classifyFile(self, inputFile, outputFile):
        '''reads data from an input file, runs it through the neural net and writes the results to an output file.'''
        data = self.readFile(inputFile)
        
        results = []
        for i in range(0, len(data)):
            print("processing", data[i])
            results.append(self.net.process(data[i]))

        with open(outputFile, 'w') as myFile:
            for i in range(0, len(results)):
                for j in range(0, len(results[i])):
                    
                    myFile.write(str(round(results[i][j],3)) + " ")
                myFile.write("\n")
                
            


#if __name__ == "__main__"): is used to ensure the enclosed code is executed only if the program is run alone. This means that running python nnClassifier.py will cause the below to happen, but importing nnClassifier inside another python program will not

#''' Jan's version
if( __name__ == "__main__"):
    #with open("/home/fruitdragon/workspace/NeuralClassifier/old/data/bladderCancer_full.pkl", 'rb') as cancerFile:
    #    cancer = pickle.load(cancerFile)
    #with open("/home/fruitdragon/workspace/NeuralClassifier/old/data/bladderHealthy_full.pkl", 'rb') as healthyFile:
    #    healthy= pickle.load(healthyFile)
    
    with open("/home/fruitdragon/workspace/NeuralClassifier/old/data/prostate_top7.pkl", 'rb') as cancerFile:
        data = pickle.load(cancerFile)
    classifier = Classifier(data["training"])
    
    classifier.train(maxIterations = 100)
    #classifier.classify("data1Test.txt", "data1Results.txt")
    #classifier.save("/home/fruitdragon/workspace/NeuralClassifier/old/nnObject.pyo")
    
    
    for i in range(0, len(classifier.dataSets)):
        answer = classifier.net.process(classifier.dataSets[i])
        roundedInput = []
        for j in range(0, len(classifier.dataSets[i])):
            roundedInput.append(round(classifier.dataSets[i][j],2))
        #print("INPUT OF", roundedInput, "GAVE", round(answer[0],3), "INSTEAD OF", classifier.answers[i][0])
        print(round(classifier.answers[i][0] - answer[0],3))
    
    print("FINISHED")
    



