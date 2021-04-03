'''
Created on 14 Nov 2016

@team: marcin&jan
'''



import random
import math
import time
from math import sqrt

class Neuron():
    def __init__(self, inputs):
        '''The neuron has a list of weights, each connecting to a neuron in the previous layer. When the neuron is activated, it multiplied the output of all neurons in the previous layer by the connecting weight'''
        #weights are initially set to random
        self.weights = []
        for i in range(0, inputs):
            self.weights.append(random.uniform(-1,1))

        #the bias behaves as a special weight connected to an input that is always 1
        self.bias = random.uniform(-1,1)

    
    #Sigmoid activation function
    def activate(self, inputs):
        '''Sums all the inputs*weights + bias then returns a sigmoid of that value. Sigmoids are often used as the activation function for neurons as the backpropagation algorithm needs to know the derivative and the derivative of a sigmoid is simply its output times (1- its output), which is fast to calculate.'''
        #the charge is the total input coming into the node
        if(len(inputs) != len(self.weights)):
            print("ERROR")
        charge = self.bias
        for i in range (0, len(inputs)):
            charge +=inputs[i] * self.weights[i]

        #trying to calculate sigmoids with extremely high/low charges is time consuming and the result will be very close to 1/0 anyway
        if(charge <-100):
            return 0
        elif (charge > 100):
            return 1
        #this is the sigmoid function
        return (1/(1+math.exp(-charge)))
    
    def train(self, inputs, error):
        '''Change the bias and all weights by their input times the error'''
        self.bias += error
        for i in range(0, len(inputs)):
            self.weights[i] += inputs[i] * error
            
class NN():
    def __init__(self, nodes):
        '''Nodes should be a list of numbers specifying how many neurons should be in each layer. e.g. if there are 10 inputs, 6 nodes in the hidden layer and 4 outputs then nodes should be [10,6,4]'''

        #create the layers
        self.layers = []
        for i in range(1, len(nodes)): #for each layer
            row = []
            for j in range(0, nodes[i]): #for each node in that layer
                row.append(Neuron(nodes[i-1])) #create a new neuron with inputs equal to the size of the previous layer
            self.layers.append(row)


        self.layerSize = len(self.layers)
        self.outputLayer = self.layers[self.layerSize - 1]

    def multiProcess(self, allInputs):
        '''Runs process on a list of inputs and returns a list of the outputs for each.'''
        outputs = []
        for current in allInputs:
            outputs.append(self.process(current))
        return outputs

    def process(self, inputs):
        '''Feeds the inputs into the first layer, then feeds the outputs of each layer into the next. After running out of layers returns the outputs of the last layer.'''
        for currentLayer in self.layers:
            results = []
            for node in range(0, len(currentLayer)):
                currentNodeOutput = currentLayer[node].activate(inputs)
                results.append(currentNodeOutput)
            inputs = results
        return results

    def train(self, inputs, answers, acceptSSE, maxIterations):
        '''Repeatedly calls trainOnce until either the desired sum squared error has been met or the maxIterations has been exceeded.'''
        print("Starting Sse is ", self.calcSse(inputs, answers))
        beginTime = time.time()
        for i in range(0, maxIterations):
            print("training iteration", i)

            #the learningRate determines how quickly the network learns new information, but also how quickly it forgets old stuff. Too high and the network overrides what it learned in the previous pattern as soon as it trains on the next one. Too low and it takes a very long time to get anywhere. The best solution is to have a learning rate that gradually decreases over time. After a bit of experimenting I found that the equation below seems to work quite well most of the time.
            #learningRate = 5/math.log(i + 100)
            
            
            learningRate = 10/math.log(i + 10)
            for i in range(0, len(inputs)):
                self.trainSinglePattern(inputs[i], answers[i], learningRate)
            
            sse = self.calcSse(inputs, answers)
            
            print("current sse =", sse)
            if(sse < acceptSSE):
                print("NETWORK TRAINED")
                print("took", round(time.time()-beginTime, 3), "seconds and", i, "iterations to achieve a sum squared error of", round(sse, 3))
                return sse
        
        print("TRAINING STOPPED")
        print("After", round(time.time()-beginTime, 3), "seconds and", i, "iterations the network stopped training. The final sum squared error is still", round(sse, 3))
        return sse
    
    def calcSse(self, inputs, answers):
        sse = 0
        for i in range(0, len(inputs)):
            workingLayerIO = self.process(inputs[i])
            for node in range (0, len(self.outputLayer)):
                difference = answers[i][node] - workingLayerIO[node]
                sse += difference * difference
        sse /= len(inputs)
        return sse

    def trainSinglePattern(self, inputs, answers, lRate):
        '''Feed the inputs into the neural net. Compare the outputs with the desired results. This gives the error for the last layer. The error for every previous layer can be calculated using the backpropagation algorithm. finally update the weights of every neuron according to the neuron's error'''
        #get the inputs/outputs for every layer
        layersIO = self.forwardPropagation(inputs)

        
        workingLayerIO = layersIO[self.layerSize]
        differences = []
        for node in range (0, len(self.outputLayer)):
            differences.append(answers[node] - workingLayerIO[node])

        #the delta for each node in the final layer is the error * the derivative of its output
        deltas = []
        for i in range(0, self.layerSize):
            deltas.append([])
   
        for node in range(0, len(differences)):
            nodesOutput = layersIO[self.layerSize][node]
            derivOfOutput = nodesOutput *(1-nodesOutput)
            deltas[self.layerSize-1].append(differences[node] * derivOfOutput)

        #use the backPropagation algorithm to calculate the delta values for the rest of the layers, working backwards
        for layer in range(self.layerSize -2, -1, -1):
            deltas[layer] = self.backPropagation(layer, layersIO, deltas[layer+1])

        #Update the weights
        for layer in range(0, self.layerSize):
            for node in range(0, len(self.layers[layer])):
                delta = deltas[layer][node]
                #print("updating node", layer,node, "by", lRate * delta)
                self.layers[layer][node].train(layersIO[layer], lRate * delta)

            
    def backPropagation(self, layer, layersIO, aboveDeltas):
        '''This method uses the back propagation algorithm to calculate the delta values for the current layer. I don't really understand the math behind what it is doing.'''
        #the deltas for the neurons in each other layer are:
        #the sum of (each delta in node ahead times the connecting weight) times the derivitive of this node's output
        
        currentDeltas = []

        #for every node in this layer
        for node in range(0, len(self.layers[layer])):
            #calculate the current delta
            nodesOutput = layersIO[layer+1][node]
            derivOfOutput = nodesOutput * (1-nodesOutput)
            
            deltaSum = 0
            for nextNode in range(0, len(aboveDeltas)):
                connectingWeight = self.layers[layer+1][nextNode].weights[node] 
                deltaSum += aboveDeltas[nextNode]*connectingWeight
            currentDeltas.append(deltaSum*derivOfOutput)
        return currentDeltas
        
        
    def forwardPropagation(self, inputs):
        '''Similar to process but returns a list of the inputs/outputs of each layer instead of just the outputs of the last one.

The inputs for a given layer can be obtained using layersIO[desiredLayer]. The outputs can be obtained using layersIO[desiredLayer+1]'''
        layersIO = []
        layersIO.append(inputs)
        for currentLayer in self.layers:
            results = []
            for node in range(0, len(currentLayer)):
                currentNodeOutput = currentLayer[node].activate(inputs)
                results.append(currentNodeOutput)
            layersIO.append(results)
            inputs = results
        return layersIO
            
    def printNodes(self):
        '''Useful for debugging but not actually needed'''
        for i in range(0, len(self.layers)):
            print("layer", i)
            for j in range(0, len(self.layers[i])):
                print("    node", j)
                for k in range(0, len(self.layers[i][j].weights)):
                    print("        weight", k, "=", self.layers[i][j].weights[k])
                print("        bias =", self.layers[i][j].bias)
                
