# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 19:11:21 2021

@author: User
"""

import random
from datetime import datetime
import math
import pandas as pd
import numpy as np

class NeuralNetwork:
    numberOfHiddenLayers= 0
    numberOfHiddenNeurons = 0
    numberOfInputNeurons = 0
    numberOfOutputNeurons = 1
    
    adjacencyMatrix = [[]]
    #vertexList = []
    
    matrixColumns = 0
    matrixRows = 0
    
    testDataFrame = pd.DataFrame()
    trainingDataFrame = pd.DataFrame()
    
    networkOutput = 0
    learningRate = 0.5
    learningRateCounter = 1
    outputError = 0

    # This function will initialize the data points
    def __init__(self, layers, hNeuron, testDataFrame, trainingDataFrame):
        self.numberOfHiddenLayers = layers
        self.numberOfHiddenNeurons = hNeuron
        self.numberOfInputNeurons = testDataFrame.shape[1] - 1
        self.matrixColumns = self.numberOfInputNeurons + (self.numberOfHiddenLayers * self.numberOfHiddenNeurons) + self.numberOfOutputNeurons
        self.matrixRows = self.matrixColumns
        self.testDataFrame = testDataFrame
        self.trainingDataFrame = trainingDataFrame
        self.outputErrorAcc = 0
        self.learningRate = 0.75

    def initializeGraph(self):
        #Weights are now initialized in the matrix
        
        self.adjacencyMatrix = [[0 for i in range(self.matrixColumns)] for j in range(self.matrixRows)]
                    
        #initialize Vertex list
        # for i in range (self.numberOfInputNeurons):
        #     self.vertexList.append("IL" + str(i))
        
        # for i in range (self.numberOfHiddenLayers):
        #     identifier = "HL" + str(i);
        #     for j in range (self.numberOfHiddenNeurons):
        #         self.vertexList.append(identifier + str(j))
        # for i in range(self.numberOfOutputNeurons):
        #     self.vertexList.append("OL" + str(i))
                
        #Now the matrix is full of 0, we need to put the weights were they make sense
        #we need the input layer and hidden node connected - special case
        #We then need the hidden nodes connected - loop
        #We need to connect the last hidden node to the output node - special case
        
        
        #First Case: Input layer to hidden 1
        random.seed(8888)
        for i in range (self.numberOfHiddenNeurons):
            for j in range (self.numberOfInputNeurons):
                randomNum = random.uniform(-0.5, 0.5)
                self.adjacencyMatrix[self.numberOfInputNeurons + i][j] = randomNum
        
        #Must be reseeded to get the same numbers
        # random.seed(123)
        # for i in range (self.numberOfInputNeurons):
        #     for j in range (self.numberOfHiddenNeurons):
        #         randomNum = random.uniform(0, 1)
        #         self.adjacencyMatrix[i][self.numberOfInputNeurons + j] = randomNum
                
                
        #Second Case: Hidden Layers connected to eachother
        # countOfHiddens = 1
        # while(countOfHiddens < self.numberOfHiddenLayers):
        #     for i in range (self.numberOfHiddenNeurons):
        #         for j in range (self.numberOfHiddenNeurons):
        #             randomNum = random.uniform(-0.5, 0.5)
        #             self.adjacencyMatrix[self.numberOfInputNeurons + self.numberOfHiddenNeurons * countOfHiddens + i][self.numberOfInputNeurons + self.numberOfHiddenNeurons*(countOfHiddens-1) + j] = randomNum
        #             #self.adjacencyMatrix[self.numberOfInputNeurons + self.numberOfHiddenNeurons*(countOfHiddens-1) + i] [self.numberOfInputNeurons + self.numberOfHiddenNeurons * countOfHiddens + j] = randomNum
            
        #     countOfHiddens = countOfHiddens + 1

        #Thid Case: Last Hidden to Output
        random.seed(8888)
        for i in range(self.numberOfOutputNeurons):
            for j in range(self.numberOfHiddenNeurons):
                randomNum = random.uniform(-0.5, 0.5)
                self.adjacencyMatrix[self.numberOfHiddenNeurons + self.numberOfInputNeurons + i][self.numberOfInputNeurons + j] = randomNum
        
        #THIS IS THE ISSUE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
        # random.seed(123)
        # for i in range(self.numberOfHiddenNeurons):
        #     for j in range(self.numberOfOutputNeurons):
        #         randomNum = random.uniform(0, 1)
        #         print(((self.numberOfHiddenLayers -1) * self.numberOfHiddenNeurons + self.numberOfInputNeurons + i), self.numberOfHiddenLayers * self.numberOfHiddenNeurons + self.numberOfInputNeurons + j)
        #         #self.adjacencyMatrix = [(self.numberOfHiddenLayers -1) * self.numberOfHiddenNeurons + self.numberOfInputNeurons + i][self.numberOfHiddenLayers * self.numberOfHiddenNeurons + self.numberOfInputNeurons + j] = randomNum
                
        #We would initialize biases here, but we want to start them at 0 anyways     
        random.seed(8888)
        index = self.numberOfHiddenNeurons + self.numberOfOutputNeurons
        for i in range(index):
                self.adjacencyMatrix[self.numberOfInputNeurons + i][self.numberOfInputNeurons + i] = random.uniform(-0.5, 0.5)
                
  
        
    #coded for one output neuron
    def backPropagation(self, dataRow, epoch):
        #Output Layer Error
        outputError = self.networkOutput * (1-self.networkOutput)* (dataRow[len(dataRow) -1 ] - self.networkOutput)

        #Output Node Bias Updated
        if epoch == 1 :
            s=0

        deltaBias = outputError * self.learningRate
        self.adjacencyMatrix[self.numberOfInputNeurons + self.numberOfHiddenNeurons][self.numberOfInputNeurons + self.numberOfHiddenNeurons] += deltaBias
       
        #Last hidden to output weights updated
        for i in range (self.numberOfHiddenNeurons):
            for j in range(self.numberOfInputNeurons):
            # deltaWeight = self.learningRate * outputError * self.adjacencyMatrix[self.numberOfInputNeurons + self.numberOfHiddenNeurons][self.numberOfInputNeurons + i]
                deltaWeight = self.learningRate * outputError * self.adjacencyMatrix[j][self.numberOfInputNeurons + i]
                self.adjacencyMatrix[self.numberOfHiddenNeurons + self.numberOfInputNeurons][self.numberOfInputNeurons + i] += deltaWeight
        

        errorList = [0]* self.numberOfHiddenNeurons
        #this will multiply the output error with the weights of each hidden neuron. The summation part of the formula 
        for i in range (self.numberOfHiddenNeurons):
                errorList[i] = self.adjacencyMatrix[self.numberOfInputNeurons + self.numberOfHiddenNeurons][self.numberOfInputNeurons + i] * outputError
                
        for i in range (self.numberOfHiddenNeurons):
                errorList[i] = errorList[i] * self.adjacencyMatrix[i][self.numberOfInputNeurons] * (1 - (self.adjacencyMatrix[i][self.numberOfInputNeurons]))
                
        #At this point, the for loop did the summation and the seocnd for loop multipled in the rest of the components
        #The error list now holds the respective errors
        
        #Now, process the weights and bias for the input - hidden layer connection
        
        #Update Bias Here
        for i in range(self.numberOfHiddenNeurons):
            biasChange = self.learningRate * errorList[i]
            self.adjacencyMatrix[self.numberOfInputNeurons + i][self.numberOfInputNeurons + i] += biasChange
        
        #Update the weights here
        for i in range (self.numberOfHiddenLayers):
            for j in range (self.numberOfInputNeurons):
                #weight change is based on output so we dont need to increment by j
                weightChange = self.learningRate * errorList[i] * dataRow[j]
                
                self.adjacencyMatrix[self.numberOfInputNeurons + i][j] += weightChange
        
        # self.learningRateCounter += 1
        # self.learningRate = 1/self.learningRateCounter
        self.outputErrorAcc += outputError
        
        
    def trainNeuralNetwork(self, epoch_num):
        #We need to pass values into the input layer
        #We then need to calculate the summation + bias for each input data point
        #We need to save these varaibles in a list
        #We need to do this to all the input variables
        #Each time we calcualte a new value, go to that list position and add weight * value
        
        
        #This will increment the numbers of rows
        testData = self.trainingDataFrame.to_numpy()
       
        runCounter = 0
        while(runCounter < len(testData)):
            #Take care of the input values 
            resultsList = [0] * self.numberOfHiddenNeurons
            outputValue = 0
            
            for i in range (self.numberOfHiddenNeurons):
                for j in range (self.numberOfInputNeurons):
                    resultsList[i]  = resultsList[i] + self.adjacencyMatrix[self.numberOfInputNeurons + i][j] * testData[runCounter][j]
            #at this point, the resultsList will hold the input value into the first hidden layer
            
            #Now add the bias to the values
            for i in range(len(resultsList)):
                resultsList[i] = resultsList[i] + self.adjacencyMatrix[self.numberOfInputNeurons + i][self.numberOfInputNeurons + i]                
                #print(resultsList[i])
                resultsList[i] = 1.0 / (1 + math.exp(-1*resultsList[i]))
                
            
            for i in range (self.numberOfHiddenNeurons):
                for j in range (self.numberOfInputNeurons):
                    #print(resultsList[i])
                    self.adjacencyMatrix[j][self.numberOfInputNeurons + i] = resultsList[i]
            #Now, the output values are in columns in the upper half of the graph

            #Process hidden layers 
            # countOfHiddens = 1
            # while(countOfHiddens < self.numberOfHiddenNeurons):
            #     tempResultsList = [0]* self.numberOfHiddenNeurons 
            #     for i in range (self.numberOfHiddenNeurons):
            #         for j in range (self.numberOfHiddenNeurons):
            #             tempResultsList[i] = tempResultsList[i] + self.adjacencyMatrix[self.numberOfInputNeurons + self.numberOfHiddenNeurons * countOfHiddens + i][self.numberOfInputNeurons + self.numberOfHiddenNeurons*(countOfHiddens-1) + j]* resultsList[j]
                
            #     resultsList = tempResultsList
                
            #     for i in range(len(resultsList)):
            #         resultsList[i] = resultsList[i] + self.adjacencyMatrix[self.numberOfInputNeurons + self.numberOfHiddenNeurons * countOfHiddens + i][self.numberOfInputNeurons + self.numberOfHiddenNeurons * countOfHiddens + i]
            #         resultsList[i] =  1 / (1 + math.exp(-1*resultsList[i]))
                
            #     for i in range (self.numberOfHiddenNeurons):
            #         for j in range (self.numberOfHiddenNeurons):
            #             self.adjacencyMatrix[self.numberOfInputNeurons + self.numberOfHiddenNeurons*(countOfHiddens-1) + j][self.numberOfInputNeurons + self.numberOfHiddenNeurons * countOfHiddens + i] = resultsList[i]
                        
            #     countOfHiddens = countOfHiddens + 1
            
            #Output Layer Neuron(s)
            for i in range(self.numberOfOutputNeurons):
                for j in range(self.numberOfHiddenNeurons):
                    outputValue = outputValue + self.adjacencyMatrix[self.numberOfHiddenNeurons + self.numberOfInputNeurons + i][self.numberOfInputNeurons + j] * resultsList[j]
                    
            
            # outputValue = outputValue + self.adjacencyMatrix[self.numberOfInputNeurons + self.numberOfHiddenNeurons][self.numberOfInputNeurons + self.numberOfHiddenNeurons]
            # outputValue =  1 / (1 + math.exp(-1*resultsList[i]))

            outputValue =  1 / (1 + math.exp(-1*outputValue))

            for i in range(self.numberOfOutputNeurons):
                for j in range(self.numberOfHiddenNeurons):
                    self.adjacencyMatrix[self.numberOfInputNeurons + j][self.numberOfHiddenNeurons + self.numberOfInputNeurons + i] = outputValue
                    
            
            self.networkOutput = outputValue
            self.backPropagation(testData[runCounter], epoch_num)
            
            runCounter += 1
           

    def runTestData(self, outputFile):
        TP = 0 #Predicted = yes, actual = yes
        FP = 0 #Predicted = yes, actual = No
        FN = 0 #Predicted = no, actual = yes
        TN = 0 #Predicted = no, actual = no
    

        #This will increment the numbers of rows
        testData = self.testDataFrame.to_numpy()
        
        runCounter = 0
        while(runCounter < len(testData)):
            resultsList = [0] * self.numberOfHiddenNeurons
            outputValue = 0
            
            for i in range (self.numberOfHiddenNeurons):
                for j in range (self.numberOfInputNeurons):
                    resultsList[i]  = resultsList[i] + self.adjacencyMatrix[self.numberOfInputNeurons + i][j] * testData[runCounter][j]
            #at this point, the resultsList will hold the input value into the first hidden layer
            
            #Now add the bias to the values
            for i in range(len(resultsList)):
                resultsList[i] = resultsList[i] + self.adjacencyMatrix[self.numberOfInputNeurons + i][self.numberOfInputNeurons + i]                
                #print(resultsList[i])
                resultsList[i] = 1.0 / (1 + math.exp(-1*resultsList[i]))

            #Process hidden layers 
            # countOfHiddens = 1
            # while(countOfHiddens < self.numberOfHiddenNeurons):
            #     tempResultsList = [0]* self.numberOfHiddenNeurons 
            #     for i in range (self.numberOfHiddenNeurons):
            #         for j in range (self.numberOfHiddenNeurons):
            #             tempResultsList[i] = tempResultsList[i] + self.adjacencyMatrix[self.numberOfInputNeurons + self.numberOfHiddenNeurons * countOfHiddens + i][self.numberOfInputNeurons + self.numberOfHiddenNeurons*(countOfHiddens-1) + j]* resultsList[j]
                
            #     resultsList = tempResultsList
                
            #     for i in range(len(resultsList)):
            #         resultsList[i] = resultsList[i] + self.adjacencyMatrix[self.numberOfInputNeurons + self.numberOfHiddenNeurons * countOfHiddens + i][self.numberOfInputNeurons + self.numberOfHiddenNeurons * countOfHiddens + i]
            #         resultsList[i] =  1 / (1 + math.exp(-1*resultsList[i]))
                
            #     countOfHiddens = countOfHiddens + 1
            
            
            #Output Layer Neuron(s)
            for i in range(self.numberOfOutputNeurons):
                for j in range(self.numberOfHiddenNeurons):
                    outputValue = outputValue + self.adjacencyMatrix [self.numberOfHiddenNeurons + self.numberOfInputNeurons + i][self.numberOfInputNeurons + j] * resultsList[j]
                    
            
            outputValue = outputValue + self.adjacencyMatrix[self.numberOfInputNeurons + self.numberOfHiddenNeurons][self.numberOfInputNeurons + self.numberOfHiddenNeurons]
            outputValue =  1 / (1 + math.exp(-1*resultsList[i]))    
            
            #Now we have the output value
            #print(outputValue)
            if(outputValue>0.5):
                if(self.testDataFrame["class"].iloc[runCounter] == 1):
                    #Predicted >, actual >
                    TP = TP + 1
                else:
                    #Predicted >, actual <
                    FP = FP + 1
    
            elif(outputValue<=0.6):
                if(self.testDataFrame["class"].iloc[runCounter] == 0):
                    #Predicted <, actaul = <
                    TN = TN + 1
                else:
                    #Predicted <, actual >
                    FN = FN + 1
                
            runCounter += 1
        # print(TP)
        # print(TN)
        # print(FP)
        # print(FN)
        accuracy = (TP + TN)/(TP+TN+FP+FN)
        print(f'Accuracy: {accuracy}')
        precisionPositive = 0
        precisionNegative = 0
        recallPositive = 0
        recallNegative = 0
        positiveF1Measure = 0
        negativeF1Measure = 0
    
        if(TP != 0):
            precisionPositive = TP/(TP + FP)  
            
        if(TN != 0):
            precisionNegative = TN/ (TN + FN)
            
        if(TP != 0):
            recallPositive = TP/ (TP + FN)        
            
        if(TN != 0):
            recallNegative = TN/ (TN + FP)
            
        if(recallPositive and precisionPositive != 0):
            positiveF1Measure = (2* recallPositive * precisionPositive)/ (recallPositive + precisionPositive)
            
        if(recallNegative and precisionNegative != 0):
            negativeF1Measure = (2* recallNegative * precisionNegative)/ (recallNegative + precisionNegative)
        
        macroPrecision = (precisionPositive + precisionNegative)/2
        macroRecall = (recallPositive + recallNegative)/2
        macroF1Measure = (positiveF1Measure + negativeF1Measure)/2
        
        #TN is a true positive for the second class value
        microPrecision = (TP + TN)/(TP + TN + FP + FN)
        microRecall = (TP + TN)/ (TP + TN + FN + FP)
        microF1Measure = (2 * microPrecision * microRecall)/ (microPrecision + microRecall)

        outputFile.write("Micro Precision: " + str(microPrecision) + "\n")
        outputFile.write("Micro Recall: " + str(microRecall)+ "\n")
        outputFile.write("Micro F1 Measure: " + str(microF1Measure)+ "\n")
    
        outputFile.write("Macro Precision: " + str(macroPrecision)+"\n")
        outputFile.write("Macro Recall: " + str(macroRecall)+"\n")
        outputFile.write("Macro F1 Measure: " + str(macroF1Measure)+"\n")
        outputFile.write("Accuracy: " + str(accuracy)+"\n")
            

            
        
        
        