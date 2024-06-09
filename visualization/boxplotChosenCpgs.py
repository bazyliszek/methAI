import pickle

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages




for cancerType in ["Bladder","Kidney","Prostate"]:
    for cancerType2 in [cancerType]:
    
        pickleFile = "/home/fruitdragon/workspace/NeuralClassifier/old/data/MIXED/"+cancerType+"_Tree10.pkl"
        if(cancerType == cancerType2):
            pickleFile = "/home/fruitdragon/workspace/NeuralClassifier/old/data/MIXED/"+cancerType+"_Tree10.pkl"
        else:
            pickleFile = "/home/fruitdragon/workspace/NeuralClassifier/old/data/MIXED/"+cancerType+cancerType2+"Mixed.pkl"
        
        with open(pickleFile, 'rb') as cancerFile:
            patients = pickle.load(cancerFile)
            
            
        patientKeys = list(patients["training"][0].keys())
        
        #Toss out any with the same median between both cancer and normal groups
        badKeys = []
        for key in patientKeys:
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
                badKeys.append(key)
                
        print("in " +cancerType +" throwing out " + str(badKeys))
                        
        for key in badKeys:
            for setName in["training", "validation"]:
                for patient in patients[setName]:
                    del patient[key]
            patientKeys.remove(key)
                    
                    
                
        
        
        #for setName in ["training", "validation"]:
        #    for patient in patients[setName]:
        #        for key in patient.keys():
        #            if(patient[key]>0.3):
        #                patient[key] = 1
        #            else:
        #                patient[key] = 0
        
        
        
        #boxes = []
        cancerBoxes  = [[]]
        labelBoxes  = [[]]
        normalBoxes = [[]]
        labels = [""]
        
        for key in patientKeys:
            if(key == "Cancer"):
                continue
            labels.append(key)
            cancerBox = []
            normalBox = []
            for patient in patients["training"]:
                if(patient["Cancer"]):
                    cancerBox.append(patient[key])
                else:
                    normalBox.append(patient[key])
            cancerBoxes.append(cancerBox)
            normalBoxes.append(normalBox)
            labelBoxes.append([])
            #colors.append("r")
            #colors.append("b")
        
        
        
        cancerBoxes.append([])
        normalBoxes.append([])
        labelBoxes.append([])
        labels.append("")
        
        
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111)
        fig.suptitle(cancerType +' Beta Values', y=0.95)
        #ax = fig.add_subplot(211)
        
        
        
        
        pos = []
        for i in range(0, 5*len(cancerBoxes), 5):
            pos.append(i-1)
        box1 = plt.boxplot(cancerBoxes, positions=pos)
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box1[item], color="#ff5555")
            #plt.setp(box1["boxes"], facecolor=colors[i])
            plt.setp(box1["fliers"], markeredgecolor="#ff5555")
            
            
        
        
        
        pos = []
        for i in range(0, 5*len(cancerBoxes), 5):
            pos.append(i+1)
        box1 = plt.boxplot(normalBoxes, positions=pos)
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box1[item], color="#5588ff")
            #plt.setp(box1["boxes"], facecolor=colors[i])
            plt.setp(box1["fliers"], markeredgecolor="#5588ff")
        
            
            
        
        
        pos = []
        for i in range(0, 5*len(cancerBoxes), 5):
            pos.append(i)
        box1 = plt.boxplot(labelBoxes, positions=pos)
        
        
        
        plt.setp(ax, xticklabels=labels)
        plt.xticks(rotation=-90)
        
        #ax.set_xlabel("locations")
        ax.set_ylabel('Percentage methylation')
        plt.axhline(0.3, c="#777777", ls = ":")
        
        c = ["#ff5555", "#5588ff"]
        
        legends = ["Cancer", "Normal"]
        
        legend = ax.legend(legends, bbox_to_anchor=(0.97,0.2), bbox_transform=plt.gcf().transFigure)
        legend.legendHandles[1].set_color("#5588ff")
        
        
        fig.tight_layout()
        plt.show()
        
        
        plt.close()
print("FINISHED")
