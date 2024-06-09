import pickle

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns;
#sns.set_theme(color_codes=True)
import pandas


for tissue in ["Bladder", "Kidney", "Prostate"]:
    with open("/home/fruitdragon/workspace/NeuralClassifier/old/data/MIXED/"+tissue+"_Tree10.pkl", 'rb') as cancerFile:
        patients = pickle.load(cancerFile)
    #with open("/home/fruitdragon/workspace/NeuralClassifier/old/data/MIXED/Kidney_Tree10.pkl", 'rb') as cancerFile:
    #    patients = pickle.load(cancerFile)
    #with open("/home/fruitdragon/workspace/NeuralClassifier/old/data/MIXED/Prostate_Tree10.pkl", 'rb') as cancerFile:
    #    patients = pickle.load(cancerFile)
    
    
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
    
    
    print("in " +tissue +" throwing out " + str(badKeys))
        
    for key in badKeys:
        for setName in["training", "validation", "testing"]:
            for patient in patients[setName]:
                del patient[key]
        print("Thrown out " + key)
        print(patients["training"][0].keys())
    
    
    
    
    
    rowColorsList = []
    allPatients = []
    for patient in patients["training"]:
        data = []
        for key in patient.keys():
            if(key != "Cancer") :
                data.append(patient[key])
            else:
                if(patient[key]):
                    rowColorsList.append(("#f65844"))
                else:
                    rowColorsList.append(("#72c9d4"))
        allPatients.append(data)
    
    
    
    
    
    
    df = pandas.DataFrame(allPatients, columns = list(patient.keys())[:-1])
    
    rowColors = pandas.DataFrame(rowColorsList, columns = ["Cancer or Normal"])
    
    #g = sns.clustermap(df, cmap='jet')
    g = sns.clustermap(df, method = "ward", figsize=(10, 14), row_colors=rowColors, cmap='Spectral_r')
    
    
    #fig = plt.figure(figsize=(12,6))
    #ax = fig.add_subplot(111)
    #plt.setp(ax, xticklabels=list(patient.keys())[:-1])
    #plt.xticks(rotation=-90)
    
    
    #plt.show()
    #g.savefig("test_clustermap.png")
    
    with PdfPages(tissue+"_clustermap.pdf") as pdf:
        pdf.savefig()
    
    
print("Finished")
    
    
