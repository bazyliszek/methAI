import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def drawGraph(patientsFile, tissueType):
    cancerPointsX = []
    cancerPointsY = []
    normalPointsX = []
    normalPointsY = []
    with open(patientsFile, 'rb') as loadFrom:
        patients = pickle.load(loadFrom)
        for patient in patients["training"]:
            if(patient["Cancer"]):
                cancerPointsX.append(patient["PCA0"])
                cancerPointsY.append(patient["PCA1"])
            else:
                normalPointsX.append(patient["PCA0"])
                normalPointsY.append(patient["PCA1"])
                
        for patient in patients["validation"]:
            if(patient["Cancer"]):
                cancerPointsX.append(patient["PCA0"])
                cancerPointsY.append(patient["PCA1"])
            else:
                normalPointsX.append(patient["PCA0"])
                normalPointsY.append(patient["PCA1"])
            
        
        
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title(tissueType + ' Methylation PCA2', fontsize = 20)
    
    ax.scatter(cancerPointsX, cancerPointsY, c = 'r', s = 20)
    ax.scatter(normalPointsX, normalPointsY, c = 'b', s = 20)
    
    ax.legend(["cancer","normal"])
    ax.grid()

    #plt.show()
        
    with PdfPages(tissueType + ' Methylation PCA2.pdf') as pdf:
        pdf.savefig()
        
        
        
'''#Marcin Version
pickleLocation = "/rds/projects/2018/colboujk-wojewodzic/nnetwork/newdata/prostate_PCA2.pkl"
'''#Jan Version
#cancerLoc = "/home/fruitdragon/workspace/NeuralClassifier/old/data/bladderCancerSmall.pkl"
#healthyLoc = "/home/fruitdragon/workspace/NeuralClassifier/old/data/bladderHealthySmall.pkl"
#cancerLoc = "/home/fruitdragon/workspace/NeuralClassifier/old/data/bladderCancer_full.pkl"
#healthyLoc = "/home/fruitdragon/workspace/NeuralClassifier/old/data/bladderHealthy_full.pkl"

pickleLocation = "/home/fruitdragon/workspace/NeuralClassifier/old/data/141119/prostate_PCA2.pkl"
#'''

drawGraph(pickleLocation, "Prostate")
