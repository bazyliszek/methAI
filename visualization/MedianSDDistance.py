import pickle

# - ((CancerMedian - cutoff)/CancerSD) * ((NormalMedian - cutoff)/NormalSD) 
cutoff = 0.3

for tissue in ["Bladder", "Kidney", "Prostate"]:
    #tissue = "Prostate"
    with open("/home/fruitdragon/workspace/NeuralClassifier/old/data/MIXED/"+tissue+"_Tree10.pkl", 'rb') as cancerFile:
        patients = pickle.load(cancerFile)
    
    for key in patients["training"][0].keys():
        if key == "Cancer":
            continue
        
        cancerGroup = []
        normalGroup = []
        for i in patients["training"]:
            if i["Cancer"]:
                cancerGroup.append(i[key])
            else:
                normalGroup.append(i[key])
        
        cancerGroup.sort()
        normalGroup.sort()
        
        cancerMedian = cancerGroup[int(len(cancerGroup)/2)]
        normalMedian = normalGroup[int(len(normalGroup)/2)]
        
        cancerMean = sum(cancerGroup) / len(cancerGroup) 
        cancerVar = sum([((x - cancerMean) ** 2) for x in cancerGroup]) / len(cancerGroup) 
        cancerSD = cancerVar ** 0.5
        
        normalMean = sum(normalGroup) / len(normalGroup) 
        normalVar = sum([((x - normalMean) ** 2) for x in normalGroup]) / len(normalGroup) 
        normalSD = normalVar ** 0.5
        
        sepScore = -((cancerMedian-cutoff)/cancerSD) * ((normalMedian-cutoff)/normalSD)
        
        #print (tissue + " : " + key + " : Meds = " + str(cancerMedian) + "," + str(normalMedian) + " : SDs = " + str(cancerSD) + "," + str(normalSD) + " : sepscore = " + str(sepScore))
        print (tissue + " : " + key + " : sepscore = " + str(round(sepScore, 3)))
        
    print()
    
    
