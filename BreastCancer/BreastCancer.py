import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle

#Original code at: 
#https://colab.research.google.com/drive/1LghdzAGSE9W5IRc7P0KMl9z9DQFv0bIJ?usp=sharing

def predict_output(model, input):
  prediction = model.predict_proba(input)
  return prediction[0]

if __name__== '__main__':
    
    filename = 'BreastCancer_model.sav'
    model = pickle.load(open(filename, 'rb'))
    
    #model.classes_: array([0, 1]) -> 0: Benign, 1: Malignant
    
    input0 = [[566.3, 0.06664, 0.04781, 15.11, 19.26, 99.7, 711.2, 0.1288]]
    prediction = predict_output(model, input0)
    print('Probability sample is benign: ' + str(prediction[0]*100) + '%')
    print('Probability sample is Malignant: ' + str(prediction[1]*100) + '%')
    
    #Output0:
    #Probability sample is benign: 100.0%
    #Probability sample is Malignant: 0.0%
    
    input1 = [[1001, 0.3001, 0.1471, 25.38, 17.33, 184.6, 2019, 0.2654]]
    prediction = predict_output(model, input1)
    print('Probability sample is benign: ' + str(prediction[0]*100) + '%')
    print('Probability sample is Malignant: ' + str(prediction[1]*100) + '%')
    
    #Output1:
    #Probability sample is benign: 1.0%
    #Probability sample is Malignant: 99.0%
