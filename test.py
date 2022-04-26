from time import sleep, time
import numpy as np
import heart_attack as ht
import  pandas as pd 
from heart_attack import predic

def test_inputs():
    input_features = []

    age = int(input('How old are you?\n '))
    sex = int(input('What is your gender? , 1 for Male or 0 for Feamle\n'))
    exng = int(input('Do you do exercises? 1 = yes 0 =no\n'))
    caa = int(input('Number of major vessels. 0, 1, 2 or 3\n'))
    cp = int(input('Chain type pain. 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic\n')) 
    trtbps = int(input('Resting blood pressure (in mm Hg)\n'))
    chol = int(input('cholestoral in mg/dl fetched via BMI sensor\n'))
    fbs = int(input('(Fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)\n'))
    restecg  = int(input('Resting electrocardiographic results. 0, 1, or 2\n'))
    thalachh = int(input('maximum heart rate achieved?\n'))
    oldpeak = float(input('Old peak?\n'))
    slp = int(input('Slope of the peak. 0, 1, 2, 3\n'))
    thall = int(input('Thall: 3- normal, 6- fixed defect, 7-reverseble defect\n'))
    
    
    input_features.append([age,sex,exng,caa,cp,trtbps,chol,fbs,restecg,thalachh,oldpeak,slp,thall])
    return pd.DataFrame(input_features, columns=['age','sex','exng','caa','cp','trtbps','chol','fbs','restecg','thalachh','oldpeak','slp','thall'])

predictions = predic(test_inputs())    
time(sleep(3))
if predictions == 1:
    print('you may have heart attack')
else:
    print('you don\'t have a heart attack')