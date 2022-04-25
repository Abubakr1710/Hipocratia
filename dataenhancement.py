import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score
df = pd.read_csv('heart.csv')
df
np.random.seed(0)
def data_enhancement(df):
    
    data = df
    
    for output in data['output'].unique():
        output_data       =  data[data['output'] == output]
        trtbps_std = output_data['trtbps'].std()
        chol_std = output_data['chol'].std()
        thalachh_std = output_data['thalachh'].std()
       
        
        for i in data[data['output'] == output].index:
            if np.random.randint(2) == 1:
                data['trtbps'].values[i] += trtbps_std/10
            else:
                data['trtbps'].values[i] -= trtbps_std/10
                
            if np.random.randint(2) == 1:
                data['chol'].values[i] += chol_std/10
            else:
                data['chol'].values[i] -= chol_std/10
                
            if np.random.randint(2) == 1:
                data['thalachh'].values[i] += thalachh_std/10
            else:
                data['thalachh'].values[i] -= thalachh_std/10

    return data
new_data = data_enhancement(df)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)
extra_sample = new_data.sample(new_data.shape[0] // 4)
x_train = pd.concat([x_train, extra_sample.drop(['output'], axis=1 ) ])
y_train = pd.concat([y_train, extra_sample['output'] ])

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)


rf = RandomForestClassifier(random_state=0, max_depth=4, n_estimators=200)
rf.fit(X_train,y_train)
pred =  rf.predict(X_test)
acc = accuracy_score(y_test, pred)

print(acc)
