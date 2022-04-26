import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import  accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn import pipeline 
from sklearn.svm import SVC
import time
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
# x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)
# extra_sample = new_data.sample(new_data.shape[0] // 4)
# x_train = pd.concat([x_train, extra_sample.drop(['output'], axis=1 ) ])
# y_train = pd.concat([y_train, extra_sample['output'] ])

# scaler = StandardScaler()
# X_train = scaler.fit_transform(x_train)
# X_test = scaler.transform(x_test)


# rf = RandomForestClassifier(random_state=0, max_depth=4, n_estimators=200)
# rf.fit(X_train,y_train)
# pred =  rf.predict(X_test)
# acc = accuracy_score(y_test, pred)

# print(acc)
scaler_Models = pipeline.Pipeline(steps=[('scaling' , StandardScaler())])

tree_classifiers = {
  "Decision Tree": DecisionTreeClassifier(random_state=0),
  "Extra Trees":ExtraTreesClassifier(random_state=0),
  "Random Forest":RandomForestClassifier(random_state=0),
  "AdaBoost":AdaBoostClassifier(random_state=0),
  "Skl GBM": GradientBoostingClassifier(random_state=0),
  "Skl HistGBM":HistGradientBoostingClassifier(random_state=0),
  "XGBoost": XGBClassifier(random_state=0),
  "LightGBM":LGBMClassifier(random_state=0),
  "CatBoost": CatBoostClassifier(random_state=0),
  "Svm":      SVC(random_state = 0)}
tree_classifiers = {name: pipeline.make_pipeline(scaler_Models, model) for name, model in tree_classifiers.items()} 

#split the data into train set and test set
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0, test_size=0.2)

# lets run the model befor data enhancement 

results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})


for model_name, model in tree_classifiers.items():
    start_time = time.time()
    
    # FOR EVERY PIPELINE (PREPRO + MODEL) -> TRAIN WITH TRAIN DATA (x_train)
    # tree_prepro.fit(x_train)
    # X_train_transformed = tree_prepro.transform(x_train)
    # X_train_transformed = pd.DataFrame(X_train_transformed, columns=list(num_vars) + list(cat_vars))
    model.fit(x_train,y_train)
    # GET PREDICTIONS USING x_val
    pred = model.predict(x_test)

    total_time = time.time() - start_time

    results = results.append({"Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_test, pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_test, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)
results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
print(results)

#adding an extra data(25 percent of the enhanced data) to the original data
extra_sample = new_data.sample(new_data.shape[0] // 4)
x_train_enh = pd.concat([x_train, extra_sample.drop(['output'], axis=1 ) ])
y_train_ehn = pd.concat([y_train, extra_sample['output'] ])


results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})


for model_name, model in tree_classifiers.items():
    start_time = time.time()

    model.fit(x_train_enh,y_train_ehn)
   
    pred = model.predict(x_test)

    total_time = time.time() - start_time

    results = results.append({"Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_test, pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_test, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)
                              




results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
print(results)


