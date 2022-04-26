import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold,cross_val_predict,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import time
# Load data
data = pd.read_csv('heart_attack\heart.csv')

# remove duplicates
data = data.drop_duplicates()
X = data.drop(['output'],axis=1)
y = data['output']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


# Preprocessor
num_var = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
cat_var = ['sex','cp','fbs','restecg','exng','slp','caa','thall']
tree_prep = ColumnTransformer(transformers=[('ordinal', OrdinalEncoder(),  
                            cat_var), ('scaler', StandardScaler(), num_var)], remainder='drop')

# models dictionary
classifiers = {'Random Forest': RandomForestClassifier(n_estimators=100),
                'AdaBoost': AdaBoostClassifier(n_estimators=50, algorithm='SAMME.R', learning_rate=1),
                'Gradient Boost': GradientBoostingClassifier(n_estimators=100),
                'Extratree': ExtraTreesClassifier(n_estimators=100),
                'Svc':SVC(C=1),
                'Logis_Regres':LogisticRegression()}

classifiers = {name: make_pipeline(tree_prep,model) for name, model in classifiers.items()}      

# checking accuracy in differnt models

results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})
for model_name, model in classifiers.items():
    start_time = time.time()
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    total_time = time.time()- start_time
    results = results.append({'Model': model_name,
                            'Accuracy': metrics.accuracy_score(y_test, predictions)*100,
                            'Bal Acc.': metrics.balanced_accuracy_score(y_test,predictions)*100,
                            'Time': total_time},
                            ignore_index=True)

results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
results_ord.index += 1 
print(results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d'))       

#creating best model




def predic(pth):
    best_model = classifiers['AdaBoost']
    best_model.fit(X_train,y_train)
    predicts = best_model.predict(pth)
    

    return predicts