import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, plot_confusion_matrix


# Load data
data = pd.read_csv('data/heart.csv')

# remove duplicates
data = data.drop_duplicates()

# Train test sets
X = data.drop(columns='output')
y = data['output']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Preprocessor
num_var = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
cat_var = [col for col in X_train.columns if col not in num_var]

num_prep = ColumnTransformer([('num_prepo', StandardScaler(), num_var)],
                             remainder='passthrough')

# Model
rf = RandomForestClassifier(random_state=0, max_depth=4, n_estimators=200)

# Model pipeline
rf_pipe = Pipeline([('prep', num_prep), ('rf', rf)])

# Fit
rf_pipe_fitted = rf_pipe.fit(X_train, y_train)

# Performance (initial score is 0.89)
pred = rf_pipe_fitted.predict(X_test)

print(f'accuracy_score: {round(accuracy_score(y_test, pred), 2) }')

# Confusion matrix
plot_confusion_matrix(rf_pipe, X_test, y_test)
plt.show()
