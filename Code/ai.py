import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

match_data_path = 'momentum_file_final.csv'
df1 = pd.read_csv(match_data_path)
df = df1[df1['match_id'] == '2023-wimbledon-1311']
df['Momentum_delta'] = df['Momentum_delta'].shift(-1)
df = df[:-1]
df['Momentum_delta_categorical'] = df['Momentum_delta'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

conditions = [
    df['p1_winner'] == 1,
    df['p2_winner'] == 1
]

choices = [1,2]

default = 0

df['winner'] = np.select(conditions, choices, default=default)

X = df[['ace', 'winner', 'double_fault', 'consecutive_points_won', 'backhand_won', 'unf_err', 'consecutive_points_loss']]
y = df['Momentum_delta_categorical']
y = y.astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Or RandomForestClassifier for classification
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred) 
print(f"Mean Squared Error: {mse}") 
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
feature_importances = model.feature_importances_
features = X.columns
importances = pd.Series(feature_importances, index=features)
sorted_importances = importances.sort_values(ascending=False)

# Display the sorted importances
print(sorted_importances)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['-1', '0', '1'])

# Convert the confusion matrix to a DataFrame for better labeling
cm_df = pd.DataFrame(cm,
                     index=['-1', '0', '1'],  # Actual labels
                     columns=['-1', '0', '1'])  # Predicted labels

# Plotting the confusion matrix
............
plt.show()