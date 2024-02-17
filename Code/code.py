import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression

data = pd.read_csv('modified_file.csv')
selected_columns = ['max_con_point','serve','cum_untouchable','cum_ace','cum_b','max_con_point_loss','cum_unf_err']
X = data[selected_columns]
y = data['game_won']
print(type(X))

model = LinearRegression()

# Fit the model on the data
model.fit(X, y)

# Display the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
# Predict the target values
y_pred = model.predict(X)

# Convert the predictions to integers (classification)
y_pred_class = [round(pred) for pred in y_pred]
accuracy = accuracy_score(y, y_pred_class)

print("Accuracy:", accuracy)