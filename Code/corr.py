import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
df = pd.read_csv('momentum_file.csv')

# runs of success
df['runs_of_success'] = 0
for index, row in df.iterrows():
  next_three = df['consecutive_points_won'][index:index+4]
  df.at[index, 'runs_of_success'] = next_three.max()

# Calculate Pearson correlation coefficient and p-value
col_A = df['Momentum_delta']
col_B = df['runs_of_success']
correlation_coefficient, p_value = pearsonr(col_A, col_B)

# Print the results
print(f"Pearson Correlation Coefficient: {correlation_coefficient}")
print(f"P-Value: {p_value}")

# Check if the correlation is statistically significant
alpha = 0.05
if p_value < alpha:
    print('There is a statistically significant relationship between the variables.')
else:
    print('There is no statistically significant relationship between the variables.')

# swings in play
df['net_p1_runs_of_success'] = df['p1_consecutive_points_won'] - df['p2_consecutive_points_won']
df['net_p1_runs_of_success_t-1'] = df['net_p1_runs_of_success'].shift(1)
df['swings_respected_to_p1'] = df['net_p1_runs_of_success'] - df['net_p1_runs_of_success_t-1']
df['Momentum_delta_t-1'] = df['Momentum_delta'].shift(1)
df['swings_in_Momentum_delta'] = df['Momentum_delta'] - df['Momentum_delta_t-1']

# Calculate Pearson correlation coefficient and p-value
col_A = df['swings_respected_to_p1'].fillna(0)
col_B = df['Momentum_delta'].fillna(0)

correlation_coefficient, p_value = pearsonr(col_A, col_B)

# Print the results
print(f"Pearson Correlation Coefficient: {correlation_coefficient}")
print(f"P-Value: {p_value}")

# Check if the correlation is statistically significant
alpha = 0.05
if p_value < alpha:
    print('There is a statistically significant relationship between the variables.')
else:
    print('There is no statistically significant relationship between the variables.')