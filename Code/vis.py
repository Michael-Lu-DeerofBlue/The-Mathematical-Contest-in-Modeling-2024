import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
# Load the CSV file into a DataFrame
df = pd.read_csv('Wimbledon_featured_matches_copy.csv')

def cal_momentum(con_win, untouch, back, con_loss, unf_err ):
  b_0 =  0.009394573521668753;
  b_1 = 0.16845085;
  b_2 = 0.00091111;
  b_3 = 0.0163816;
  b_4 = 0.03170258;
  b_5 = 0.017100571;
  result = (1 + b_1 * con_win + b_2 * untouch + b_3 * back) / (1+ b_4 * con_loss + b_5 * unf_err);
  return float(result)

# Calculate
df['Momentum_1'] = 0;
df['Momentum_2'] = 0;
for index, row in df.iterrows():
  if index > 1:
    df.at[index, 'Momentum_1'] = cal_momentum(row['p1_consecutive_points_won'], row['p1_winner'], row['p1_backhand_won'], row['p1_consecutive_points_loss'], row['p1_unf_err'])
    df.at[index, 'Momentum_2'] = cal_momentum(row['p2_consecutive_points_won'], row['p2_winner'], row['p2_backhand_won'], row['p2_consecutive_points_loss'], row['p2_unf_err'])
df['Momentum_delta'] = df['Momentum_1'] - df['Momentum_2']

# Plot
line_label = 'Momentum Differential Positive: Carlos Negative: Nicolas'
plt.figure(figsize=(40, 5))
plt.plot(df['elapsed_time'].head(300), df['Momentum_delta'].head(300), marker='o', linestyle='-', color='b', label = line_label)
plt.title('Game Momentum and Game Point Won (Carlos Alcaraz vs. Nicolas Jarry)')
plt.xlabel('Elpased Time(Seconds)')
plt.ylabel('Momentum Differential')
plt.grid(True)

for index, row in df.head(300).iterrows():
  if row['game_victor'] == 1:
    plt.plot(row['elapsed_time'],1, 'o', color='green', markersize=8, zorder=3, label = 'Game won by Carlos')
    if g == 0:
      plt.legend();
      g = 1
  elif row['game_victor'] == 2:
    plt.plot(row['elapsed_time'],-1, 'o', color='red', markersize=8, zorder=3, label = 'Game won by Nicolas')
    if r == 0:
      plt.legend();
      r = 1
    
plt.axhline(y=0, color='grey', linestyle='--', zorder=1)
plt.ylim(-1.5, 1.5)
plt.show()