# %% 
import pandas as pd
import matplotlib.pyplot as plt

# %%
# parallele trend assumptions
# Sample data
data = {
    'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'unit': ['control'] * 10 + ['treated'] * 10,
    'outcome': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'treated': [0] * 10 + [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Separate pre and post intervention periods
pre_intervention = df[df['time'] <= 5]
post_intervention = df[df['time'] > 5]

# Calculate mean outcomes for control and treated units
pre_control_mean = pre_intervention[pre_intervention['unit'] == 'control']['outcome'].mean()
pre_treated_mean = pre_intervention[pre_intervention['unit'] == 'treated']['outcome'].mean()
post_control_mean = post_intervention[post_intervention['unit'] == 'control']['outcome'].mean()
post_treated_mean = post_intervention[post_intervention['unit'] == 'treated']['outcome'].mean()

# Plotting
plt.figure(figsize=(10, 6))

# Pre-intervention trends
plt.plot(pre_intervention['time'], pre_intervention[pre_intervention['unit'] == 'control']['outcome'], label='Control (Pre)', linestyle='--', marker='o')
plt.plot(pre_intervention['time'], pre_intervention[pre_intervention['unit'] == 'treated']['outcome'], label='Treated (Pre)', linestyle='--', marker='o')

# Post-intervention trends
plt.plot(post_intervention['time'], post_intervention[post_intervention['unit'] == 'control']['outcome'], label='Control (Post)', linestyle='-', marker='o')
plt.plot(post_intervention['time'], post_intervention[post_intervention['unit'] == 'treated']['outcome'], label='Treated (Post)', linestyle='-', marker='o')

# Adding vertical line to indicate intervention point
plt.axvline(x=5, color='red', linestyle='--', label='Intervention')

# Labels and legend
plt.xlabel('Time')
plt.ylabel('Outcome')
plt.title('Segmented Trends Pre and Post Intervention')
plt.legend()
plt.grid(True)
plt.show()