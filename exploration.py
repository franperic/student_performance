import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/train.csv")
df.head()

df.shape
df.session_id.nunique()

RANDOM_SESSION = df.session_id.sample(2).values.tolist()
sample_df = df.loc[df.session_id.isin(RANDOM_SESSION)]

inspect = ["room_coor_x", "room_coor_y", "screen_coor_x", "screen_coor_y"]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

# Flatten the axes array to make it easier to iterate
axes = axes.flatten()

groups = sample_df.groupby("session_id")
# Iterate over the columns and plot data on subplots
for i, col in enumerate(inspect):
    for j, (group, data) in enumerate(groups):
        axes[i].plot(range(len(data)), data[col], label=f"Session {group}")
    axes[i].set_title(col)

    axes[i].legend()

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()
