import pandas as pd
from pathlib import Path

# Fils in Folders
path = Path('C:/Users/Admin/OneDrive/Cumulative Test/DataSets')

# List all CSV files in the directory
csv_files = path.glob('*.csv')

# Read and concatenate all CSV files into a single DataFrame
merged_df = pd.concat([pd.read_csv(file, encoding='ISO-8859-1') for file in csv_files], ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_data.csv', index=False)

print("CSV files have been successfully merged and saved to 'merged_data.csv'")


# EDA

print(merged_df.shape)

print(merged_df.columns)

merged_df.head(3)

merged_df.describe()

merged_df.isnull().sum()


merged_df.columns = merged_df.columns.str.strip().str.lower().str.replace(" ", "-").str.replace("-", "_")


# Feature Engineering


#### -Total workers, -Gender ratio, -Rural/urban ratios


merged_df["total_workers"] = merged_df["main_workers___total____persons"] + merged_df["marginal_workers___total____persons"]

merged_df["total_workers"] = merged_df["main_workers___total____persons"] + merged_df["marginal_workers___total____persons"]

merged_df[['main_workers___total____persons','marginal_workers___total____persons','total_workers']].head(3)

merged_df["female_to_male_ratio"] = merged_df['main_workers___total___females']/merged_df['main_workers___total___males']

merged_df[['main_workers___total___females','main_workers___total___males','female_to_male_ratio']].head(3)

merged_df["rural_ratio"] = merged_df['main_workers___rural____persons']/merged_df['main_workers___total____persons']

merged_df[['main_workers___rural____persons','main_workers___total____persons','rural_ratio']].head(3)

merged_df["urban_ratio"] = merged_df['main_workers___urban____persons'] / merged_df["main_workers___total____persons"]
merged_df[['main_workers___urban____persons','main_workers___total____persons','urban_ratio']].head(3)



# Grouping and Summarizing

##  State-wise total workers, Industry division summary

state_group = merged_df.groupby('india/states')[['total_workers', 'main_workers___total____persons']].sum().sort_values("total_workers", ascending=False)
state_group


division_group = merged_df.groupby('division')[['main_workers___total____persons','marginal_workers___total____persons']].sum()
division_group



# Geo-Visualization Preparation
####  Save aggregated data for mapping


state_summary = merged_df.groupby("india/states").agg({
    'main_workers___total____persons':'sum',
    'marginal_workers___total____persons': 'sum'
}).reset_index()
    

state_summary.columns = ["state", "main_workers", "marginal_workers"]
state_summary.to_csv("state_workforce_summary.csv", index=False)


# NLP for Industry Grouping

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# Example: Cluster industries based on NIC Name
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(merged_df['nic_name'])

kmeans = KMeans(n_clusters=10, random_state=42)
merged_df['industry_cluster'] = kmeans.fit_predict(X)


# Visualization

import seaborn as sns
import matplotlib.pyplot as plt

top_states = state_group.head(10).reset_index()
sns.barplot(x="india/states", y="total_workers", data=top_states)
plt.xticks(rotation=45)
plt.title("Top 10 States by Total Workers")
plt.show()


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from shapely.geometry import Point

# === 1. Load the GeoJSON (India State Boundaries) ===
map_df = gpd.read_file("gadm41_IND_1.json")
map_df = map_df.rename(columns={"NAME_1": "state"})

# === 2. Your State-wise Data ===
# Replace this with your actual data
state_summary = pd.DataFrame({
    "state": ["Maharashtra", "Uttar Pradesh", "AndhraPradesh", "Karnataka", "Gujarat", "Bihar"],
    "main_workers": [5000000, 6000000, 4500000, 3000000, 3500000, 4000000]
})

# === 3. Merge GeoDataFrame with Data ===
merged_map_df = map_df.merge(state_summary, on="state", how="left")

# === 4. Assign a Unique Color to Each State ===
# Use colormap with enough variety
num_states = len(merged_map_df)
colors = plt.cm.get_cmap('tab20b', num_states)
merged_map_df['color'] = [colors(i % 20) for i in range(num_states)]

# === 5. Create the Plot ===
fig, ax = plt.subplots(figsize=(16, 12))

# Plot each state with its assigned color
merged_map_df.plot(
    color=merged_map_df['color'],
    edgecolor='black',
    linewidth=0.6,
    ax=ax
)

# === 6. Add Worker Count Labels to Each State ===
for idx, row in merged_map_df.iterrows():
    if pd.notnull(row['main_workers']):
        # Place text at the center of the polygon
        centroid = row['geometry'].centroid
        ax.text(
            centroid.x,
            centroid.y,
            f"{int(row['main_workers']):,}",  # format with commas
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=9,
            fontweight='bold',
            color='white',
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2')
        )

# === 7. Final Touches ===
ax.set_title("Main Workers by State in India", fontsize=18, weight='bold')
ax.axis("off")
plt.tight_layout()
plt.show()
