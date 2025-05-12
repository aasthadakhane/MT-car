import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# Page config
st.set_page_config(page_title="MTCARS Clustering", layout="centered")

st.title("🚗 MTCARS KMeans Clustering App")

# Load dataset
df = pd.read_csv("MTCARS.csv")

# Rename column
df = df.rename(columns={'Unnamed: 0': 'name'})

# Label encode car names
label = LabelEncoder()
df['name_encoded'] = label.fit_transform(df['name'])

# Drop original name for clustering
df_cluster = df.drop(columns=['name'])

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(df_cluster)

# Sidebar - number of clusters
k = st.sidebar.slider("Select number of clusters (K)", 1, 10, 3)

# KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Display clustered data
st.subheader("Clustered Data")
st.dataframe(df)

# Elbow method plot
st.subheader("Elbow Method (WSS vs K)")
wss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X)
    wss.append(km.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), wss, marker='o')
ax.set_title("Elbow Method")
ax.set_xlabel("Number of clusters")
ax.set_ylabel("Within-cluster Sum of Squares (WSS)")
st.pyplot(fig)
