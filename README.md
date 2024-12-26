# Hotel Reviews Clustering Project

This project aims to cluster hotel reviews based on sentiment and key features using KMeans clustering and Principal Component Analysis (PCA). The objective is to identify distinct groups of hotel reviews, enabling insights into customer feedback and improving hotel services.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Code Explanation](#code-explanation)
- [Results](#results)
- [Findings and Insights](#findings-and-insights)
- [Evaluation Metrics](#evaluation-metrics)
- [Challenges and Future Work](#challenges-and-future-work)
- [How to Run](#how-to-run)
- [Contributors](#contributors)

## Overview
This project performs clustering on a dataset of hotel reviews. By applying PCA for dimensionality reduction and KMeans for clustering, we segment the reviews into meaningful clusters. These clusters reflect different types of customer feedback, providing valuable insights for hotel management.

## Requirements
- Python 3.x
- pandas
- matplotlib
- seaborn
- sklearn
- numpy

## Dataset
The dataset consists of hotel reviews collected from various platforms. Each review is associated with different features such as review text, sentiment score, and customer rating.

## Code Explanation
1. **Data Loading:**
   ```python
   df = pd.read_csv('hotel_reviews.csv')
   ```
   - Loads the hotel review data into a dataframe.

2. **Feature Scaling:**
   ```python
   scaler = StandardScaler()
   df_scaled = scaler.fit_transform(df[['review_score', 'sentiment_score']])
   ```
   - Standardizes the review score and sentiment score for uniformity.

3. **PCA:**
   ```python
   pca = PCA(n_components=2)
   principal_components = pca.fit_transform(df_scaled)
   ```
   - Reduces the data to two principal components.

4. **Clustering:**
   ```python
   kmeans = KMeans(n_clusters=3, random_state=42)
   df['cluster'] = kmeans.fit_predict(principal_components)
   ```
   - Applies KMeans clustering with 3 clusters.

5. **Visualization:**
   ```python
   plt.scatter(principal_components[:, 0], principal_components[:, 1], c=df['cluster'], cmap='viridis')
   plt.title('Hotel Reviews Clustering')
   plt.xlabel('Principal Component 1')
   plt.ylabel('Principal Component 2')
   plt.colorbar()
   plt.show()
   ```
   - Visualizes the clustered data with color-coded clusters.

## Results
- The plot displays three distinct clusters of hotel reviews.
- Each cluster represents reviews with similar patterns and characteristics.

## Findings and Insights
### Findings
- **Cluster 0 (Purple):** Represents highly positive reviews with high sentiment scores and ratings.
- **Cluster 1 (Yellow):** Reflects mixed reviews, indicating customers with moderate satisfaction.
- **Cluster 2 (Green):** Indicates negative reviews with low sentiment scores and lower ratings.

### Insights
- The clustering helps identify areas for improvement by analyzing negative clusters.
- Positive clusters highlight strengths that can be emphasized in marketing campaigns.
- Mixed reviews can guide targeted interventions to enhance customer experiences.
- Clustering reviews provide actionable insights that can directly impact service quality and customer satisfaction.

## Evaluation Metrics
- **Inertia (Sum of Squared Distances):** Measures how tight the clusters are. Lower inertia indicates better-defined clusters.
- **Silhouette Score:** Assesses how similar data points are within a cluster compared to other clusters. Higher scores indicate better clustering.

## Challenges and Future Work
- **Challenges:**
  - Determining the optimal number of clusters.
  - Addressing outliers that may skew clustering results.
  - Balancing computational cost with accuracy for larger datasets.

- **Future Work:**
  - Experimenting with different clustering algorithms (e.g., DBSCAN, Hierarchical Clustering).
  - Incorporating additional features like review length, keywords, and timestamps.
  - Enhancing visualization with interactive dashboards.
  
## How to Run
1. Clone this repository.
2. Install required packages:
   ```bash
   pip install pandas matplotlib seaborn sklearn numpy
   ```
3. Run the Python script:
   ```bash
   python hotel_clustering.py
   ```

## Contributors
- Dimas Dzaki Adani 202110370311003
- Nadira Furqani 202110370311019
- Muhammad Abdi Harliansyah 202110370311042

