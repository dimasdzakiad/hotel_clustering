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
The dataset consists of hotel reviews collected from various platforms. Each review is associated with different features such as hotel, is_canceled, lead_time, arrival_date_year, arrival_date_month, arrival_date_week_number, arrival_date_day_of_month, stays_in_weekend_nights, stays_in_week_nights, adults, children, babies, meal, country, market_segment, distribution_channel, is_repeated_guest, previous_cancellations, previous_bookings_not_canceled, reserved_room_type, assigned_room_type, booking_changes, deposit_type, agent, company, days_in_waiting_list, customer_type, adr, required_car_parking_spaces, total_of_special_requests, reservation_status, reservation_status_date.

## Code Explanation
1. **Data Loading:**
   ```python
   import pandas as pd
   df = pd.read_csv('hotels.csv')
   df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')

   # Handling missing values
   df.fillna(df.median(), inplace=True)
   df.fillna(df.mode().iloc[0], inplace=True)
   df.drop_duplicates(inplace=True)
   ```
   - Loads the hotel review data into a dataframe.

2. **Feature Scaling:**
   ```python
   from sklearn.preprocessing import StandardScaler

   features = ['adr', 'total_of_special_requests', 'booking_changes', 'previous_cancellations', 'days_in_waiting_list']
   scaler = StandardScaler()
   df_scaled = scaler.fit_transform(df[features])
   ```
   - Standardizes the review score and sentiment score for uniformity.

3. **PCA:**
   ```python
   from sklearn.decomposition import PCA
   
   pca = PCA(n_components=2)
   principal_components = pca.fit_transform(df_scaled)
   ```
   - Reduces the data to two principal components.

4. **Clustering:**
   ```python
   from sklearn.cluster import KMeans

   kmeans = KMeans(n_clusters=3, random_state=42)
   df['cluster'] = kmeans.fit_predict(df_scaled)
   ```
   - Applies KMeans clustering with 3 clusters.

5. **Visualization:**
   ```python
   import matplotlib.pyplot as plt
   
   plt.scatter(principal_components[:, 0], principal_components[:, 1], c=df['cluster'], cmap='viridis')
   plt.title('Hotel Reviews Clustering')
   plt.xlabel('Principal Component 1')
   plt.ylabel('Principal Component 2')
   plt.colorbar()
   plt.show()
   ```
   - Visualizes the clustered data with color-coded clusters.

## Results
## Clustering Visualization
Here is the clustering visualization:

![Clustering Plot](./imag/clustering.png)

- The plot displays three distinct clusters of hotel reviews.
- Each cluster represents reviews with similar patterns and characteristics.

## Findings and Insights
### Findings
- **Cluster 0 (Purple):** Represents highly positive reviews with high sentiment scores and ratings.
- **Cluster 1 (Yellow):** Reflects mixed reviews, indicating customers with moderate satisfaction.
- **Cluster 2 (Green):** Indicates negative reviews with low sentiment scores and lower ratings.

### Segmentation by Customer Type:
#### Contract:
- **Average ADR:** 92.75
- **Total ADR Revenue:** 291,151.78
- **Repeat Customer Rate:** Very low (1.05%)
- **Average Special Requests:** 0.84
- **Previous Cancellations:** Relatively low (204)
- **Waiting List Time:** Almost zero (0.02)

#### Group:
- **Average ADR:** 84.36 (lowest among all segments)
- **Total ADR Revenue:** 45,892.90
- **Repeat Customer Rate:** Highest (28.68%)
- **Special Requests:** Low (0.65)
- **Waiting List Time:** Highest (0.34)

### Key Insights
The clustering results reveal valuable patterns:
1. **Customer Feedback Patterns:** Clustering hotel reviews based on sentiment scores identifies groups of customers with distinct feedback. Positive reviews dominate one cluster, indicating satisfaction, while mixed and negative reviews highlight areas for improvement.
2. **Revenue Analysis:** The **Contract** segment contributes significantly to revenue despite a low repeat customer rate. Its focus on long-term agreements makes it a key driver of steady income. Conversely, the **Group** segment shows higher repeat rates but generates less revenue per booking, suggesting potential for optimizing offers tailored to group dynamics.
3. **Customer Retention:** Repeat customers, although associated with lower ADR, provide consistent revenue. Enhancing marketing strategies for these customers could maximize loyalty and lifetime value.

### Further Analysis
- Investigating seasonal or event-based factors influencing segment behavior can uncover patterns to further refine marketing strategies.
- Focusing on high-revenue individual bookings while retaining group and contract balances can enhance overall profitability.

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
- Dimas Dzaki Adani         202110370311003
- Nadira Furqani            202110370311019
- Muhammad Abdi Harliansyah 202110370311042

