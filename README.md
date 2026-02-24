# Customer Segmentation (Clustering)
## Overview
This project demonstrates customer segmentation using **K-Means clustering**, a popular unsupervised machine learning technique. The goal is to group customers into meaningful segments based on demographic and behavioral features.

The workflow includes data preprocessing, optimal cluster selection, model training, and PCA-based visualization.

## Features
✔ Handles missing data
✔ Scales numeric features
✔ Encodes categorical variables
✔ Selects optimal clusters using Silhouette Score
✔ Visualizes clusters using PCA

## Dataset
The dataset (`customers.csv`) contains customer attributes:

* **Age** – Customer age
* **Income** – Annual income
* **SpendingScore** – Spending behavior indicator
* **Gender** – Customer gender
* **Region** – Customer location

## Technologies Used
* Python
* Pandas
* Scikit-learn
* Matplotlib

## Workflow

### 1. Data Preprocessing
* Missing values handled using `SimpleImputer`
* Numeric features scaled with `StandardScaler`
* Categorical features encoded with `OneHotEncoder`

### 2. Cluster Selection
* Tested cluster sizes from **2 to 7**
* Optimal number chosen via **Silhouette Score**

### 3. Model Training
* K-Means clustering applied
* Cluster labels added to dataset

### 4. Visualization
* PCA reduces dimensions for plotting
* Scatter plot displays customer segments
* 
## Results
The model automatically determines the best number of clusters and generates distinct customer groups, enabling insights into customer behavior patterns.

## Visualization Example
Clusters are visualized using PCA:
```
Customer Segments (PCA Projection)
```
Each color represents a different customer segment.

## How to Run

### 1. Clone Repository
```bash
git clone https://github.com/prvn18/customer-segmentation.git
cd customer-segmentation
```
### 2. Install Dependencies
```bash
pip install pandas scikit-learn matplotlib
```
### 3. Run Script

```bash
python customer_segmentation.py
```
## Project Structure
```
customer-segmentation/
│── customers.csv
│── customer_segmentation.py
│── README.md
```

## Future Improvements
* Try other clustering algorithms (DBSCAN, Hierarchical)
* Add interactive visualization
* Deploy as a web dashboard
* Perform deeper cluster analysis

## Purpose

This project is designed for learning and demonstrating **unsupervised machine learning**, **data preprocessing pipelines**, and **dimensionality reduction techniques**.
