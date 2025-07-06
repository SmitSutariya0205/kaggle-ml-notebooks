# 📚 Machine Learning Projects by **Smit Sutariya**

Welcome to my personal collection of ML notebooks. Each notebook is **self-contained**: load the data, follow the preprocessing pipeline, train the model(s), and review the results/visualisations.

---

## 📂 Repository Layout

| 📒 Notebook                                          | Task                                           | Main Techniques                                                     |
| ---------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------- |
| `svm_sms_spam_classifier.ipynb`                      | Text classification (spam vs ham)              | TF‑IDF + SVM                                                        |
| `knn-breast-cancer.ipynb`                            | Medical diagnosis                              | K‑Nearest Neighbours                                                |
| `knn-adult-income.ipynb`                             | Tabular income prediction                      | KNN                                                                 |
| `creditrisk-random-forest.ipynb`                     | Credit‑risk scoring                            | Random Forest                                                       |
| `kmeans-mall-customer-segmentation.ipynb`            | Customer segmentation                          | K‑Means                                                             |
| `kmeans-country-clustering.ipynb`                    | Macro‑economic clustering                      | K‑Means                                                             |
| `mall_customer_clustering_kmeans_hierarchical.ipynb` | Customer segmentation (Hierarchical + K‑Means) | Agglomerative Clustering                                            |
| `concrete_strength_rf_vs_linear.ipynb`               | Regression – concrete compressive strength     | Linear Regression, Ridge, Random Forest, Gradient Boosting, PCA     |
| `home_credit_default_modeling.ipynb`                 | Default prediction (Kaggle competition)        | Feature Engineering, Random Forest, Stratified K-Fold, Manual Preprocessing |

> **Tip:** Clone the repo, open any notebook, run all cells – no extra wiring needed.

---

## 🗃️ Datasets

All data are public and hosted on **Kaggle** unless noted.

| Notebook          | Dataset                       | Link                                                                                                                                                               |
| ----------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| SMS Spam          | SMS Spam Collection           | [SMS Spam Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)                                                                                       |
| Breast Cancer     | Wisconsin Diagnostic          | [Breast Cancer Dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)                                                                                 |
| Adult Income      | Adult Census Income           | [Adult Income Dataset](https://www.kaggle.com/uciml/adult-census-income)                                                                                           |
| Credit Risk       | Loan Dataset                  | [Loan Dataset](https://www.kaggle.com/datasets/zaurbegiev/my-dataset)                                                                                              |
| Mall Customers    | Mall Customer Segmentation    | [Mall Customers](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)                                                                    |
| Country Data      | Country-level Indicators      | [Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data)                                                                   |
| Concrete Strength | Concrete Compressive Strength | [Concrete Dataset](https://www.kaggle.com/datasets/ujjwalchowdhury/concrete-compressive-strength)                                                                 |
| Home Credit       | Default Risk Prediction       | [Home Credit Dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data)                                                                          |

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
$ git clone https://github.com/<your‑username>/ml-models-smit.git
$ cd ml-models-smit

# 2. (Recommended) create a virtual environment
$ python -m venv .venv
$ source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Launch Jupyter Lab / VS Code and run the desired notebook 🎉
> **Note:** Raw datasets are **not** stored in the repo (to keep it lightweight). Each notebook auto-downloads or links to the source on Kaggle.

---

## 📊 Skills & Techniques Covered

### 🧠 Supervised Learning
- **Classification**: SVM, KNN, Random Forest  
- **Regression**: Linear Regression, Ridge, Gradient Boosting, Random Forest Regressor

### 📈 Unsupervised Learning
- K-Means  
- Hierarchical (Agglomerative) Clustering  
- PCA (Dimensionality Reduction)

### 🛠️ Feature Engineering
- Group-based Aggregation  
- Manual Preprocessing & Scaling  
- One-Hot Encoding  
- Handling Missing Data

### 📋 Model Evaluation
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- **Regression**: R², Adjusted R², MAE, RMSE, MAPE  
- **Clustering**: Silhouette Score, Elbow Method, Dendrograms

### 🔍 Model Tuning
- GridSearchCV  
- RandomizedSearchCV

### 📊 Visualization
- Matplotlib  
- Seaborn  
- Feature Importance  
- PCA Variance Plots  
- Correlation Heatmaps

---

## 👤 Author

**Smit Sutariya**  
B.Tech Computer Science (MIT‑WPU)  
AI/ML Enthusiast  

- 🌐 [LinkedIn](https://www.linkedin.com/in/smitsutariya)  
- 📫 [ssutariya8801@gmail.com](mailto:ssutariya8801@gmail.com)
