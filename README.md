# 📚 Machine‑Learning Projects by **Smit Sutariya**

Welcome to my personal collection of ML notebooks. Each notebook is **self‑contained**: load the data, follow the preprocessing pipeline, train the model(s), and review the results/visualisations.

---

## 📂 Repository Layout

| 📒 Notebook                                          | Task                                           | Main Techniques                                                     |
| ---------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------- |
| svm_sms_spam_classifier.ipynb                      | Text classification (spam vs ham)              | TF‑IDF + SVM                                                        |
| knn-breast-cancer.ipynb                            | Medical diagnosis                              | K‑Nearest Neighbours                                                |
| knn-adult-income.ipynb                             | Tabular income prediction                      | KNN                                                                 |
| creditrisk-random-forest.ipynb                     | Credit‑risk scoring                            | Random Forest                                                       |
| kmeans-mall-customer-segmentation.ipynb            | Customer segmentation                          | K‑Means                                                             |
| kmeans-country-clustering.ipynb                    | Macro‑economic clustering                      | K‑Means                                                             |
| mall_customer_clustering_kmeans_hierarchical.ipynb | Customer segmentation (Hierarchical + K‑Means) | Agglomerative Clustering                                            |
| concrete_strength_rf_vs_linear.ipynb               | **Regression** – concrete compressive strength | Linear Regression, Ridge, Random Forest, Gradient Boosting, **PCA** |

> **Tip:** Clone the repo, open any notebook, run all cells – no extra wiring needed.

---

## 🗃️ Datasets

All data are public and hosted on **Kaggle** unless noted.

| Notebook          | Dataset                       | Link                                                                                                                                                               |
| ----------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| SMS Spam          | SMS Spam Collection           | [https://www.kaggle.com/uciml/sms-spam-collection-dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)                                               |
| Breast‑Cancer     | Wisconsin Diagnostic          | [https://www.kaggle.com/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)                                             |
| Adult Income      | Adult Census Income           | [https://www.kaggle.com/uciml/adult-census-income](https://www.kaggle.com/uciml/adult-census-income)                                                               |
| Credit Risk       | Loan Dataset                  | [https://www.kaggle.com/datasets/zaurbegiev/my-dataset](https://www.kaggle.com/datasets/zaurbegiev/my-dataset)                                                     |
| Mall Customers    | Mall Customer Segmentation    | [https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)       |
| Country Data      | Country‑level Indicators      | [https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data) |
| Concrete Strength | Concrete Compressive Strength | [https://www.kaggle.com/datasets/ujjwalchowdhury/concrete-compressive-strength](https://www.kaggle.com/datasets/ujjwalchowdhury/concrete-compressive-strength)     |

---

## 🚀 Getting Started

bash
# 1. Clone the repo
$ git clone https://github.com/<your‑user>/ml‑models‑smit.git
$ cd ml‑models‑smit

# 2. (Recommended) create a virtual environment
$ python -m venv .venv
$ source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Launch Jupyter Lab / VS Code and run the desired notebook 🎉


> **Note:** Raw datasets are **not** stored in the repo (to keep it light). Each notebook auto‑downloads the data or provides a Kaggle link.

---

## 📊 Skills & Techniques Covered

* **Supervised Learning**

  * Classification: SVM, KNN, Random Forest
  * Regression: Linear & Ridge Regression, Gradient Boosting, Random Forest Regressor
* **Unsupervised Learning**

  * K‑Means, Agglomerative/Hierarchical Clustering, PCA for dimensionality reduction
* **Model Evaluation**

  * Accuracy, Precision, Recall, F1‑Score (classification)
  * R², Adjusted R², MAE, RMSE, MAPE (regression)
  * Silhouette Score, Elbow Method, Dendrograms (clustering)
* **Data Processing**

  * Cleaning, Feature Scaling (StandardScaler), Text Vectorisation (TF‑IDF)
* **Model Selection & Tuning**

  * GridSearchCV & RandomizedSearchCV for hyper‑parameter optimisation
* **Visualisation**

  * Matplotlib & Seaborn plots, feature‑importance charts, PCA variance plots

---

## 👤 Author

**Smit Sutariya**
B.Tech Computer Science (MIT‑WPU)
AI/ML Enthusiast

* 🌐 LinkedIn: [https://www.linkedin.com/in/smitsutariya](https://www.linkedin.com/in/smitsutariya)
* 📫 Email: [ssutariya8801@gmail.com](mailto:ssutariya8801@gmail.com)

---

*If you find these notebooks helpful, give the repo a ⭐ and feel free to open issues or pull requests!*  
