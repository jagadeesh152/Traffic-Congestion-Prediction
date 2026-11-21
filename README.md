\# 🚦 TrafficLens – Traffic Congestion Prediction



\*\*Predicting Urban Traffic Congestion Using PySpark and Machine Learning\*\*



\## 📌 About the Project



TrafficLens is a machine learning project that analyzes traffic data and predicts congestion levels using Apache Spark and PySpark MLlib.



The project implements and compares two classification algorithms:



\- Logistic Regression

\- Random Forest



The system classifies traffic congestion into three categories: Low, Medium and High.



\## 🎯 Project Objectives



\- Analyze urban traffic data.

\- Perform data preprocessing using PySpark.

\- Convert categorical values into numerical representations.

\- Apply feature engineering.

\- Train Logistic Regression and Random Forest models.

\- Evaluate and compare model performance.



\## 📊 Dataset Information



| Details | Description |

|---|---|

| Dataset | Traffic\_dataset.csv |

| Total Records | 1,200 |

| Original Columns | 9 |

| Cities | 5 |

| Areas | 25 |

| Location | Maharashtra, India |

| Target Column | Congestion\_Level |



\### Dataset Columns



1\. Date

2\. City

3\. Area

4\. Vehicle\_Type

5\. Vehicle\_Count

6\. Accidents

7\. Traffic\_Violations

8\. Avg\_Speed\_kmph

9\. Congestion\_Level



\## 🛠️ Technologies Used



\- Python

\- Apache Spark

\- PySpark

\- PySpark MLlib

\- NumPy

\- Pandas

\- Matplotlib

\- Seaborn



\## ⚙️ Project Workflow



1\. Dataset loading

2\. Data exploration

3\. Data preprocessing

4\. Feature engineering

5\. StringIndexer

6\. VectorAssembler

7\. Train-test splitting

8\. Model training

9\. Model evaluation

10\. Result visualization



\## 🤖 Machine Learning Algorithms



\### 1. Logistic Regression



A supervised classification algorithm that predicts the probability of an observation belonging to a particular congestion category.



\### 2. Random Forest



An ensemble learning algorithm that combines multiple decision trees to classify traffic congestion.



\## 📈 Model Performance



| Algorithm | Accuracy |

|---|---:|

| Logistic Regression | 98.72% |

| Random Forest (20 Trees) | 99.68% |



Random Forest achieved higher accuracy in the reported experiment.



\*\*Note:\*\* These results are from the project's test split. The congestion labels are closely related to vehicle-count ranges, so further testing with independent real-world traffic data is needed to establish generalization.



\## 🔍 Key Findings



\- Random Forest achieved higher accuracy than Logistic Regression.

\- Vehicle\_Count was the most important feature in the Random Forest model.

\- Traffic congestion was classified into Low, Medium and High categories.



\## 🚀 Future Enhancements



\- Real-time traffic data integration.

\- Interactive traffic prediction dashboard.

\- Integration with live traffic APIs.

\- Larger and more diverse traffic datasets.

\- Model deployment using Streamlit.



\## 📂 Project Structure



```text

TrafficLens/

│

├── dataset/

│   └── Traffic\_dataset.csv

│

├── src/

│   └── traffic\_prediction.py

│

├── results/

│

├── README.md

├── requirements.txt

└── .gitignore

```



\## 💻 Installation and Execution



\### Step 1: Clone the repository



```bash

git clone https://github.com/jagadeesh152/Traffic-Congestion-Prediction.git

```



\### Step 2: Open the project directory



```bash

cd Traffic-Congestion-Prediction

```



\### Step 3: Install dependencies



```bash

pip install -r requirements.txt

```



\### Step 4: Run the project



```bash

python src/traffic\_prediction.py

```



\*\*Prerequisite:\*\* Java and a compatible Apache Spark/PySpark environment must be installed and configured.



\## 👨‍💻 Author



\*\*Uppalapati Jagadeesh\*\*



B.Tech – Computer Science and Engineering



GitHub: \[jagadeesh152](https://github.com/jagadeesh152)





