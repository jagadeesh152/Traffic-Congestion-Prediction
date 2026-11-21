# ================== LIBRARIES ==================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, to_date, month, dayofweek
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ================== 1. LOAD DATA ==================
spark = SparkSession.builder.getOrCreate()
data = spark.read.csv("dataset/Traffic_dataset.csv", header=True, inferSchema=True)
data.show(5)
data.printSchema()
print("Total records:", data.count())
print("Total columns:", len(data.columns))
print(data.columns)

# ================== 2. DATA INSPECTION ==================
data.select([sum(col(c).isNull().cast("int")).alias(c) for c in data.columns]).show()
print("Total records:", data.count())
print("Records after removing duplicates:", data.dropDuplicates().count())
data.describe().show()
data.select("Vehicle_Count", "Accidents", "Traffic_Violations", "Avg_Speed_kmph").describe().show()

# ================== 3. CATEGORICAL DISTRIBUTION ==================
data.groupBy("City").count().show()
data.groupBy("Vehicle_Type").count().show()
data.groupBy("Congestion_Level").count().show()
print("Distinct areas:", data.select("Area").distinct().count())

# ================== 4. VALIDATION AND DATE FEATURES ==================
print("Vehicle_Count <= 0 :", data.filter(col("Vehicle_Count") <= 0).count())
print("Avg_Speed_kmph <= 0:", data.filter(col("Avg_Speed_kmph") <= 0).count())
print("Negative accidents :", data.filter(col("Accidents") < 0).count())
print("Negative violations:", data.filter(col("Traffic_Violations") < 0).count())
data = data.filter((col("Vehicle_Count") > 0) & (col("Avg_Speed_kmph") > 0) &
                   (col("Accidents") >= 0) & (col("Traffic_Violations") >= 0))
print("Records after validity filtering:", data.count())

data = data.withColumn("Date", to_date(col("Date"), "dd-MM-yyyy"))
data = data.withColumn("Month", month(col("Date"))).withColumn("Day_of_Week", dayofweek(col("Date")))
data.select("Date", "Month", "Day_of_Week", "Congestion_Level").show(5)

# ================== 5. DATA VISUALIZATION ==================
level_data = (data.groupBy("Congestion_Level").avg("Vehicle_Count").toPandas()
              .set_index("Congestion_Level").loc[["Low", "Medium", "High"]].reset_index())
plt.bar(level_data["Congestion_Level"], level_data["avg(Vehicle_Count)"], color=["green", "orange", "red"])
plt.xlabel("Congestion Level")
plt.ylabel("Average Vehicle Count")
plt.title("Average Vehicle Count by Congestion Level")
plt.show()

count_data = data.select("Vehicle_Count").toPandas()
plt.hist(count_data["Vehicle_Count"], bins=30, edgecolor="black")
plt.xlabel("Vehicle Count")
plt.ylabel("Frequency")
plt.title("Vehicle Count Distribution")
plt.show()

speed_data = data.select("Vehicle_Count", "Avg_Speed_kmph", "Congestion_Level").toPandas()
for level, colour in [("Low", "green"), ("Medium", "orange"), ("High", "red")]:
    part = speed_data[speed_data["Congestion_Level"] == level]
    plt.scatter(part["Vehicle_Count"], part["Avg_Speed_kmph"], alpha=0.4, color=colour, label=level)
plt.xlabel("Vehicle Count")
plt.ylabel("Average Speed (km/h)")
plt.title("Vehicle Count vs Average Speed")
plt.legend()
plt.show()

corr_data = data.select("Vehicle_Count", "Accidents", "Traffic_Violations", "Avg_Speed_kmph").toPandas()
correlation = corr_data.corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
print(correlation.round(3).to_string())

city_data = data.groupBy("City", "Congestion_Level").count().toPandas()
city_pivot = city_data.pivot(index="City", columns="Congestion_Level", values="count")[["Low", "Medium", "High"]]
city_pivot.plot(kind="bar", color=["green", "orange", "red"])
plt.xlabel("City")
plt.ylabel("Number of Records")
plt.title("Congestion Level by City")
plt.xticks(rotation=45)
plt.show()

# ================== 6. STRING INDEXING AND FEATURE ASSEMBLY ==================
indexer1 = StringIndexer(inputCol="City", outputCol="City_Index")
indexer2 = StringIndexer(inputCol="Area", outputCol="Area_Index")
indexer3 = StringIndexer(inputCol="Vehicle_Type", outputCol="Vehicle_Type_Index")
label_indexer = StringIndexer(inputCol="Congestion_Level", outputCol="label")
data = indexer1.fit(data).transform(data)
data = indexer2.fit(data).transform(data)
data = indexer3.fit(data).transform(data)
label_model = label_indexer.fit(data)
data = label_model.transform(data)
print("Label mapping:", list(enumerate(label_model.labels)))

feature_columns = ["City_Index", "Area_Index", "Vehicle_Type_Index", "Vehicle_Count",
                   "Accidents", "Traffic_Violations", "Avg_Speed_kmph", "Month", "Day_of_Week"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data_2 = assembler.transform(data)
data_2.select("features", "label").show(5, truncate=False)

# ================== 7. TRAIN / TEST SPLIT ==================
train, test = data_2.randomSplit([0.7, 0.3], seed=42)
print("Training records:", train.count())
print("Testing records:", test.count())

# ================== 8. LOGISTIC REGRESSION ==================
algo = LogisticRegression(featuresCol="features", labelCol="label")
model = algo.fit(train)
predictions = model.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
print("Logistic Regression Accuracy:", evaluator.evaluate(predictions))
evaluator.setMetricName("f1")
print("Logistic Regression F1 Score:", evaluator.evaluate(predictions))
evaluator.setMetricName("weightedPrecision")
print("Logistic Regression Precision:", evaluator.evaluate(predictions))
evaluator.setMetricName("weightedRecall")
print("Logistic Regression Recall:", evaluator.evaluate(predictions))
predictions.select("Congestion_Level", "label", "prediction").show(10)

# ================== 9. RANDOM FOREST (20 TREES) ==================
algo2 = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20, seed=42)
model2 = algo2.fit(train)
predictions2 = model2.transform(test)
evaluator.setMetricName("accuracy")
print("Random Forest Accuracy:", evaluator.evaluate(predictions2))
evaluator.setMetricName("f1")
print("Random Forest F1 Score:", evaluator.evaluate(predictions2))
evaluator.setMetricName("weightedPrecision")
print("Random Forest Precision:", evaluator.evaluate(predictions2))
evaluator.setMetricName("weightedRecall")
print("Random Forest Recall:", evaluator.evaluate(predictions2))
predictions2.select("Congestion_Level", "label", "prediction").show(10)

# ================== 10. ONE-HOT ENCODING EXPERIMENT ==================
encoder = OneHotEncoder(inputCols=["City_Index", "Area_Index", "Vehicle_Type_Index"],
                        outputCols=["City_Vector", "Area_Vector", "Vehicle_Type_Vector"])
data_3 = encoder.fit(data).transform(data)
data_3.select("City_Vector", "Area_Vector", "Vehicle_Type_Vector").show(5, truncate=False)

feature_columns = ["City_Vector", "Area_Vector", "Vehicle_Type_Vector", "Vehicle_Count",
                   "Accidents", "Traffic_Violations", "Avg_Speed_kmph", "Month", "Day_of_Week"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data_3 = assembler.transform(data_3)
data_3.select("features", "label").show(5, truncate=False)

train3, test3 = data_3.randomSplit([0.7, 0.3], seed=42)
print("Training records:", train3.count())
print("Testing records:", test3.count())

algo3 = LogisticRegression(featuresCol="features", labelCol="label", regParam=0.01)
model3 = algo3.fit(train3)
predictions3 = model3.transform(test3)
algo4 = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50, maxDepth=10, seed=42)
model4 = algo4.fit(train3)
predictions4 = model4.transform(test3)

# ================== 11. MODEL EVALUATION ==================
for name, pred in [("OneHot Logistic Regression", predictions3),
                   ("Random Forest (50 trees, depth 10)", predictions4)]:
    for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
        evaluator.setMetricName(metric)
        print(name, metric + ":", round(evaluator.evaluate(pred), 4))

cm = (predictions.groupBy("label", "prediction").count().toPandas()
      .pivot(index="label", columns="prediction", values="count").fillna(0).astype(int))
print("Logistic Regression confusion matrix (rows = actual, columns = predicted)")
print(cm)
cm2 = (predictions2.groupBy("label", "prediction").count().toPandas()
       .pivot(index="label", columns="prediction", values="count").fillna(0).astype(int))
print("Random Forest (20 trees) confusion matrix (rows = actual, columns = predicted)")
print(cm2)

importance = model2.featureImportances.toArray()
feature_names = ["City_Index", "Area_Index", "Vehicle_Type_Index", "Vehicle_Count",
                 "Accidents", "Traffic_Violations", "Avg_Speed_kmph", "Month", "Day_of_Week"]
for name, value in sorted(zip(feature_names, importance), key=lambda x: -x[1]):
    print(f"{name:20s} {value:.4f}")

models_info = {
    "Logistic Regression": predictions,
    "Random Forest (20 trees)": predictions2,
    "Logistic Regression (One-hot, regParam 0.01)": predictions3,
    "Random Forest (50 trees, depth 10)": predictions4,
}
results = {}
for name, pred in models_info.items():
    results[name] = {}
    for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
        evaluator.setMetricName(metric)
        results[name][metric] = round(evaluator.evaluate(pred), 4)
print(pd.DataFrame(results).T.round(4).to_string())

# ================== 12. RESULT GRAPHS ==================
labels = label_model.labels
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (title, pred) in zip(axes, [("Logistic Regression", predictions),
                                    ("Random Forest (50 trees, depth 10)", predictions4)]):
    matrix = (pred.groupBy("label", "prediction").count().toPandas()
              .pivot(index="label", columns="prediction", values="count")
              .reindex(index=range(3), columns=range(3)).fillna(0).astype(int))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
plt.tight_layout()
plt.show()

imp = sorted(zip(feature_names, importance), key=lambda x: x[1])
plt.barh([i[0] for i in imp], [i[1] for i in imp])
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

names = list(results.keys())
short = ["LR", "RF-20", "LR One-hot", "RF-50"]

plt.bar(range(len(names)), [results[n]["accuracy"] for n in names], color=["blue", "green", "blue", "green"])
plt.xticks(range(len(names)), short)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Comparison - Accuracy")
plt.show()

plt.bar(range(len(names)), [results[n]["f1"] for n in names], color=["blue", "green", "blue", "green"])
plt.xticks(range(len(names)), short)
plt.ylim(0, 1)
plt.ylabel("F1 Score")
plt.title("Model Comparison - F1 Score")
plt.show()

x = np.arange(len(names))
plt.bar(x - 0.2, [results[n]["weightedPrecision"] for n in names], width=0.4, label="Precision")
plt.bar(x + 0.2, [results[n]["weightedRecall"] for n in names], width=0.4, label="Recall")
plt.xticks(x, short)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Comparison - Precision and Recall")
plt.legend()
plt.show()