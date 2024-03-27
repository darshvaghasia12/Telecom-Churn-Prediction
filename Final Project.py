pip install imblearn


from pyspark.mllib.evaluation import BinaryClassificationMetrics
import matplotlib.pyplot as graphplt
from sklearn.metrics import roc_curve, auc
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import QuantileDiscretizer
import pandas as pd
from imblearn.over_sampling import ADASYN
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import DenseVector
from math import exp



dataFrame = spark.read.format("csv").option("inferSchema", "true").option("header", "true").option("sep", ',').option("nanValue", ' ').option("nullValue", ' ').load("dbfs:/FileStore/WA_Fn_UseC__Telco_Customer_Churn.csv")
display(dataFrame)


 temp_table = "churn_analysis"
 dataFrame.createOrReplaceTempView(temp_table)




 dataFrame.select('tenure', 'TotalCharges', 'MonthlyCharges').describe().show()


result = dataFrame.groupBy('gender', 'churn').agg(count("*").alias("count"))


result_pd = result.toPandas()


tablePivot = result_pd.pivot(index='gender', columns='churn', values='count')

axis1 = tablePivot.plot(kind='bar', stacked=True, figsize=(10, 6))
graphplt.title('Churn Count by Gender')
graphplt.xlabel('Gender')
graphplt.ylabel('Count')
graphplt.show()


result = dataFrame.groupBy('SeniorCitizen', 'churn').agg(count("*").alias("count"))


result_pd = result.toPandas()

tablePivot = result_pd.pivot(index='SeniorCitizen', columns='churn', values='count')


axis1 = tablePivot.plot(kind='bar', stacked=True, figsize=(18, 10))
graphplt.title('Churn Count by SeniorCitizen')
graphplt.xlabel('SeniorCitizen')
graphplt.ylabel('Count')
graphplt.show()


result = spark.sql("""
    SELECT CAST(tenure AS INT) AS tenure_int, churn, COUNT(churn) AS count
    FROM churn_analysis
    GROUP BY tenure, churn
    ORDER BY tenure_int
""")


result_pd = result.toPandas()


tablePivot = result_pd.pivot(index='tenure_int', columns='churn', values='count')


axis1 = tablePivot.plot(kind='line', stacked=True, figsize=(15, 6))
graphplt.title('Churn Count by Tenure')
graphplt.xlabel('Tenure')
graphplt.ylabel('Count')
graphplt.show()



dataFrame.stat.crosstab("SeniorCitizen", "InternetService").show()


result = spark.sql("""
    SELECT PaperlessBilling,
           COUNT(CASE WHEN churn = 'Yes' THEN 1 ELSE NULL END) AS churn_yes,
           COUNT(CASE WHEN churn = 'No' THEN 1 ELSE NULL END) AS churn_no
    FROM churn_analysis
    GROUP BY PaperlessBilling
""")

result_pd = result.toPandas()


result_pd.plot(x='PaperlessBilling', kind='bar', stacked=True, figsize=(18, 10))
graphplt.title('Churn Count by PaperlessBilling')
graphplt.xlabel('PaperlessBilling')
graphplt.ylabel('Count')
graphplt.show()


result = spark.sql("""
    SELECT PaymentMethod,
           COUNT(CASE WHEN churn = 'Yes' THEN 1 ELSE NULL END) AS churn_yes,
           COUNT(CASE WHEN churn = 'No' THEN 1 ELSE NULL END) AS churn_no
    FROM churn_analysis
    GROUP BY PaymentMethod
""")

result_pd = result.toPandas()


result_pd.plot(x='PaymentMethod', kind='line', stacked=True, figsize=(12, 6))
graphplt.title('Churn Count by PaymentMethod')
graphplt.xlabel('PaymentMethod')
graphplt.ylabel('Count')
graphplt.xticks(rotation=45, ha='right')  
graphplt.show()


dataFrame_churn = dataFrame
(training, testing) = dataFrame_churn.randomSplit([0.7,0.3],  30)
print("training Data:", training.count())
print("testing Data:", testing.count())


transformations = []
categorycols = ['DeviceProtection','Contract' , 'Partner', 'Dependents', 'OnlineSecurity','PhoneService', 'MultipleLines', 'InternetService', 'OnlineBackup','gender', 'TechSupport', 'StreamingTV', 'StreamingMovies','SeniorCitizen', 'PaperlessBilling', 'PaymentMethod']
for category in categorycols:
  Indexer = StringIndexer(inputCol = category, outputCol = category + "Index")
  encoder = OneHotEncoder(inputCols= [Indexer.getOutputCol()], outputCols= [category + "catVec"])
  transformations += [Indexer, encoder]


imputer = Imputer(inputCols=["TotalCharges"], outputCols=['Out_TotalCharges'])
transformations += [imputer]

labelIndex = StringIndexer(inputCol = "Churn", outputCol= "label")
transformations += [labelIndex]


tenureBin = QuantileDiscretizer(numBuckets= 3, inputCol="tenure", outputCol="tenureBin")
transformations += [tenureBin]

numericCols = ['tenureBin', 'Out_TotalCharges', 'MonthlyCharges']
assembleInputs = assemblerInputs = [c + "catVec" for c in categorycols] + numericCols
assembler = VectorAssembler(inputCols=assembleInputs, outputCol= "features")
transformations += [assembler]




temp = labelIndex.fit(training).transform(training)
temp.display()


 dataFrame.stat.corr("TotalCharges", "MonthlyCharges")


setTransform = Pipeline().setStages(transformations)
model = setTransform.fit(training)
training_dataFrame = model.transform(training)
testing_dataFrame = model.transform(testing)


training_rdd = training_dataFrame.rdd.map(lambda x: (x.features.toArray().tolist(), x.label))
dataFrame = spark.createDataFrame(training_rdd, ["features", "value"])
pdataFrame = dataFrame.toPandas() 
expanded_dataFrame = pd.concat([pdataFrame, pdataFrame['features'].apply(pd.Series)], axis=1).drop('features', axis=1)


testing_rdd = testing_dataFrame.rdd.map(lambda x: (x.features.toArray().tolist(), x.label))
dataFrame = spark.createDataFrame(testing_rdd, ["features", "value"])
pdataFrame = dataFrame.toPandas()
expandedtesting_dataFrame = pd.concat([pdataFrame, pdataFrame['features'].apply(pd.Series)], axis=1).drop('features', axis=1)

expanded_dataFrame


X_training = expanded_dataFrame[expanded_dataFrame.columns[1:]]
y_training = expanded_dataFrame['value']

X_testing = expandedtesting_dataFrame[expandedtesting_dataFrame.columns[1:]]
y_testing = expandedtesting_dataFrame['value']

# The Dataset is highly skewed 
X_training = expanded_dataFrame[expanded_dataFrame.columns[1:]]
y_training = expanded_dataFrame['value']
oversample = ADASYN(random_state=42, sampling_strategy="minority")
X_training, y_training = oversample.fit_resample(X_training, y_training)

X_testing = expandedtesting_dataFrame[expanded_dataFrame.columns[1:]]
y_testing = expandedtesting_dataFrame['value']
oversample = ADASYN(random_state=42, sampling_strategy="minority")
X_testing, y_testing = oversample.fit_resample(X_testing, y_testing)


X_training['value'] = y_training
X_testing['value'] = y_testing

training_dataFrame = X_training

testing_dataFrame = X_testing

training_dataFrame['label'] = training_dataFrame['value']
training_dataFrame.drop('value', axis= 1, inplace= True)

testing_dataFrame['label'] = testing_dataFrame['value']
testing_dataFrame.drop('value', axis= 1, inplace= True)

training_dataFrame = spark.createDataFrame(training_dataFrame)
testing_dataFrame = spark.createDataFrame(testing_dataFrame)


feature_cols = training_dataFrame.columns[:-1]
feature_cols = [str(i) for i in feature_cols]
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
training_dataFrame = vector_assembler.transform(training_dataFrame).select("features", "label")

testingfeatures_cols = testing_dataFrame.columns[:-1]
testingfeatures_cols = [str(i) for i in testingfeatures_cols]
testing_vector_assembler = VectorAssembler(inputCols= testingfeatures_cols, outputCol= "features")
testing_dataFrame = testing_vector_assembler.transform(testing_dataFrame).select('features', 'label')




testing_dataFrame.display()


training_dataFrame.display()


scaleVector = StandardScaler(inputCol="features", outputCol="new_features", withStd=True, withMean=True)
scale = scaleVector.fit(training_dataFrame)
training_dataFrame = scale.transform(training_dataFrame)

testing_scale = scaleVector.fit(testing_dataFrame)
testing_dataFrame = scale.transform(testing_dataFrame)



training_dataFrame.display()


testing_dataFrame.display()


training_rdd = training_dataFrame.rdd.map(lambda x: (DenseVector(x.new_features.toArray()), x.label))
testing_rdd = testing_dataFrame.rdd.map(lambda x: (DenseVector(x.new_features.toArray()), x.label))



training_rdd.collect()
testing_rdd.collect()


def sig(val):
    val = max(min(val, 20), -20)
    return 1 / (1 + exp(-float(val)))

def logistic_regression(data, num_iterations, learning_rate):
    weights = DenseVector([0.0] * len(data.first()[0]))
    regularization = 0.01
    for _ in range(num_iterations):
        gradient = data.map(lambda d: ((sig(weights.dot(d[0])) - d[1]) * d[0]) + regularization * weights).mean()
        
        gradient = gradient.toArray()
        gradient[0] -= regularization * weights[0]

        weights -= learning_rate * gradient
    
    return weights

num_iterations = 100
learning_rate = 0.02
weights = logistic_regression(training_rdd, num_iterations, learning_rate)





def predict(features):
    return sig(weights.dot(features))

predictions_rdd = training_rdd.map(lambda x: (predict(x[0]), x[1]))
accuracy = predictions_rdd.filter(lambda x: (x[0] >= 0.5) == x[1]).count() / float(training_rdd.count())
print(f"Accuracy: {accuracy}")

scores_and_labels = testing_dataFrame.rdd.map(lambda y: (float(weights.dot(y.new_features)), y.label))

metrics = BinaryClassificationMetrics(scores_and_labels)

print(f"Area under ROC curve (AUC): {metrics.areaUnderROC}")
print(f"Area under precision-recall curve (AUC-PR): {metrics.areaUnderPR}")

scores_and_labels = testing_dataFrame.rdd.map(lambda data_point: (float(weights.dot(data_point.new_features)), data_point.label)).collect()
scores, labels = zip(*scores_and_labels)
fp_rate, tp_rate, _ = roc_curve(labels, scores)
area_under_curve = auc(fp_rate, tp_rate)

graphplt.figure(figsize=(8, 6))
graphplt.plot(fp_rate, tp_rate, label=f'Receiver Operating Characteristic Curve (AUC = {area_under_curve:.2f})')
graphplt.plot([0, 1], [0, 1], 'k--', label='Random')
graphplt.xlabel('False Positive')
graphplt.ylabel('True Positive (Recall)')
graphplt.title('Receiver Operating Characteristic Curve')
graphplt.legend()
graphplt.show()







