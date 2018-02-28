from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np
import functools
from pyspark.ml.feature import OneHotEncoder
 
tableData = sqlContext.table('your_table_containing_products_feedback_information')
 
cols_select = ['prod_price',
               'prod_feat_1',
               'prod_feat_2',
               'cust_age',
               'prod_feat_3',
               'cust_region',
               'prod_type',
               'cust_sex',
               'cust_title',
               'feedback']
df = tableData.select(cols_select).dropDuplicates()

from matplotlib import pyplot as plt
%matplotlib inline
 
responses = df.groupBy('feedback').count().collect()
categories = [i[0] for i in responses]
counts = [i[1] for i in responses]
 
ind = np.array(range(len(categories)))
width = 0.35
plt.bar(ind, counts, width=width, color='r')
 
plt.ylabel('counts')
plt.title('Response distribution')
plt.xticks(ind + width/2., categories)


from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
n 
binarize = lambda x: 'Negative' if x == 'Neutral' else x
 
udfValueToCategory = udf(binarize, StringType())
df = df.withColumn("binary_response", udfConvertResponse("feedback"))


# preprocessing data :

cols_select = ['prod_price',
               'prod_feat_1',
               'prod_feat_2',
               'cust_age',
               'prod_feat_3',
               'cust_region',
               'prod_type',
               'cust_sex',
               'cust_title',
               'feedback',
               'binary_response']
 
df = df.select(df.prod_price.cast('float'), # convert numeric cols (int or float) into a 'int' or 'float'
               df.prod_feat_1.cast('float'),
               df.prod_feat_2.cast('float'),
               df.cust_age.cast('int'),
               *cols_select[4:])
 
df = df.fillna({'cust_region': 'NA', 'cust_title': 'NA', 'prod_type': 'NA'}) # fill in 'N/A' entries for certain cols


# group categorical data 

for col in df.columns[4:-2]:
    print(col, df.select(col).distinct().count())


from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
 
COUNT_THRESHOLD = 150 # threshold to filter 
 
# create a temporary col "count" as counting for each value of "prod_feat_3"

prodFeat3Count = df.groupBy("prod_feat_3").count()
df = df.join(prodFeat3Count, "prod_feat_3", "inner")
 
def convertMinority(originalCol, colCount):
    if colCount > COUNT_THRESHOLD:
        return originalCol
    else:
        return 'MinorityCategory'
createNewColFromTwo = udf(convertMinority, StringType())
df = df.withColumn('prod_feat_3_reduced', createNewColFromTwo(df['prod_feat_3'], df['count']))
df = df.drop('prod_feat_3')
df = df.drop('count')


# One Hot Encoder 

column_vec_in = ['prod_feat_3_reduced', 'cust_region', 'prod_type', 'cust_sex', 'cust_title']
column_vec_out = ['prod_feat_3_reduced_catVec','cust_region_catVec', 'prod_type_catVec','cust_sex_catVec',
'cust_title_catVec']
 
indexers = [StringIndexer(inputCol=x, outputCol=x+'_tmp')
            for x in column_vec_in ]
 
encoders = [OneHotEncoder(dropLast=False, inputCol=x+"_tmp", outputCol=y)
for x,y in zip(column_vec_in, column_vec_out)]
tmp = [[i,j] for i,j in zip(indexers, encoders)]
tmp = [i for sublist in tmp for i in sublist]



# prepare labeled sets
cols_now = ['prod_price',
            'prod_feat_1',
            'prod_feat_2',
            'cust_age',
            'prod_feat_3_reduced_catVec',
            'cust_region_catVec',
            'prod_type_catVec',
            'cust_sex_catVec',
            'cust_title_catVec']
assembler_features = VectorAssembler(inputCols=cols_now, outputCol='features')
labelIndexer = StringIndexer(inputCol='binary_response', outputCol="label")
tmp += [assembler_features, labelIndexer]
pipeline = Pipeline(stages=tmp)



# prepare a pipline 

allData = pipeline.fit(df).transform(df)
allData.cache()
trainingData, testData = allData.randomSplit([0.8,0.2], seed=0) # need to ensure same split for each time
print("Distribution of Pos and Neg in trainingData is: ", trainingData.groupBy("label").count().take(3))

# prediction and evaluation data 


from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric
results = transformed.select(['probability', 'label'])
 
## prepare score-label set
results_collect = results.collect()
results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]
scoreAndLabels = sc.parallelize(results_list)
 
metrics = metric(scoreAndLabels)
print("The ROC score is (@numTrees=200): ", metrics.areaUnderROC)


from sklearn.metrics import roc_curve, auc
 
fpr = dict()
tpr = dict()
roc_auc = dict()
 
y_test = [i[1] for i in results_list]
y_score = [i[0] for i in results_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
 
%matplotlib inline

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


ll_probs = transformed.select("probability").collect()
pos_probs = [i[0][0] for i in all_probs]
neg_probs = [i[0][1] for i in all_probs]
 
from matplotlib import pyplot as plt
%matplotlib inline
 
# position
plt.hist(pos_probs, 50, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('predicted_values')
plt.ylabel('Counts')
plt.title('Probabilities for positive cases')
plt.grid(True)
plt.show()
 
# neg plot

plt.hist(neg_probs, 50, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('predicted_values')
plt.ylabel('Counts')
plt.title('Probabilities for negative cases')
plt.grid(True)
plt.show()


