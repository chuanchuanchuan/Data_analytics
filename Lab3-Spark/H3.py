from pyspark import SparkConf,SparkContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Row
from pyspark.ml.classification import NaiveBayes
from pyspark.sql import SQLContext
def extract_n(list,n):
    count = 0
    record = {}
    for i in list:
        record[count] = i
        count = count + 1
    record = sorted(record.items(),key=lambda x:x[1],reverse=True)
    result = []
    for i in range(n):
        result.append(record[i][0])
    return result
conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
data = sc.textFile("/home/bigdatalab27/Downloads/mernis/data_dump_temp.sql")
data = data.filter(lambda line:line!='')
schemaVal = data.map(lambda x:(x[3],x[7],x[9])).map(lambda x:Row(label_0=x[0],birth_city=x[1],id_city=x[2]))
schemaVal = sqlContext.createDataFrame(schemaVal)
(train_data,valid_data,test_data) = schemaVal.randomSplit([0.7,0.1,0.2],123)
indexer = StringIndexer(inputCol = "label_0",outputCol="label")
indexed = indexer.fit(train_data).transform(train_data)
indexer = StringIndexer(inputCol = "birth_city",outputCol="bc")
indexed = indexer.fit(indexed).transform(indexed)
indexer = StringIndexer(inputCol = "id_city",outputCol="ic")
indexed = indexer.fit(indexed).transform(indexed)
assembler = VectorAssembler(inputCols=["ic","bc"],outputCol="feature")
train = assembler.transform(indexed)
nb = NaiveBayes(smoothing=1.0)
indexer = StringIndexer(inputCol = "label_0",outputCol="label")
indexed = indexer.fit(test_data).transform(test_data)
indexer = StringIndexer(inputCol = "birth_city",outputCol="bc")
indexed = indexer.fit(indexer).transform(indexer)
indexer = StringIndexer(inputCol = "id_city",outputCol="ic")
indexed = indexer.fit(indexer).transform(indexer)
assembler = VectorAssembler(inputCols=["ic","bc"],outputCol="feature")
test = assembler.transform(indexed)
predictions = model.transform(test)
predictions = predictions.select("probability","label")
top1_pre = predictions.rdd.map(lambda x:(extract_n(x[0],1),x[1]))
top1 = top1_pre.filter(lambda x:int(x[1]) in x[0]).count() / float(top1_pre.count())
top2_pre = predictions.rdd.map(lambda x:(extract_n(x[0],2),x[1]))
top2 = top2_pre.filter(lambda x:int(x[1]) in x[0]).count() / float(top1_pre.count())
top3_pre = predictions.rdd.map(lambda x:(extract_n(x[0],3),x[1]))
top3 = top2_pre.filter(lambda x:int(x[1]) in x[0]).count() / float(top1_pre.count())
top4_pre = predictions.rdd.map(lambda x:(extract_n(x[0],4),x[1]))
top4 = top2_pre.filter(lambda x:int(x[1]) in x[0]).count() / float(top1_pre.count())
top5_pre = predictions.rdd.map(lambda x:(extract_n(x[0],5),x[1]))
top5 = top2_pre.filter(lambda x:int(x[1]) in x[0]).count() / float(top1_pre.count())
top1
top2
top3
top4
top5