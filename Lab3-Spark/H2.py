from pyspark import SparkConf,SparkContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Row
from pyspark.ml.classification import NaiveBayes
from pyspark.sql import SQLContext
conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
data = sc.textFile("/home/bigdatalab27/Downloads/mernis/data_dump_temp.sql")
data = data.filter(lambda line:line!='')
schemaVal = data.map(lambda x:(x[6],x[2],x[9])).map(lambda x:Row(label_0=x[0],birth_city=x[1],id_city=x[2]))
schemaVal = sqlContext.createDataFrame(schemaVal)
(train_data,valid_data,test_data) = schemaVal.randomSplit([0.7,0.1,0.2],123)
indexer = StringIndexer(inputCol = "label_0",outputCol="label")
indexed = indexer.fit(train_data).transform(train_data)
indexer = StringIndexer(inputCol = "first_name",outputCol="fn")
indexed = indexer.fit(indexed).transform(indexed)
indexer = StringIndexer(inputCol = "id_city",outputCol="ic")
indexed = indexer.fit(indexed).transform(indexed)
assembler = VectorAssembler(inputCols=["ic","bc"],outputCol="feature")
train = assembler.transform(indexed)
nb = NaiveBayes(smoothing=1.0)
indexer = StringIndexer(inputCol = "label_0",outputCol="label")
indexed = indexer.fit(test_data).transform(test_data)
indexer = StringIndexer(inputCol = "first_name",outputCol="fn")
indexed = indexer.fit(indexer).transform(indexer)
indexer = StringIndexer(inputCol = "id_city",outputCol="ic")
indexed = indexer.fit(indexer).transform(indexer)
assembler = VectorAssembler(inputCols=["ic","bc"],outputCol="feature")
test = assembler.transform(indexed)
predictions = model.transform(test)
predictions = predictions.select("label","prediction")
top1 = predictions.rdd.filter(lambda x:x[0] == x[1]).count() / float(predictions.rdd.count())
