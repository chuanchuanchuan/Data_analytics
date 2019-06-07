data = sc.textFile("/home/bigdatalab27/Downloads/mernis/data_dump_temp.sql")
data = data.filter(lambda line:line!='')
data = data.map(lambda lines:lines.split("\t"))
data = data.map(lambda line:((line[9],extract_month(line[8])),1))
data.reduceByKey(lambda x,y:x+y)
data = data.map(lambda line:(line[0][0],(line[0][1],line[1])))
data = data.groupByKey()
data.map(lambda line:(line[0],sorted(line[1],key=lambda d:d[1],reverse=True)))
data = data.map(lambda line:(line[0],(line[1][0][0],line[1][1][0])))
country_top10 = ['SANLIURFA','ADANA','AYDIN','SAMSUN','SIVAS','BURSA',\
                 'ANKARA','IZMIR','KONYA','ISTANBUL']
result = data.filter(lambda line:line[0] in country_top10)
result.collect()