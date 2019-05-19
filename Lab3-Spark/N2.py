def get_accurate_age(age):
    if age > 100:age = 100
    return age
data = sc.textFile("/home/bigdatalab27/Downloads/mernis/data_dump_temp.sql")
data = data.filter(lambda line:line!='')
data = data.map(lambda lines:lines.split("\t"))
data = data.map(lambda line:(line[9],get_accurate_age(get_age(line[8]))))
sumCount = data.combineByKey((lambda x:(x,1)),(lambda x,y:(x[0]+y,x[1]+1)),(lambda x,y:(x[0]+y[0],x[1]+y[1])))
result = sumCount.map(lambda lines:(lines[1][0]/lines[1][1],lines[0]))
result.sortByKey().collect()