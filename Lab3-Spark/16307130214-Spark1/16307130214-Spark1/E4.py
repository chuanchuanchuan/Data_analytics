def extract_month(string):
    return string_toDatetime(string).month
data = sc.textFile("/home/bigdatalab27/Downloads/mernis/data_dump_temp.sql")
new = data.filter(lambda line:line!='')
age = new_data.map(lambda lines:lines.split("\t"))
new_age = age.map(lambda line:extract_month(line[8]))
new_age.countByValue()