from datetime import datetime
import pyspark.sql
def string_toDatetime(string):
    time = string.split("/")
    day = time[0]
    month = time[1]
    year = time[2]
    if len(year) == 2:
        year = '19' + year
    if year == "":
        year = '1900'
    if month not in ['1','2','3','4','5','6','7','8','9','10',\
                     '11','12']:
        month = '12'
    if day == '' or day == "0":
        day = str(1)
    standard_time = year + "-" + month + "-" + day
    try:
        s = datetime.strptime(standard_time,"%Y-%m-%d")
    except:
        standard_time = year + "-" + month + "-" + '28'
        s = datetime.strptime(standard_time, "%Y-%m-%d")
    return s

data = sc.textFile("/home/bigdatalab27/Downloads/mernis/data_dump_temp.sql")
new = data.filter(lambda line:line!='')
Turkish = new.map(lambda line:line.split('\t'))
Turkish = Turkish.map(lambda lines:(int(lines[0]),int(lines[1]),\
                                    lines[2],lines[3],lines[4],lines[5],lines[6],\
                                    lines[7],string_toDatetime(lines[8]),lines[9],\
                                    lines[10],lines[11],lines[12],lines[13],lines[14], \
                                    int(lines[15]),lines[16]))
list_dataframe = sqlContext.createDataFrame(Turkish,['uid','national_identifier','first_name','last_name','mother_first','father_first','gender','birth_city','date_of_birth','id_registration_city','id_registration_district','address_city',' address_district','address_neighborhood','stree_address','door_or_entrance_number','misc'])
list_dataframe.registerTempTable("test")
top = sqlContext.sql("""SELECT MIN(date_time) FROM test WHERE gender = 'E'""")
top.show()
