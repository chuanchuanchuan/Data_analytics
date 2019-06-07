def get_age(string):
    birth_date = string_toDatetime(string)
    birth_year = birth_date.year
    birth_month = birth_date.month
    birth_day = birth_date.day
    current_year = datetime.today().year
    current_month = datetime.today().month
    current_day = datetime.today().day
    year_gap = current_year - birth_year
    if current_month > birth_month:
        return  year_gap
    elif current_month < birth_month:
        return year_gap - 1
    else:
        if current_day >= birth_day:
            return year_gap
        else:
            return year_gap - 1

def age_rank(string):
    age = get_age(string)
    rank = 0
    if age <= 18:
        rank = 0
    elif age <= 28:
        rank = 1
    elif age <= 38:
        rank = 2
    elif age <= 48:
        rank = 3
    elif age <= 59:
        rank =  4
    else:
        rank = 5
    return rank

data = sc.textFile("/home/bigdatalab27/Downloads/mernis/data_dump_temp.sql")
new = data.filter(lambda line:line!='')
age = new_data.map(lambda lines:lines.split("\t"))
new_age = age.map(lambda line:age_rank(line[8]))
new_age.countByValue()