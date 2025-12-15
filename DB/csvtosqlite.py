import csv, sqlite3

connect = sqlite3.connect("./DB/childrensData.db")
cursor = connect.cursor()



# insert history table from csv
cursor.execute("DROP TABLE history")
cursor.execute("CREATE TABLE history (child_id, last_year_gift, gift_satisfaction_rating);")

with open('./DB/csv/past_gift_history_large.csv','r', encoding="utf8") as fin:
    dictReader = csv.DictReader(fin)
    toDB = [(i['child_id'], i["last_year_gift"], i["gift_satisfaction_rating"]) for i in dictReader]

cursor.executemany("INSERT INTO history (child_id, last_year_gift, gift_satisfaction_rating) VALUES (?, ?, ?);", toDB)
connect.commit()

# insert intrests table from csv
cursor.execute("DROP TABLE intrests")
cursor.execute("CREATE TABLE intrests (child_id, primary_interest, secondary_interest);")

with open('./DB/csv/interests_large.csv','r', encoding="utf8") as fin:
    dictReader = csv.DictReader(fin)
    toDB = [(i['child_id'], i['primary_interest'], i["secondary_interest"]) for i in dictReader]

cursor.executemany("INSERT INTO intrests (child_id, primary_interest, secondary_interest) VALUES (?, ?, ?);", toDB)
connect.commit()

# insert wishlist table from csv
cursor.execute("DROP TABLE wishlist")
cursor.execute("CREATE TABLE wishlist (child_id, name, wishlist_items, submitted_date);")

with open('./DB/csv/wishlist_large.csv','r', encoding="utf8") as fin:
    dictReader = csv.DictReader(fin)
    toDB = [(i['child_id'], i['name'], i["wishlist_items"], i["submitted_date"]) for i in dictReader]

cursor.executemany("INSERT INTO wishlist (child_id, name, wishlist_items, submitted_date) VALUES (?, ?, ?, ?);", toDB)
connect.commit()
connect.close()
