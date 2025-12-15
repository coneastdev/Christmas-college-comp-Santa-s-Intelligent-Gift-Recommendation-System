import csv, sqlite3

con = sqlite3.connect("./DB/childrensData.db")
cur = con.cursor()



# insert history table from csv
cur.execute("DROP TABLE history")
cur.execute("CREATE TABLE history (child_id, last_year_gift, gift_satisfaction_rating);")

with open('./DB/csv/past_gift_history_large.csv','r', encoding="utf8") as fin:
    dr = csv.DictReader(fin)
    to_db = [(i['child_id'], i["last_year_gift"], i["gift_satisfaction_rating"]) for i in dr]

cur.executemany("INSERT INTO history (child_id, last_year_gift, gift_satisfaction_rating) VALUES (?, ?, ?);", to_db)
con.commit()

# insert intrests table from csv
cur.execute("DROP TABLE intrests")
cur.execute("CREATE TABLE intrests (child_id, primary_interest, secondary_interest);")

with open('./DB/csv/interests_large.csv','r', encoding="utf8") as fin:
    dr = csv.DictReader(fin)
    to_db = [(i['child_id'], i['primary_interest'], i["secondary_interest"]) for i in dr]

cur.executemany("INSERT INTO intrests (child_id, primary_interest, secondary_interest) VALUES (?, ?, ?);", to_db)
con.commit()

# insert wishlist table from csv
cur.execute("DROP TABLE wishlist")
cur.execute("CREATE TABLE wishlist (child_id, name, wishlist_items, submitted_date);")

with open('./DB/csv/wishlist_large.csv','r', encoding="utf8") as fin:
    dr = csv.DictReader(fin)
    to_db = [(i['child_id'], i['name'], i["wishlist_items"], i["submitted_date"]) for i in dr]

cur.executemany("INSERT INTO wishlist (child_id, name, wishlist_items, submitted_date) VALUES (?, ?, ?, ?);", to_db)
con.commit()
con.close()
