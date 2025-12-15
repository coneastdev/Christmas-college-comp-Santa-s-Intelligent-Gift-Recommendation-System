import sqlite3

con = sqlite3.connect("./DB/childrensData.db")
cur = con.cursor()

# fetch and flatten history (handles comma-separated cells)
cur.execute("SELECT DISTINCT last_year_gift FROM history;")
rows = cur.fetchall()
history = []
for row in rows:
    val = row[0]
    if val is None:
        continue
    parts = [p.strip() for p in str(val).split(',') if p.strip()]
    history.extend(parts)
# keep order, remove duplicates
history = list(dict.fromkeys(history))

# fetch wishlists
cur.execute("SELECT wishlist_items FROM wishlist;")
rows = cur.fetchall()
formatted_wishlist = []
for row in rows:
    val = row[0]
    if val is None:
        continue
    parts = [p.strip() for p in str(val).split(',') if p.strip()]
    for gift in parts:
        if gift not in formatted_wishlist:
            formatted_wishlist.append(gift)

# remove gifts already in history
formatted_wishlist = [g for g in formatted_wishlist if g not in history]

# build final gift list with dedupe
seen = set()
gift_list = []
for g in history + formatted_wishlist:
    g = g.strip()
    if not g or g in seen:
        continue
    seen.add(g)
    gift_list.append(g)

print(gift_list)

# get age limits and categorys
cur.execute("CREATE TABLE IF NOT EXISTS gifts (gift, age_limit, category);")

for gift in gift_list:
    print("\n" + gift)
    while True:
        try:
            ageLimit = int(input("enter age limit $ "))
            break
        except:
            print("not a valid number\n")
    category = input("enter category $ ")
    con.execute("INSERT INTO gifts (gift, age_limit, category) VALUES (?, ?, ?)", [gift, ageLimit, category])
    
con.commit()

cur.execute("SELECT * FROM gifts;")

print(cur.fetchall())

# add gifts to gifts table
cur.execute("CREATE TABLE IF NOT EXISTS gifts (gift, age_limit, category);")

for gift in gift_list:
    print("\n" + gift)
    while True:
        try:
            ageLimit: int = int(input("enter age limit $ "))
            break
        except:
            print("not a valid number\n")
    category = input("enter category $ ")
    con.execute("INSERT INTO gifts (gift, age_limit, category) VALUES (?, ?, ?)", [gift, ageLimit, category])
    
con.commit()

cur.execute("SELECT * FROM gifts;")

print(cur.fetchall())