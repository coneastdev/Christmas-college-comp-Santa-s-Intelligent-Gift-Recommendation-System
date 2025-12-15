import sqlite3

connect = sqlite3.connect("./DB/childrensData.db")
cursor = connect.cursor()

# fetch and flatten history (handles comma-separated cells)
cursor.execute("SELECT DISTINCT last_year_gift FROM history;")
rows = cursor.fetchall()
history = []
for row in rows:
    value = row[0]
    if value is None:
        continue
    parts = [part.strip() for part in str(value).split(',') if part.strip()]
    history.extend(parts)
# keep order, remove duplicates
history = list(dict.fromkeys(history))

# fetch wishlists
cursor.execute("SELECT wishlist_items FROM wishlist;")
rows = cursor.fetchall()
formattedWishlist = []
for row in rows:
    value = row[0]
    if value is None:
        continue
    parts = [part.strip() for part in str(value).split(',') if part.strip()]
    for gift in parts:
        if gift not in formattedWishlist:
            formattedWishlist.append(gift)

# remove gifts already in history
formattedWishlist = [gift for gift in formattedWishlist if gift not in history]

# build final gift list with dedupe
seen = set()
giftList = []
for gift in history + formattedWishlist:
    ggift = gift.strip()
    if not gift or gift in seen:
        continue
    seen.add(gift)
    giftList.append(gift)

print(giftList)

# get age limits and categorys
cursor.execute("DROP TABLE gifts")
cursor.execute("CREATE TABLE gifts (gift, age_limit, category);")

for gift in giftList:
    print("\n" + gift)
    while True:
        try:
            ageLimit = int(input("enter age limit $ "))
            break
        except:
            print("not a valid number\n")
    category = input("enter category $ ")
    connect.execute("INSERT INTO gifts (gift, age_limit, category) VALUES (?, ?, ?)", [gift, ageLimit, category])
    
connect.commit()

cursor.execute("SELECT * FROM gifts;")

print(cursor.fetchall())

# add gifts to gifts table
cursor.execute("CREATE TABLE IF NOT EXISTS gifts (gift, age_limit, category);")

for gift in giftList:
    print("\n" + gift)
    while True:
        try:
            ageLimit: int = int(input("enter age limit $ "))
            break
        except:
            print("not a valid number\n")
    category = input("enter category $ ")
    connect.execute("INSERT INTO gifts (gift, age_limit, category) VALUES (?, ?, ?)", [gift, ageLimit, category])
    
connect.commit()

cursor.execute("SELECT * FROM gifts;")

print(cursor.fetchall())