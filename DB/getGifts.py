import sqlite3

con = sqlite3.connect("./DB/childrensData.db")
cur = con.cursor()

cur.execute("SELECT DISTINCT last_year_gift FROM history;")

history:list = cur.fetchall()

cur.execute("SELECT wishlist_items FROM wishlist;")

wishlists:list = cur.fetchall()

print(wishlists)