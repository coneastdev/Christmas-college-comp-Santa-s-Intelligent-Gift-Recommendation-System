import sqlite3

def recomendGift(childID: str, intrests: list, wishlist: list):
    con = sqlite3.connect("./DB/childrensData.db")
    cur = con.cursor()

    cur.execute("SELECT * FROM wishlist WHERE child_id=?", (childID))

    return cur.fetchall()

def getChildData(type:str, details:str):
    con = sqlite3.connect("./DB/childrensData.db")
    cur = con.cursor()

    if type == "id":
        cur.execute("SELECT * FROM wishlist, intrests, history WHERE child_id=?", (details,))
        return cur.fetchall()
    elif type == "name":
        cur.execute("SELECT * FROM * WHERE name=?", (details,))
        return cur.fetchall()