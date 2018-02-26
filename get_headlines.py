import sqlite3

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

con = sqlite3.connect("top_headlines.db")
con.row_factory = dict_factory
cur = con.cursor()
cur.execute("select * from headlines")
results = cur.fetchall()
print(results)