import sqlite3

for db in ["S2", "S3"]:
    path = f"output/v6_full/{db}.sqlite"
    c = sqlite3.connect(path)

    print("\n====", db, "====")

    rows = c.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table'"
    ).fetchall()

    for name, sql in rows:
        print("\nTABLE:", name)
        print(sql)

    c.close()