import os
import mariadb

DB_PASSWORD = os.getenv("MOODLE_DOCKER_DBPASS")

conn = mariadb.connect(
    host="127.0.0.1",
    port=3306,
    database="moodle",
    user="moodle",
    password=DB_PASSWORD,
)

cursor = conn.cursor()

cursor.execute("SELECT DATABASE(), VERSION()")

row = cursor.fetchone()

print("Database:", row[0])
print("MariaDB version:", row[1])

cursor.execute("""
    SELECT COUNT(*)
    FROM m_logstore_standard_log
""")

print("Total Moodle log events:", cursor.fetchone()[0])

cursor.close()
conn.close()
