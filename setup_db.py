import sqlite3

# Membuat koneksi ke database (jika belum ada, file akan dibuat)
conn = sqlite3.connect('log_database.db')
cursor = conn.cursor()

# Membuat tabel untuk menyimpan log proses
# Tabel ini akan memiliki ID, timestamp, dan pesan log
cursor.execute('''
    CREATE TABLE IF NOT EXISTS process_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        activity TEXT NOT NULL
    )
''')

print("Database dan tabel 'process_log' berhasil dibuat.")

# Menutup koneksi
conn.commit()
conn.close()