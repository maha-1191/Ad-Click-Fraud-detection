
import sqlite3
import os
import sys

DB_PATH = 'db.sqlite3'
REPORT_PATH = 'db_report.txt'

def inspect_db():
    if not os.path.exists(DB_PATH):
        try:
            with open(REPORT_PATH, 'w', encoding='utf-8') as f:
                f.write(f"Error: {DB_PATH} not found.")
        except:
             print(f"Error: {DB_PATH} not found.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write(f"--- Database Inspection: {DB_PATH} ---\n\n")

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            if not tables:
                f.write("No tables found.\n")
                return

            for table_name_tuple in tables:
                table_name = table_name_tuple[0]
                f.write(f"Table: {table_name}\n")
                
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchall()[0][0]
                    f.write(f"  Total Rows: {count}\n")
                    
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
                    rows = cursor.fetchall()
                    
                    if cursor.description:
                        columns = [d[0] for d in cursor.description]
                        f.write(f"  Columns: {columns}\n")
                    
                    if rows:
                        f.write("  Sample Data:\n")
                        for row in rows:
                            f.write(f"    {repr(row)}\n")
                    else:
                        f.write("  (Table is empty)\n")
                except sqlite3.Error as e:
                    f.write(f"  Error reading table {table_name}: {e}\n")
                
                f.write("-" * 30 + "\n\n")
                
        print(f"Report written to {REPORT_PATH}")

    except Exception as e:
        print(f"Script Error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    inspect_db()
