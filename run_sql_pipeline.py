import os
import sqlite3
import pandas as pd

def main():
    # Resolve directories dynamically relative to script location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "data", "processed", "olist.db")
    sql_file = os.path.join(BASE_DIR, "sql", "cohort_rfm.sql")
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    
    print(f"Connecting to SQLite database at {db_path}...")
    conn = sqlite3.connect(db_path)
    
    print(f"Reading SQL script from {sql_file}...")
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_script = f.read()
        
    print("Executing SQL script (this may take a moment to compute window functions)...")
    # executescript handles drop, create, views, and multiple statements in one go
    conn.executescript(sql_script)
    conn.commit()
    print("SQL script executed successfully.")
    
    # Export cohort_retention
    print("Exporting cohort_retention table...")
    cohort_df = pd.read_sql_query("SELECT * FROM cohort_retention", conn)
    cohort_csv_path = os.path.join(processed_dir, "cohort_retention.csv")
    cohort_df.to_csv(cohort_csv_path, index=False)
    print(f"Saved {len(cohort_df)} rows to {cohort_csv_path}")
    
    # Export rfm_table
    print("Exporting rfm_table...")
    rfm_df = pd.read_sql_query("SELECT * FROM rfm_table", conn)
    rfm_csv_path = os.path.join(processed_dir, "rfm_table.csv")
    rfm_df.to_csv(rfm_csv_path, index=False)
    print(f"Saved {len(rfm_df)} rows to {rfm_csv_path}")
    
    conn.close()
    print("SQL Pipeline run complete.")

if __name__ == "__main__":
    main()
