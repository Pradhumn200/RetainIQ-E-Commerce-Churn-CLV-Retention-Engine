import os
import shutil
import pandas as pd
import sqlite3

def main():
    # Resolve directories dynamically relative to script location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "temp_brazil2"))
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    db_path = os.path.join(processed_dir, "olist.db")
    
    # Create directories
    print("Creating directory structure...")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "notebooks"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "sql"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "dashboard", "data"), exist_ok=True)
    
    # CSV files to copy and load
    files = [
        "olist_customers_dataset.csv",
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_payments_dataset.csv",
        "olist_products_dataset.csv",
        "product_category_name_translation.csv"
    ]
    
    # Copy files (if source directory exists)
    if os.path.exists(src_dir):
        for f in files:
            src_path = os.path.join(src_dir, f)
            dest_path = os.path.join(raw_dir, f)
            if os.path.exists(src_path):
                print(f"Copying {f} to {raw_dir}...")
                shutil.copy2(src_path, dest_path)
            else:
                print(f"Warning: {f} not found in {src_dir}!")
    else:
        print(f"Note: Source temp directory {src_dir} not found. Assuming raw CSV files already exist in {raw_dir}.")
            
    # Connect to SQLite
    print(f"Connecting to SQLite database at {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # Load each CSV into SQLite
    for f in files:
        csv_path = os.path.join(raw_dir, f)
        if os.path.exists(csv_path):
            table_name = f.replace("_dataset.csv", "").replace(".csv", "")
            print(f"Loading {f} into table '{table_name}'...")
            df = pd.read_csv(csv_path)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"Loaded {len(df)} rows into '{table_name}'.")
        else:
            print(f"Error: {f} is missing from {raw_dir}!")
            
    conn.close()
    print("Data setup and SQLite database loading complete.")

if __name__ == "__main__":
    main()
