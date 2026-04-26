import pandas as pd
import requests
import os
import re
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
CSV_FILE = 'WildScan_Data/Image_Data.csv' 
FOLDER = 'inat_images'

if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

def clean_filename(name):
    """Removes special characters and replaces spaces with underscores"""
    return re.sub(r'[^\w\s-]', '', str(name)).strip().replace(' ', '_')

def download_image(row):
    url = row['image_url']
    obs_id = row['id']
    
    # Naming: ID_Group_ScientificName.jpg
    group = clean_filename(row['iconic_taxon_name'])
    name = clean_filename(row['scientific_name'])
    filename = f"{FOLDER}/{obs_id}_{group}_{name}.jpg"
    
    if os.path.exists(filename):
        print(f"⏭️  Skipped (Exists): {obs_id}_{name}")
        return

    try:
        if pd.isna(url):
            print(f"⚠️  Missing URL for ID: {obs_id}")
            return

        response = requests.get(url, timeout=15, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            # THIS IS THE LINE YOU ASKED FOR:
            print(f"✅ Downloaded: {obs_id} | {group} | {name}")
        else:
            print(f"❌ Failed (HTTP {response.status_code}): {obs_id}")
    except Exception as e:
        print(f"❌ Error on {obs_id}: {e}")

# --- EXECUTION ---
print(f"🚀 Loading {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

# Fill blanks to prevent crashes
df['scientific_name'] = df['scientific_name'].fillna('Unknown')
df['iconic_taxon_name'] = df['iconic_taxon_name'].fillna('Unknown')

print(f"📦 Total images to process: {len(df)}")
print("--- STARTING LIVE FEED ---")

# Multi-threading for speed
with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(download_image, [row for _, row in df.iterrows()])

print(f"\n--- ALL DONE ---")
print(f"Total files in folder: {len(os.listdir(FOLDER))}")


# import pandas as pd
# import os
# import requests
# from concurrent.futures import ThreadPoolExecutor

# # Paths based on your folder structure (inside 'Data')
# CSV_PATH = "WildScan_Data/Image_Data.csv"
# IMAGE_DIR = "inat_images"

# print("--- STARTING DOWNLOAD & CLEANING ---")

# if not os.path.exists(CSV_PATH):
#     print(f"❌ ERROR: Cannot find {CSV_PATH}")
# else:
#     os.makedirs(IMAGE_DIR, exist_ok=True)
#     df = pd.read_csv(CSV_PATH)
    
#     # --- DATA CLEANING ---
#     # Fill missing values so the script doesn't crash
#     df['scientific_name'] = df['scientific_name'].fillna('Unknown')
#     df['iconic_taxon_name'] = df['iconic_taxon_name'].fillna('Unknown')
    
#     # Make names look nice (Capitalized)
#     df['iconic_taxon_name'] = df['iconic_taxon_name'].str.title()

#     def download(row):
#         # Filename format: ID_ScientificName.jpg
#         clean_sci_name = str(row['scientific_name']).replace(' ', '_')
#         filename = f"{row['id']}_{clean_sci_name}.jpg"
#         path = os.path.join(IMAGE_DIR, filename)
        
#         if os.path.exists(path): return path
        
#         try:
#             r = requests.get(row['image_url'], timeout=10)
#             if r.status_code == 200:
#                 with open(path, 'wb') as f:
#                     f.write(r.content)
#                 return path
#         except:
#             return None
#         return None

#     print(f"Processing {len(df)} rows...")
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         # We save the paths back into the dataframe
#         df['image_path'] = list(executor.map(download, [row for _, row in df.iterrows()]))

#     # --- FINAL SAVE ---
#     # Drop rows where the download failed
#     df_clean = df.dropna(subset=['image_path'])
#     df_clean.to_csv("Cleaned_Animal_Data.csv", index=False)
    
#     print(f"✅ FINISHED!")
#     print(f"Total images in folder: {len(os.listdir(IMAGE_DIR))}")
#     print(f"Cleaned CSV saved with Iconic Taxon and Scientific names included.")


import pandas as pd
import requests
import os
import re
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
CSV_FILE = 'observations-713590.csv'
FOLDER = 'inat_images'
THREADS = 12 

if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

def clean_filename(name):
    """Removes special characters and replaces spaces with underscores"""
    return re.sub(r'[^\w\s-]', '', str(name)).strip().replace(' ', '_')

def download_image(row):
    url = row['image_url']
    obs_id = row['id']
    # Create a useful name: ID_ScientificName.jpg
    # Replace your line 23 with this:
    taxon = clean_filename(row['scientific_name'])
    group = clean_filename(row['iconic_taxon_name'])
    name = clean_filename(row['scientific_name'])
    filename = f"{FOLDER}/{obs_id}_{group}_{name}.jpg"
    
    if os.path.exists(filename):
        return f"Skipped: {filename}"

    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return f"✅ Saved: {filename}"
    except Exception:
        return f"❌ Error: {obs_id}"