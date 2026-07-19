"""Build FAISS index with SSL fix for macOS."""
import os

import truststore
truststore.inject_into_ssl()
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from dotenv import load_dotenv
load_dotenv(override=True)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

import pandas as pd
from engine import build_index

df = pd.read_csv('Output/blinkit_clean_data.csv')
print(f'Building FAISS index for {len(df)} reviews...')
idx, meta = build_index(df)
print(f'Done! {idx.ntotal} vectors, {len(meta)} metadata entries')
