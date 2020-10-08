import re
import numpy as np
import pandas as pd

def prod_code(row):
    c = str(row['Product_Code'])
    # Parse
    m = re.search('Product_([0-9]+)', c)
    n = m.group(1)
    return int(n) if n is not None else np.nan

def prod_category(row):
    c = str(row['Product_Category'])
    # Parse
    m = re.search('Category_([0-9]+)', c)
    n = m.group(1)
    return int(n) if n is not None else np.nan

def prod_warehouse(row):
    c = str(row['Warehouse'])
    # Parse
    m = re.search('Whse_([A-Z]+)', c)
    n = m.group(1)
    return (ord(n) - 65) if n is not None else np.nan

def transform():
    df = pd.read_csv('../data/initial/product_demand.csv')
    df['Product_Code'] = df.apply(prod_code, axis=1).astype(np.short)
    df['Product_Category'] = df.apply(prod_category, axis=1).astype(np.short)
    df['Warehouse'] = df.apply(prod_warehouse, axis=1).astype(np.short)
    df['Date'] = pd.to_datetime(df['Date'])
    df.to_csv('../data/intermediate/data.csv')