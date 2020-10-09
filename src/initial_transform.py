import re
import numpy as np
import pandas as pd

write_loc = '../data/intermediate/data.csv'

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

def clean_demand(row):
    c = str(row['Order_Demand'])
    m = re.search('[(]?([0-9]+)[)]?', c)
    n = m.group(1)
    return int(n) if n is not None else np.nan


def transform_write(df):
    df['Product_Code'] = df.apply(prod_code, axis=1)
    df['Product_Category'] = df.apply(prod_category, axis=1)
    df['Order_Demand'] = df.apply(clean_demand, axis=1)
    df = df.drop('Warehouse',axis=1)
    df = df.rename(columns={'Product_Code':'code', 'Product_Category':'category', 'Date':'date', 'Order_Demand':'demand'})
    df = df.dropna(axis=0)
    df.to_csv(write_loc, index=False)