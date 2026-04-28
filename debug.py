import pandas as pd
df = pd.read_csv('data/churn prediction.csv')
df = df.drop(['CustomerID', 'LastPurchaseDate'], axis=1, errors='ignore')
df = df.drop_duplicates()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# Get unique values for categorical columns
print('PaymentMethod unique:', df['PaymentMethod'].unique()[:5])
print('City unique:', df['City'].unique()[:5])
print('State unique:', df['State'].unique()[:5])
print('Country unique:', df['Country'].unique()[:5])
print('Browser unique:', df['Browser'].unique()[:5])

