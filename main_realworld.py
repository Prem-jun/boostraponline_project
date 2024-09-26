''' 
 Decription:  
 Arguments:
 Output:
''' 
import pandas as pd
file_name = 'laptop_prices.csv'
col_name = 'Price_euros'

df = pd.read_csv(file_name)
print(df.columns)
price = df[col_name]
print(price)