import pandas as pd
import mysql.connector

data = pd.read_csv('data/Melbourne_housing_imputed.csv')

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='1234',
    database='house_price_db'
)
cursor = conn.cursor()

for _, row in data.iterrows():
    cursor.execute("""
        INSERT INTO training_data (Suburb, Address, Rooms, Type, Method, Distance, Postcode,
                                   Bathroom, Car, Landsize, BuildingArea, YearBuilt, CouncilArea,
                                   Regionname, Propertycount)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, tuple(row[col] for col in data.columns))

conn.commit()
cursor.close()
conn.close()
