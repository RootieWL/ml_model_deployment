import pyodbc
import pandas as pd

server = 'aice.database.windows.net'
database = 'aice'
username = 'aice_candidate'
password = '@ic3_a3s0c1at3'
driver= '{ODBC Driver 17 for SQL Server}'

cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';\
                      PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

sql = '''
      SELECT date,  hr, weather, temperature, feels_like_temperature, relative_humidity,
             windspeed, psi, guest_scooter, registered_scooter
      from rental_data
      where date >= '2011-01-01' and date <= '2012-12-31'
      '''

data = pd.read_sql(sql, cnxn)

data.to_csv('extract.csv',index=False)