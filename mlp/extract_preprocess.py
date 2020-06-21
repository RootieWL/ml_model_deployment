import pyodbc
import numpy as np
import pandas as pd
import params as params
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os 
import time


# Start time
def get_datetime():
    print('')
    print('---------- ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ----------')
    print('')

# extract file
def extract_csv(driver, server, database, username, password):
    cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';\
                           PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()

    sql = ''' SELECT date,  hr, weather, temperature, feels_like_temperature, relative_humidity,
                     windspeed, psi, guest_scooter, registered_scooter
              FROM rental_data
              WHERE date >= '2011-01-01' and date <= '2012-12-31' '''
                
    df = pd.read_sql(sql, cnxn)
    df.to_csv('extract.csv',index=False)
    if os.path.isfile('extract.csv'):
        print('File extraction successful \n')
        print(df[:5])
    else:
        print('File extraction failed')
    
    return df

# Preprocess File
def preprocess_csv_file(raw_df):
    df = raw_df
    def to_celsius(x):
        x = float(x)
        return round((x-32)*5/9,1)

    # Sort by date and hr
    df.sort_values(by=['date','hr'], inplace=True, ascending=True)
    df['temperature'] = df['temperature'].apply(to_celsius)
    df['feels_like_temperature'] = df['feels_like_temperature'].apply(to_celsius)
    df['total_scooter_users'] = df['guest_scooter'] + df['registered_scooter'] 

    # Fix spelling errors in weather
    df['weather'] = df['weather'].str.lower().replace('lear','clear').replace('clar','clear')
    df['weather'] = df['weather'].str.replace('loudy','cloudy').replace('ccloudy','cloudy').replace('cludy','cloudy')
    df['weather'] = df['weather'].str.replace('liht snow/rain','light snow/rain')
    df['date_hr'] = pd.to_datetime(df['date']) + df['hr'].astype('timedelta64[h]')
    
    # Fill NaN and negative values to 0
    df.fillna(0)
    num = df._get_numeric_data()
    num[num < 0] = 0
    df = df.reset_index(drop=True)

    # Group by YYYY-MM-DD-HH and Convert to timeseries
    df.drop(['date','hr'], axis=1, inplace=True)
    ts_df = df.copy(deep=True)
    ts_df = ts_df.groupby(['date_hr']).mean().reset_index()
    ts_df.set_index('date_hr', inplace=True)
    print('\n')
    print('Raw_data has been preprocessed and extracted into time series dataframe\n')
    print(ts_df.reset_index().head())
    ts_df.reset_index().to_csv('preproc.csv',index=False)
    return ts_df

def main():
    get_datetime()
    raw_data = extract_csv(params.driver, params.server, params.database, params.username, params.password)
    preproc_data = preprocess_csv_file(raw_data)
    attributes = [x for x in preproc_data._get_numeric_data().columns]
    print('\n')
    print('Available attributes for Modeling: \n' + ', '.join(attributes))

    print('Job Completed!')
    get_datetime()
    
if __name__ == '__main__':
    main()