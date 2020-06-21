import sys
from datetime import date
from datetime import datetime

date = datetime.now().strftime("%Y%m")
server = 'aice.database.windows.net'
database = 'aice'
username = 'aice_candidate'
password = '@ic3_a3s0c1at3'
driver= '{ODBC Driver 17 for SQL Server}'

# Simple LSTM
BATCH_SIZE = 256
BUFFER_SIZE = 1000
EVALUATION_INTERVAL = 200
EPOCHS = 10

# Univariate 
uni_past_history = 24
uni_future_target = 0

# Multivariate
multi_past_history = 720
multi_future_target = 72
STEP = 6
 
command = sys.argv[1]
command_2 = sys.argv[2]
attribute = sys.argv[3]