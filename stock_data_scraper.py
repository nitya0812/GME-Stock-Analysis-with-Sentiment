import yfinance as yf
from datetime import datetime

ticker = 'GME'
start_date = datetime(2012, 1, 1)
end_date = datetime(2021, 9, 1)

data = yf.download(ticker, start_date, end_date)
training_data = data['2019-01-01':'2021-05-28']
testing_data = data['2021-06-01':'2021-08-31']

data.to_csv(f"{ticker}.csv")
