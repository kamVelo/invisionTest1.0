import string
import random
import requests as rq
import os
def randomString(length = 6):
    alphas = list(string.ascii_letters) + list(range(1,10))
    rString =""
    for i in range(0,6):
        rString += str(random.choice(alphas))
    return rString


def download(symbol):
    """
    this method downloads all the files necessary for the regression algorithm. 
    """
    
    #downloads intraday dataset
    symbol = symbol.upper()
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol="+symbol+"&interval=5min&apikey="+randomString()+"&datatype=csv&outputsize=full"
    response = rq.get(url).content
    file = open(os.path.join(symbol, "dset.csv"), "wb")
    file.write(response)
    file.close()
    
    
    
    #downloads rsi dataset
    symbol = symbol.upper()
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol="+symbol+"&interval=5min&apikey="+randomString()+"&datatype=csv&outputsize=full"
    response = rq.get(url).content
    file = open(os.path.join(symbol, "dset.csv"), "wb")
    file.write(response)
    file.close()

download("aapl")