import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os
import datetime as dt
class algo:
    def __init__(self, symbol):
        self.symbol = symbol.upper()
    def poly(self):
        dset = pd.read_csv(os.path.join(self.symbol, "dset.csv")).iloc[:,1:]
        rsi = pd.read_csv(os.path.join(self.symbol, "rsi.csv")).iloc[:, 1:]
        rsiDummy = pd.DataFrame(columns={"lowRsi", "middle", "highRsi"})
        for index, row in rsi.iterrows():
            _rsi = row["RSI"]
            if _rsi < 30:
                row = [1,0,0]
            elif _rsi > 70:
                row = [0,0,1]
            else:
                row = [0,1,0]
            rsiDummy.loc[index] = row
        
                
        
        
        
        lengths = [len(dset), len(rsiDummy)]
        idx = min(lengths)
        dset = dset[:idx]
        rsiDummy = rsiDummy[:idx]
        dset = pd.concat([dset, rsiDummy], axis=1)
        dset = dset.iloc[::-1]
        dset.index = pd.RangeIndex(start = 0, stop=len(dset), step = 1)
        
        #gets the open column from dataset and converts to numpy array
        opens = dset["open"].values
        #adds a NaN value to the bottom
        opens = np.append(opens, np.NaN)
        #deletes top value to virtually shift each open up(i.e making it the 'future open')
        opens = np.delete(opens, 0, axis = 0)
        #converts it to a dataframe so its easier to append to the dataset and other things 
        opens = pd.DataFrame(opens)
        #names the column future opens for ease of identification
        opens.columns = {"Future Opens"}
        
        #gets the open column from dataset and converts to numpy array
        closes = dset["close"].values
        #adds a NaN value to the bottom
        closes = np.append(closes, np.NaN)
        #deletes top value to virtually shift each open up(i.e making it the 'future open')
        closes = np.delete(closes, 0, axis = 0)
        #converts it to a dataframe so its easier to append to the dataset and other things 
        closes = pd.DataFrame(closes)
        #names the column future opens for ease of identification
        closes.columns = {"Future Closes"}
        del dset["open"], dset["high"], dset["low"], dset["volume"]
        
        opens = opens[:-1]
        closes = closes[:-1]
        dset = dset[:-1]
        
        x = dset
        y = np.array(closes)
        
        split = int(round(len(dset)*0.8,0))
        xTrain = x[:split]
        xTest = x[split:]
        yTrain = y[:split]
        yTest = y[split:]
        
        
        self.sc_x = StandardScaler()
        xTrainTemp = self.sc_x.fit_transform(xTrain.iloc[:,:-3])
        xTestTemp = self.sc_x.fit_transform(xTest.iloc[:,:-3])
        xTrain = np.append(xTrainTemp, xTrain.iloc[:,1:], axis=1)
        xTest = np.append(xTestTemp, xTest.iloc[:,1:], axis=1)
        
        self.sc_y = StandardScaler()
        yTrain = self.sc_y.fit_transform(yTrain.reshape(-1, 1))
        
        
        self.polyReg = PolynomialFeatures(degree=2)
        xPolyTrain = self.polyReg.fit_transform(xTrain)
        xPolyTest = self.polyReg.fit_transform(xTest)
        
        self.linReg = LinearRegression()
        
        self.linReg.fit(xPolyTrain, yTrain)
        
        yPred = self.sc_y.inverse_transform(self.linReg.predict(xPolyTest))
        
        diffTest = pd.DataFrame(yPred)
        diffTest.columns = {"prediction"}
        diffTest["actual"] = yTest
        diffTest["differences"] = diffTest.diff(axis=1)["actual"]
        avg = abs(diffTest["differences"].rolling(len(diffTest)).mean()[len(diffTest)-1])
        print("The average inaccuracy is: %s" % avg)
        
        
        dirTest = pd.DataFrame(opens[split:])
        
        dirTest.columns = {"open"}
        dirTest.index = pd.RangeIndex(start = 0, stop=len(dirTest), step = 1)
        dirTest = pd.concat([dirTest,diffTest.iloc[:,:2]],axis=1)
        correct = 0
        profit = 0
        loss = 0
        for index, row in dirTest.iterrows():
            opens = row["open"]
            pred = row["prediction"] 
            close = row["actual"]
            predDif= opens - pred
            accDif = opens - close
            
            if accDif*predDif > 0:
                correct += 1
                profit += abs(accDif)
            else:
                loss += abs(accDif)
        shares = 25000/320
        profit*= shares
        pTage = correct/len(dirTest)*100
        print("The percentage of correct predictions: %s" % pTage)
        return xPolyTrain, yTrain, xPolyTest

a = algo("AAPL")
x,y,xTest =  a.poly()
        

