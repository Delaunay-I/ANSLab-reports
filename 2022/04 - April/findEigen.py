import numpy as np
import pandas
from scipy.stats import linregress
import math
import sys

# total arguments
n = len(sys.argv)
if n<5:
    sys.exit("\nDid you pass the start, end and number of rows arguments?")

print("\nTotal arguments passed:", n)
 
# Arguments passed
print("Name of Python script:", sys.argv[0])

rows = int(sys.argv[1])
start = int(sys.argv[2])
end = int(sys.argv[3])
fileName = sys.argv[4]

print("Number of rows: ", rows ,", Start iter: ", start, ", End iter: ", end, '\n')

# Found the time-step by assuming that the 2nd complex eigenpair are dominant
#Dt = 0.004385111294759333
Dt = 0.0023285572429801106

if n > 5 and n < 8:
    Dt = float(sys.argv[5])
    RelEig = float(sys.argv[6])
    print("Dt:", Dt, '\u03BB:', RelEig,'\n')


def Read(rows):
    X = open(fileName, 'r')
    convData = X.read() # Read as string
    convDataList = convData.split() # Split by space
    convDataArray = np.array(convDataList) # Convert to numpy array
    #convDataArray = np.loadtxt('../residual.dat', unpack = True)
    convDataArray = convDataArray.reshape(rows, 2)
    return (convDataArray)

data = Read(rows)
convDataFrame = pandas.DataFrame(data, columns =['iter', 'l2norm'])
convDataFrame['iter'] = convDataFrame['iter'].astype(int)
convDataFrame['l2norm'] = convDataFrame['l2norm'].astype(float)

def sliceData(start, end):
    dfExtract = convDataFrame.loc[start:end, 'iter':'l2norm']
    return(dfExtract)

def calcSigma(data):
    x = data['iter'].to_numpy()
    y = data['l2norm'].to_numpy()
    logY = np.log(y)
    result = linregress(x, logY)
    slope = result.slope
    print("Slope = {}".format(slope))  
    #print("type of slope is", type(slope)) # Debugging tool
    sigma = math.exp(slope)
    print("Sigma = {}".format(sigma))
    return(sigma)
    


def findEigen(start, end, rel = -419.3468385422):
    extDF = sliceData(start, end)
    sigma = calcSigma(extDF)
    lamDt = 1 - 1/sigma
    lamda = lamDt/Dt
    
    deltaT = lamDt/rel
    return(lamda, deltaT)
    
print("Guessed Dt: ", Dt, '\n')
if n < 7:
    Lambda, deltaT = findEigen(start, end)
    print("Related Eigenvalue has \u03BB = {}".format(Lambda))
elif n==7:
    Lambda, deltaT = findEigen(start, end, RelEig)
    print("Computed \u0394t = {}".format(deltaT))
