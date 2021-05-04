from Modules.WindScenariosGenerator.wind import wind
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dados_vento = wind('Dados_Vento_2010_2018')
anos = 1
meses = 6
vel = []
imes = 0

for iano in range(anos):
    d = dados_vento.ano[iano].wSpeed_100[imes]
    z = len(d)
    for k in range(z):
        vel.append(d[k])

dataframe = pd.DataFrame(vel)
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = numpy.zeros((testX.shape[0], 1))
testX_aux = numpy.zeros((testX.shape[0], 1, 1))
testX_aux[0, 0, 0] = testX[0, 0, 0]
for ipred in range(testX.shape[0]):
    aux = model.predict(testX_aux[:ipred+1])
    for k in range(ipred+1):
        testPredict[k, 0] = aux[k, 0]
        testX_aux[k, 0, 0] = aux[k, 0]

    # testPredict[ipred, 0] = testX_aux[ipred, 0, 0]

# testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate mean absolute percentage error
trainScore = 100*(mean_absolute_error(trainY[0], trainPredict[:, 0]))
print('Train Score: MAPE = %.2f' % (trainScore))
testScore = 100*(mean_absolute_error(testY[0], testPredict[:,0]))
print('Test Score: MAPE = %.2f' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot, 'g-')
plt.plot(testPredictPlot, 'r')
plt.show()

