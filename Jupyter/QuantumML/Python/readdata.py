import pandas as pd
import datafitting as datacleaner

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# train =  pd.read_csv('D:/Qiskit/Tutorial/Data/train.csv')
# test = pd.read_csv('D:/Qiskit/Tutorial/Data/test.csv')

# print('train has {} rows and {} columns\n'.format(*train.shape))
# print('test has {} rows and {} columns\n'.format(*test.shape))
# train.info()
# test.info()

#pd.set_option('display.max_rows', train.shape[0]+1)
# pd.set_option('display.max_rows', 10)
# pd.set_option('display.max_columns', 6)
#print(train)

data = datacleaner.DataFitting('D:/Git/pyQML-LocalTest/Tutorial/Data/train.csv')
data.info()
print(data.head(5))