import pandas as pd

train = pd.read_csv('D:/Qiskit/Tutorial/Data/train.csv')
train.info()
print('\n')
#Return the first n rows; n default is 5 rows
print(train.head())
print('\n')

#Missing Values
#Most machine learning algorithms don’t work well with missing values.
#There are three options of how we can fix this:
# - Get rid of the corresponding rows (removing the passengers from consideration)
# - Get rid of the whole column (remove the whole feature for all passengers)
# - Fill the missing values (for example with zero, the mean, or the median)

# option 1
print('We only have two passengers without it. This is bearable \nRemove all rows when Embarked missed value or null \n')
train = train.dropna(subset=["Embarked"])
train.info()
print('\n')

# option 2
print('We only have very few information about the cabin, \nlet\'s drop it and all row related dropped too \n')
train = train.drop("Cabin", axis=1)
train.info()
print('\n')

# option 3
# The age misses quite a few times. But intuition
# says it might be important for someone's chance to survive.
print('The age misses quite a few times. But intuition \nsays it might be important for someone\'s chance to survive.\n')
print('Update all age missed value with mean of ages \n')
mean = train['Age'].mean()
median = train['Age'].median()
print('Mean: {}, median: {} \n'.format(mean,median))
train['Age'] = train['Age'].fillna(mean)
train.info()
print('\n')

train = train[['Name','Sex','Age','SibSp','Parch','Survived']]
train = train[train['Sex'].str.contains('female')][train['Survived'] == 0][train['Age'] >= mean]
print(train.head(5))