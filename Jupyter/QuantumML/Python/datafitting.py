import pandas as pd
from sklearn.preprocessing import LabelEncoder

def DataFitting(filename):
    #define train as variable dataset of train data
    pdTrain = pd.read_csv(filename);
    #define label encoder from sklearn
    lblEncoder = LabelEncoder()

    #Missing Values
    #Most machine learning algorithms don’t work well with missing values.
    #There are three options of how we can fix this:
    # - Get rid of the corresponding rows (removing the passengers from consideration)
    # - Get rid of the whole column (remove the whole feature for all passengers)
    # - Fill the missing values (for example with zero, the mean, or the median)

    totalRow = len(pdTrain.index);
    #print('Total Row:', totalRow)

    #find non null from passanger consideration with under 1% to remove all that records
    #print(pdTrain["Embarked"].isna().sum())
    for (columnName, columnData) in pdTrain.iteritems():
        #print("Column Name", columnName)
        #print("Column Data", columnData)
        #print(pdTrain["Embarked"].isna().sum())
        #print(pdTrain["Embarked"].count())
        #find less than or equal 1 perseon missing value per each column to clean up
        if ((pdTrain[columnName].isna().sum() / totalRow) * 100 <= 1):
            pdTrain = pdTrain.dropna(subset=[columnName])
            totalRow = len(pdTrain.index)

    #drop column with leak valid values such as under 30% valid values.
    #print((pdTrain["Cabin"].count() / totalRow) * 100 )
    for (columnName, columnData) in pdTrain.iteritems():
        totalRowCount = pdTrain[columnName].count();
        if ((totalRowCount / totalRow) * 100 < 30):
            pdTrain = pdTrain.drop(columnName, axis=1)
            totalRow = len(pdTrain.index)

    #fill missing value of int / float under and equal 20% with mean value of their column
    for (columnName, dataType) in pdTrain.dtypes.iteritems(): 
        if (dataType == "int64" or dataType == "float64"):
            #print("Column Name {} with data type: {}".format(columnName, dataType))
            #print(pdTrain[columnName].isna().sum())
            totalIsNa = pdTrain[columnName].isna().sum()
            if ((totalIsNa / totalRow) * 100 <= 20  ):
                pdTrain[columnName] = pdTrain[columnName].fillna(pdTrain[columnName].mean)
                totalRow = len(pdTrain.index)

    # the goal algorithm is to predict labels we don’t know yet, prevent momorize data
    # clean up all indetifier unique of rows
    for (columnName, columnData) in pdTrain.iteritems():
        totalUnique = pdTrain[columnName].nunique()
        #print((totalUnique / totalRow) * 100)
        if ((totalUnique / totalRow) * 100 >= 70):
            pdTrain = pdTrain.drop(columnName, axis=1)
            totalRow = len(pdTrain.index)

    #both classic and quantum algorithms, work with numbers
    #we need to translate textual data into numbers
    for (columnName, dataType) in pdTrain.dtypes.iteritems():
        if (dataType == "object"):
            pdTrain[columnName] = lblEncoder.fit_transform(pdTrain[columnName].astype(str))
            totalRow = len(pdTrain.index)

    #Machine learning algorithms only work with numbers

    #pdTrain.info();
    return pdTrain

# data = AutoCleanDataTrain('D:/Qiskit/Tutorial/Data/train.csv')
# data.info()
