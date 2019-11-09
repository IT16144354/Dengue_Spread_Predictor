import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import explained_variance_score
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = [16,9]

print("----")
X = pd.read_csv('./data/dengue_features_train.csv')
y = pd.read_csv('./data/dengue_labels_train.csv')
X_test = pd.read_csv('./data/dengue_features_test.csv')


X.isna().sum() 

#detect missing values of dengue_features_test data get sum
X_test.isna().sum()

#get records of 'San Juan' in dengue_features_train data
X_sj = X.loc[X['city'] =='sj'].copy()

#get records of 'San Juan' in dengue_labels_train data
y_sj = y.loc[y['city'] =='sj'].copy()


#get records of 'Iquitos' in dengue_features_train data
X_iq = X.loc[X['city'] =='iq'].copy()

#get records of 'Iquitos' in dengue_labels_train data
y_iq = y.loc[y['city'] =='iq'].copy()


X_sj_test = X_test.loc[X_test['city'] == 'sj'].copy()#get the city = "sj" rows


#get records of 'Iquitos' in dengue_features_test data
X_iq_test = X_test.loc[X_test['city'] == 'iq'].copy()#get the city = "iq" rows

X_sj.shape

X_iq.shape

X_sj.drop(labels=['city', 'week_start_date'], axis=1,inplace=True)
X_iq.drop(labels=['city', 'week_start_date'], axis=1,inplace=True)#Drop specified labels from rows or columns.
X_sj_test.drop(labels=['city', 'week_start_date'], axis=1,inplace=True)#Drop specified labels from rows or columns.
X_iq_test.drop(labels=['city', 'week_start_date'], axis=1,inplace=True)


X_sj['total_cases'] = y_sj.total_cases
X_iq['total_cases'] = y_iq.total_cases

X_sj.corr()
X_iq.corr()

X_sj.corr().abs().total_cases.drop('total_cases').sort_values()
X_iq.corr().abs().total_cases.drop('total_cases').sort_values()

X_sj_medianf = X_sj.fillna(X_sj.median())#Fill NA/NaN values using the specified method -median()-calculate median value from an unsorted data-list
X_iq_medianf = X_iq.fillna(X_iq.median())#Fill NA/NaN values using the specified method -median()-calculate median value from an unsorted data-list
X_sj_test_medianf = X_sj_test.fillna(X_sj_test.median())#Fill NA/NaN values using the specified method -median()-calculate median value from an unsorted data-list
X_iq_test_medianf = X_iq_test.fillna(X_iq_test.median())

X_sj_medianf.drop('total_cases', axis=1, inplace=True)
X_iq_medianf.drop('total_cases', axis=1, inplace=True)

pca = PCA()#Principal component analysis - Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space
pca.fit(X_sj_medianf)#Fit the model with X_sj_medianf
X_sj_pca = pca.transform(X_sj_medianf)#apply the dimensionality reduction on X_sj_medianf
X_sj_pca_test = pca.transform(X_sj_test_medianf)
np.cumsum(pca.explained_variance_ratio_)

pca = PCA()#Principal component analysis - Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space
pca.fit(X_iq_medianf)#Fit the model with X_iq_medianf
X_iq_pca = pca.transform(X_iq_medianf)
X_iq_pca_test = pca.transform(X_iq_test_medianf)
np.cumsum(pca.explained_variance_ratio_)

sj_size = int(X_sj.shape[0] * 0.8)
iq_size = int(X_iq.shape[0] * 0.8)

X_sj.fillna(X_sj.median(), inplace=True)
X_iq.fillna(X_sj.median(), inplace=True)

train_sj = X_sj.head(sj_size).copy()
test_sj = X_sj.tail(X_sj.shape[0] - sj_size).copy()

train_iq = X_iq.head(iq_size).copy()
test_iq = X_iq.tail(X_iq.shape[0] - iq_size).copy()

X_sj_test.fillna(X_sj.median(), inplace=True)
X_iq_test.fillna(X_iq.median(), inplace=True)

X_train = train_sj.drop(labels=['total_cases'], axis=1)
y_train = train_sj['total_cases']
X_test = test_sj.drop(labels=['total_cases'], axis=1)
y_test = test_sj['total_cases']

MLPRegressor()

scores = ["explained_variance" , "neg_mean_absolute_error" ,"neg_mean_squared_error" ,"r2"]
model = MLPRegressor(max_iter=10000)
parameters = {'hidden_layer_sizes': [(100, ), (20, 30), (13, 13, 13)]}

for score in scores:
   

    clf = GridSearchCV(model, parameters, cv=TimeSeriesSplit(n_splits=3),
             scoring='%s' % score)
    clf.fit(X_train, y_train)

    clf.best_params_
   
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
 
    
    y_true, y_pred = y_test, clf.predict(X_test).astype(int)
    
    
    MAE(y_true, y_pred)
 
 
clf = MLPRegressor(max_iter=10000, hidden_layer_sizes=(100,))
clf.fit(X_train, y_train)
y_true, y_pred = y_test, clf.predict(X_test).astype(int)
#MAE(y_true, y_pred)
#print("San Juan Mean Absolute Error(MAE): %f" %MAE(y_true, y_pred))
#print("Explained Variance :%f  " %explained_variance_score(y_true, y_pred,multioutput='uniform_average'))

plt.plot(y_true, label='true')
plt.plot(y_pred, label='predicted')
plt.legend()
plt.title('San Juan Prediction')	
#plt.show()


X_train = train_iq.drop(labels=['total_cases'], axis=1)
y_train = train_iq['total_cases']
X_test = test_iq.drop(labels=['total_cases'], axis=1)
y_test = test_iq['total_cases']
 
MLPRegressor()

scores1 = ["explained_variance" , "neg_mean_absolute_error" ,"neg_mean_squared_error" ,"r2"]
model = MLPRegressor(max_iter=10000)
parameters = {'hidden_layer_sizes': [(100, ), (20, 30), (13, 13, 13)]} 
for scor in scores1:
    clf = GridSearchCV(model, parameters, cv=TimeSeriesSplit(n_splits=5),
             scoring='%s' % scor)
    clf.fit(X_train, y_train)

    clf.best_params_
   
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
       print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
      
    
    y_true, y_pred = y_test, clf.predict(X_test).astype(int)
    
    
    MAE(y_true, y_pred)
    
clf = MLPRegressor(max_iter=10000, hidden_layer_sizes=(13,13,13))
clf.fit(X_train, y_train)
y_true, y_pred = y_test, clf.predict(X_test).astype(int)
#print("Iquitos Mean Absolute Error(MAE): %f" %MAE(y_true, y_pred))
#print("Explained Variance :%f  " %explained_variance_score(y_true, y_pred,multioutput='uniform_average'))

plt.plot(y_true, label='true')
plt.plot(y_pred, label='predicted')
plt.legend()
plt.title('Iquitos Prediction')	
#plt.show()




X_train = train_sj.append(test_sj, ignore_index=True)
y_train = X_train['total_cases']
X_train = X_train.drop(labels=['total_cases'], axis=1)
X_test = X_sj_test 

clf = MLPRegressor(max_iter=10000, hidden_layer_sizes=(100,))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test).astype(int)


sub_df_sj = pd.DataFrame(y_pred, columns=["total_cases"])
sub_df_sj.insert(0, 'city', 'sj')
sub_df_sj.insert(1, 'year', X_test['year'])
sub_df_sj.insert(2, 'weekofyear', X_test['weekofyear'])
sub_df_sj


X_train = train_iq.append(test_iq, ignore_index=True)
y_train = X_train['total_cases']
X_train = X_train.drop(labels=['total_cases'], axis=1)
X_test = X_iq_test

clf = MLPRegressor(max_iter=10000, hidden_layer_sizes=(13,13,13))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test).astype(int)

sub_df_iq = pd.DataFrame(y_pred, columns=["total_cases"])
sub_df_iq.insert(0, 'city', 'iq')
sub_df_iq.insert(1, 'year', X_test['year'])
sub_df_iq.insert(2, 'weekofyear', X_test['weekofyear'])
sub_df_iq
 
sub = sub_df_sj.append(sub_df_iq, ignore_index=True)

X_train = train_sj.append(test_sj, ignore_index=True)
y_train = X_train['total_cases']
X_train = X_train.drop(labels=['total_cases'], axis=1)
X_test = X_sj_test

 
pca = PCA(n_components=5)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
#print("San Juan pca explained variance ratio")
#print(pca.explained_variance_ratio_)



clf = MLPRegressor(max_iter=10000, hidden_layer_sizes=(100,))
clf.fit(X_train, y_train)
y_pred = np.abs(clf.predict(X_test_pca)).astype(int)

sub_df_sj = pd.DataFrame(y_pred, columns=["total_cases"])
sub_df_sj.insert(0, 'city', 'sj')
sub_df_sj.insert(1, 'year', X_test['year'])
sub_df_sj.insert(2, 'weekofyear', X_test['weekofyear'])
sub_df_sj

X_train = train_iq.append(test_iq, ignore_index=True)
y_train = X_train['total_cases']
X_train = X_train.drop(labels=['total_cases'], axis=1)
X_test = X_iq_test


pca = PCA(n_components=5)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
#print("Iqitos pca explained variance ratio")
#print(pca.explained_variance_ratio_)


clf = MLPRegressor(max_iter=10000, hidden_layer_sizes=(13,13,13))
clf.fit(X_train, y_train)
print("Accuracy score: ",clf.score(X_train, y_train))
y_pred = np.abs(clf.predict(X_test_pca)).astype(int)


sub_df_iq = pd.DataFrame(y_pred, columns=["total_cases"])
sub_df_iq.insert(0, 'city', 'iq')
sub_df_iq.insert(1, 'year', X_test['year'])
sub_df_iq.insert(2, 'weekofyear', X_test['weekofyear'])
sub_df_iq

sub = sub_df_sj.append(sub_df_iq, ignore_index=True)

joblib.dump(clf, 'nn_model.pkl')
print("Model dumped!")

clf = joblib.load('nn_model.pkl')


model_columns = list(X_train)
joblib.dump(model_columns, 'model_columns.pkl')


print("Models columns dumped!")
