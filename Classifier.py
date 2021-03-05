
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

# In[2]:


def load_csv(in_fname):
    dataset = pd.read_csv (in_fname)
    dataset = dataset.drop(dataset[dataset.visit==6].index)
    print (dataset.shape)
    features = dataset[[x for x in dataset.columns if x != 'spneum']]
    labels = dataset[['spneum']]
    features = features.astype(float)
    labels = labels.astype(int)
    #print (dataset[0])
    return features, labels, features.columns


# In[3]:


def load_pred_set(in_fname):
    dataset = pd.read_csv(in_fname)
    pred_df = dataset.loc[dataset.visit == 6]
    print(pred_df.shape)
    pred_features = pred_df[[x for x in pred_df.columns if x != 'spneum']]
    pred_features = pred_features.astype(float)
    return pred_features


# In[4]:


def balance_dataset(features, labels):
    pos = np.where(labels==1)[0]
    neg = np.where(labels==0)[0]
    np.random.seed(42)
    pos_sel = np.random.choice(pos, 0)
    np.random.seed(42)
    neg_sel = np.random.choice(neg, len(pos_sel)+len(pos), False)
    final_indices = np.concatenate((pos, pos_sel, neg_sel))
    return pd.DataFrame(features).loc[final_indices].values, pd.DataFrame(labels).loc[final_indices].values.flatten()


# In[5]:


def make_training_and_testing(features, labels):
    c = list(zip(features, labels))
#     np.random.seed(42)
    np.random.shuffle(c)
    features, labels = zip(*c)
    features=np.array(features)
    labels=np.array(labels)
    features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.20, stratify=labels)#, random_state=42)
    return features_train, features_test, labels_train, labels_test


# In[6]:


def ml_k_nearest_neighbours(features_train, features_test, labels_train, labels_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    model = KNeighborsClassifier(n_neighbors=3)
    scores = model_selection.cross_val_score(model, features_train, labels_train, cv=cv)
    print("KNN Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    print (classification_report(labels_test, predictions))
    print (confusion_matrix(labels_test, predictions))
    print (accuracy_score(labels_test, predictions))
    return ['KNN', accuracy_score(labels_test, predictions), precision_recall_fscore_support(labels_test, predictions)], model


# In[7]:


def ml_neural_network(features_train, features_test, labels_train, labels_test):
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.layers import Activation
    from keras.optimizers import Adam

    epochs=100
    optimizer='Adam 0.0005'
    loss='binary_crossentropy'
    model = Sequential()
    model.add(Dense(16, input_dim=len(X_train[0])))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(lr=0.0005), 
                  loss=loss,
                  metrics=['accuracy'])

    model.fit(features_train, labels_train, epochs=epochs, batch_size=28)
    score = model.evaluate(features_test, labels_test, verbose=0)
    print('\n\nThe score is: ', score)
    return ['Multi Layer Perceptron', score[1], epochs, model.layers, loss, optimizer]


# In[8]:


def ml_support_vector_machine(features_train, features_test, labels_train, labels_test):
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.svm import SVC

    model = SVC(kernel='rbf', C=1e3, gamma=1e1, class_weight='balanced', random_state=42)
    scores = model_selection.cross_val_score(model, features_train, labels_train, cv=cv)
    print("SVM Gaussian: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    print (classification_report(labels_test, predictions))
    print (confusion_matrix(labels_test, predictions))
    print (accuracy_score(labels_test, predictions))
    return ['SVM', accuracy_score(labels_test, predictions), precision_recall_fscore_support(labels_test, predictions)], model


# In[9]:


def ml_logistic_regression(features_train, features_test, labels_train, labels_test):
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    scores = model_selection.cross_val_score(model, features_train, labels_train, cv=cv)
    print("Logistic Regression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    print (classification_report(labels_test, predictions))
    print (confusion_matrix(labels_test, predictions))
    print (accuracy_score(labels_test, predictions))
    return ['Logistic Regression', accuracy_score(labels_test, predictions), precision_recall_fscore_support(labels_test, predictions)]


# In[10]:


def ml_gradient_boost(features_train, features_test, labels_train, labels_test):
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.ensemble import GradientBoostingClassifier
    
    model = GradientBoostingClassifier(n_estimators=60, random_state=42)
    scores = model_selection.cross_val_score(model, features_train, labels_train, cv=cv)
    print("Gradient Boost Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    print (classification_report(labels_test, predictions))
    print (confusion_matrix(labels_test, predictions))
    print (accuracy_score(labels_test, predictions))
    return ['Gradient Boost', accuracy_score(labels_test, predictions), precision_recall_fscore_support(labels_test, predictions)], model


# In[11]:


def ml_decision_tree(features_train, features_test, labels_train, labels_test):
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier()
    scores = model_selection.cross_val_score(model, features_train, labels_train, cv=5)
    print("Decision Tree Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    print (classification_report(labels_test, predictions))
    print (confusion_matrix(labels_test, predictions))
    print (accuracy_score(labels_test, predictions))
    return ['Decision Tree', accuracy_score(labels_test, predictions), precision_recall_fscore_support(labels_test, predictions)]


# In[12]:


def ml_random_forest(features_train, features_test, labels_train, labels_test):
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=100, max_features=None, class_weight='balanced_subsample', random_state=42)
    scores = model_selection.cross_val_score(model, features_train, labels_train, cv=cv)
    print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    print (classification_report(labels_test, predictions))
    print (confusion_matrix(labels_test, predictions))
    print (accuracy_score(labels_test, predictions))
    return ['Random Forest', accuracy_score(labels_test, predictions), precision_recall_fscore_support(labels_test, predictions)], model


# In[35]:


in_fname = 'preprocessed.csv'
out_fname = 'results_final.csv'
X, Y, columns = load_csv(in_fname)
X = X.drop('visit', axis=1)
columns = list(columns)
columns.remove('visit')
pred_X = load_pred_set('preprocessed.csv')
pred_X = pred_X.drop('visit', axis=1)
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)


# In[36]:


X_train, X_test, Y_train, Y_test = make_training_and_testing(X.values, Y.values.flatten())

row, model = ml_gradient_boost(X_train, X_test, Y_train, Y_test)
predictions_gb = model.predict_proba(X_test)
predictions = [item for item in predictions_gb[:,1]]
print(roc_auc_score(Y_test, predictions))


# In[37]:


df_X_train = pd.DataFrame(X_train, columns=X.columns)
df_X_test = pd.DataFrame(X_test, columns=X.columns)

print(len(X_train), len(X_test), len(Y_train), len(Y_test))
cont_col = ['age', 'cd4', 'familyincome', 'fathersage', 'fatherschool', 'howmanyadults', 'howmanychildren', 'howmanyrooms', 'ksedu', 'ksoccupation', 'kuppuswamyindex', 'momage', 'howlongtakes', 'momschooling']
cat_col = list(set(columns) - set(cont_col))
X_train_cat = df_X_train[cat_col]
X_train_cont = df_X_train[cont_col]
X_test_cat = df_X_test[cat_col]
X_test_cont = df_X_test[cont_col]
pred_X_cat = pred_X[cat_col]
pred_X_cont = pred_X[cont_col]


# In[38]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

estimator_cont = RandomForestClassifier(n_estimators=20, max_features=None, class_weight='balanced_subsample', random_state=42)
estimator_cont.fit(X_train_cont, Y_train)
# print(sorted(list(zip(cont_col, estimator_cont.feature_importances_)), key=lambda x: x[1], reverse=True))

estimator_cat = RandomForestClassifier(n_estimators=20, max_features=None, class_weight='balanced_subsample', random_state=42)
estimator_cat.fit(X_train_cat, Y_train);
# print(sorted(list(zip(cat_col, estimator_cat.feature_importances_)), key=lambda x: x[1], reverse=True))

from sklearn.feature_selection import SelectFromModel
selector_cont = SelectFromModel(estimator_cont)
selector_cont.fit(X_train_cont, Y_train)

selector_cat = SelectFromModel(estimator_cat)
selector_cat.fit(X_train_cat, Y_train);
# In[39]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegressionCV
# estimator = LogisticRegressionCV(Cs=[10,100,1000,10000], class_weight='balanced', cv=cv, random_state=42)
# estimator.fit(X_train, Y_train)
selector_cont = RFE(estimator_cont, n_features_to_select=5)
selector_cont.fit(X_train_cont, Y_train)

selector_cat = RFE(estimator_cat, n_features_to_select=16)
selector_cat.fit(X_train_cat, Y_train);


# In[40]:


X_train_cont = selector_cont.transform(X_train_cont)
X_train_cat = selector_cat.transform(X_train_cat)
X_train = np.concatenate((X_train_cont, X_train_cat), axis=1)

X_test_cont = selector_cont.transform(X_test_cont)
X_test_cat = selector_cat.transform(X_test_cat)
X_test = np.concatenate((X_test_cont, X_test_cat), axis=1)

pred_X_cont = selector_cont.transform(pred_X_cont)
pred_X_cat = selector_cat.transform(pred_X_cat)
pred_X = np.concatenate((pred_X_cont, pred_X_cat), axis=1)
print (len(X_train_cont), len(X_train_cont[0]), len(X_train_cat), len(X_train_cat[0]), len(X_train[0]))


# In[42]:


print('1. Neural Network with 1 hidden layer')
print('2. SVM')
print('3. Logistic Regression')
print('4. K Nearest Neighbours')
print('5. Decision Tree')
print('6. Random Forest')
print('7. Gradient Boost')
classifier=input('Enter which classifier you want to use: ')

if classifier=='1':
    row=ml_neural_network(X_train, X_test, Y_train, Y_test)
elif classifier=='2':
    row, model=ml_support_vector_machine(X_train, X_test, Y_train, Y_test)
    predictions_svm = model.decision_function(X_test)
    predictions = [item for item in predictions_svm]
elif classifier=='3':
    row=ml_logistic_regression(X_train, X_test, Y_train, Y_test)
elif classifier=='4':
    row, model=ml_k_nearest_neighbours(X_train, X_test, Y_train, Y_test)
    predictions_knn = model.predict_proba(X_test)
    predictions = [item for item in predictions_knn[:, 1]]
elif classifier=='5':
    row=ml_decision_tree(X_train, X_test, Y_train, Y_test)
elif classifier=='6':
    row, model=ml_random_forest(X_train, X_test, Y_train, Y_test)
    predictions_rf = model.predict_proba(X_test)
    predictions = [item for item in predictions_rf[:, 1]]
elif classifier=='7':
    row, model=ml_gradient_boost(X_train, X_test, Y_train, Y_test)
    predictions = model.predict_proba(X_test)


# In[43]:


print(roc_auc_score(Y_test, predictions))
fpr, tpr, thresh = roc_curve(Y_test, predictions_svm)
svm_line = plt.plot(fpr, tpr, 'b')[0]
fpr, tpr, thresh = roc_curve(Y_test, predictions_knn[:,1])
knn_line = plt.plot(fpr, tpr, 'y')[0]

# plt.legend((svm_line, rf_line, gb_line, knn_line), ('SVM', 'Random Forest', 'Gradient Boosting', 'K-Nearest Neighbours'), fontsize='small', loc='lower right')
plt.legend((svm_line, knn_line), ('SVM', 'K-Nearest Neighbours'), loc='lower right')

plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')


# In[250]:


len(predictions)


# In[165]:


notes = input('Enter notes to write to file: ')
row.insert(2, notes)
with open(out_fname, "a") as out_file:
    writer = csv.writer(out_file)
    writer.writerow(row)


# In[67]:


for item in [index for index in range(len(df_X_train.columns)) if df_X_train.values.T[index].tolist() in X_train.T.tolist()]:
    print(df_X_train.columns[item])


# In[277]:


print(roc_auc_score(Y_test, predictions_svm[:]))

