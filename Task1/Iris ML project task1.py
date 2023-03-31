#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import missingno as msno


# # Reading Data from Dataset

# In[2]:


pwd


# In[3]:


df = pd.read_csv('C:\\Users\\Shreya\\iris.data')


# In[4]:


df.head()


# In[5]:


col_name = ['Sepal Length','Sepal Width','Petal Length','Petal Width','Species']


# In[6]:


df.columns = col_name


# In[7]:


df.head()


# In[8]:


df.tail()


# # Checking Data-type,Non-Null Values and Shape

# In[9]:


df.shape


# In[10]:


df.info()


# In[11]:


df.isnull().sum()


# In[12]:


df.describe()


# # Data Visualization

# In[13]:


msno.bar(df,figsize=(8,4),color='green')
plt.show()


# In[14]:


df['Species'].unique()


# In[15]:


sns.pairplot(df,hue='Species')
plt.show()


# In[17]:


plt.figure(figsize=(18,10))
plt.subplot(2,2,1)
sns.boxplot(x='Species',y='Petal Length',data=df)
plt.subplot(2,2,2)
sns.boxplot(x='Species',y='Petal Width',data=df)
plt.subplot(2,2,3)
sns.boxplot(x='Species',y='Sepal Length',data=df)
plt.subplot(2,2,4)
sns.boxplot(x='Species',y='Sepal Width',data=df)
plt.show()


# In[18]:


plt.figure(figsize=(18,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='Petal Length',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='Petal Width',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='Sepal Length',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='Sepal Width',data=df)
plt.show()


# In[19]:


plt.figure(figsize=(18,7))
sns.boxplot(data=df).set_title("Normal Distribution of Iris features\n",size=20)
plt.show()


# In[20]:


plt.figure(figsize=(18,7))
sns.violinplot(data=df).set_title("Variance of Iris features\n",size=20)
plt.show()


# In[21]:


plt.figure(figsize=(11,5))
sns.heatmap(df.corr(),annot=True,fmt='f',cmap='gist_heat').set_title('CORRELATION OF ATTRIBUTES\n',size=25)
plt.show()


# # Applying Various ML Algorithms

# In[24]:


x = df.iloc[:,0:4].values
y = df.iloc[:,4].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[26]:


from sklearn.metrics import make_scorer,accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print("All metrics included!")


# In[35]:


from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print("All ML packages included!")


# In[38]:


rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
acc_rf = round(accuracy_score(y_test,y_pred)*100,2)
rf_acc = round(rf.score(X_train,y_train)*100,2)
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred,average='micro')
recall = recall_score(y_test,y_pred,average='micro')
f1 = f1_score(y_test,y_pred,average='micro')
print("Confusion matrix of Random Forest\n",cm)
print("Accuracy of Random Forest =",acc)
print("Precision of Random Forest =",prec)
print("Recall of Random Forest =",recall)
print("f1 score of Random Forest =",f1)


# In[39]:


lg = LogisticRegression(solver='lbfgs',max_iter=400)
lg.fit(X_train,y_train)
y_pred = lg.predict(X_test)
acc_lg = round(accuracy_score(y_test,y_pred)*100,2)
lg_acc = round(lg.score(X_train,y_train)*100,2)
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred,average='micro')
recall = recall_score(y_test,y_pred,average='micro')
f1 = f1_score(y_test,y_pred,average='micro')
print("Confusion matrix of Logistic Regression\n",cm)
print("\nAccuracy of Logistic Regression =",acc)
print("\nPrecision of Logistic Regression =",prec)
print("\nRecall of Logistic Regression =",recall)
print("\nf1 score of Logistic Regression =",f1)


# In[40]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
acc_knn = round(accuracy_score(y_test,y_pred)*100,2)
knn_acc = round(knn.score(X_train,y_train)*100,2)
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred,average='micro')
recall = recall_score(y_test,y_pred,average='micro')
f1 = f1_score(y_test,y_pred,average='micro')
print("Confusion matrix of K Nearest Neighbor\n",cm)
print("\nAccuracy of K Nearest Neighbor =",acc)
print("\nPrecision of K Nearest Neighbor =",prec)
print("\nRecall of K Nearest Neighbor =",recall)
print("\nf1 score of K Nearest Neighbor =",f1)


# In[41]:


gauss = GaussianNB()
gauss.fit(X_train,y_train)
y_pred = gauss.predict(X_test)
acc_gauss = round(accuracy_score(y_test,y_pred)*100,2)
gauss_acc = round(gauss.score(X_train,y_train)*100,2)
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred,average='micro')
recall = recall_score(y_test,y_pred,average='micro')
f1 = f1_score(y_test,y_pred,average='micro')
print("Confusion matrix of K Nearest Neighbour\n",cm)
print("\nAccuracy of K Nearest Neighbour =",acc)
print("\nPrecision of K Nearest Neighbour =",prec)
print("\nRecall of K Nearest Neighbour =",recall)
print("\nf1 score of K Nearest Neighbour =",f1)


# In[42]:


lsvc = LinearSVC(max_iter=400)
lsvc.fit(X_train,y_train)
y_pred = lsvc.predict(X_test)
acc_lsvc = round(accuracy_score(y_test,y_pred)*100,2)
lsvc_acc = round(lsvc.score(X_train,y_train)*100,2)
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred,average='micro')
recall = recall_score(y_test,y_pred,average='micro')
f1 = f1_score(y_test,y_pred,average='micro')
print("Confusion matrix of K Nearest Neighbour\n",cm)
print("\nAccuracy of K Nearest Neighbour =",acc)
print("\nPrecision of K Nearest Neighbour =",prec)
print("\nRecall of K Nearest Neighbour =",recall)
print("\nf1 score of K Nearest Neighbour =",f1)


# In[47]:


plt.figure(figsize=(20,7))
a_index = list(range(1,50))
a = pd.Series()
x = range(1,50)
for i in list(range(1,50)):
  model = KNeighborsClassifier(n_neighbors=i)
  model.fit(X_train,y_train)
  prediction = model.predict(X_test)
  a = a.append(pd.Series(accuracy_score(y_test,prediction)))
plt.plot(a_index,a,marker="*")
plt.xticks(x)
plt.show()


# In[49]:


dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
acc_dt = round(accuracy_score(y_test,y_pred)*100,2)
dt_acc = round(dt.score(X_train,y_train)*100,2)
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred,average='micro')
recall = recall_score(y_test,y_pred,average='micro')
f1 = f1_score(y_test,y_pred,average='micro')
print("Confusion matrix of K Nearest Neighbour\n",cm)
print("\nAccuracy of K Nearest Neighbour =",acc)
print("\nPrecision of K Nearest Neighbour =",prec)
print("\nRecall of K Nearest Neighbour =",recall)
print("\nf1 score of K Nearest Neighbour =",f1)


# # Comparision Between Accuracies of various Applied ML Algorithms

# In[50]:


res = pd.DataFrame(
    {
        'Model':['KNN','Logistic Regression','Random Forest','Naive Bayes','Support Vector Regression','Decision Tree'],
      'Score':[acc_knn,acc_lg,acc_rf,acc_gauss,acc_lsvc,acc_dt],
       'Accuracy_score':[knn_acc,lg_acc,rf_acc,gauss_acc,lsvc_acc,dt_acc]    
    }

)

res
plt.figure(figsize=(15,10))
ax = sns.barplot(x='Model',y='Accuracy_score',data=res)
labels = (res['Accuracy_score'])
for i,v in enumerate(labels):
  ax.text(i,v+1,str(v),horizontalalignment='center',size=15,color='indigo')


# In[ ]:




