#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset = ['Hello my name is james',
          'james this is my python notebook named james',
          'james trying to create a big dataset',
          'james of words to try differnt',
          'features of count vectorizer']


# In[3]:


dataset


# # Count Vectorizer

# In[4]:


from sklearn.feature_extraction.text import CountVectorizer


# In[5]:


cv=CountVectorizer(lowercase=False)


# In[6]:


x=cv.fit_transform(dataset)


# In[7]:


feature_name= cv.get_feature_names()  #features of the dataset
feature_name


# In[ ]:


cv.vocabulary_ #position of the words in the matrix


# In[ ]:


count_array= x.toarray() 
count_array


# In[8]:


df = pd.DataFrame(data=count_array,columns = feature_name) #sparse matrix of dataset


# In[9]:


df.shape


# In[10]:


df


# # TF-IDF Vectorizer 

# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[12]:


cv=TfidfVectorizer()


# In[13]:


x=cv.fit_transform(dataset)


# In[14]:


feature_name= cv.get_feature_names()  #features of the dataset
feature_name


# In[15]:


cv.vocabulary_ #position of the words in the matrix


# In[16]:


count_array= x.toarray() 
count_array


# In[17]:


df = pd.DataFrame(data=count_array,columns = feature_name) #sparse matrix of dataset
df.shape


# In[18]:


df


# # Spam Email Detection using Vectorizer

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


df = pd.read_csv('C:/Users/Shishir/Desktop/ML-20230831T022512Z-001/ML/Data/emails.csv')


# In[25]:


df


# In[26]:


df.shape


# In[27]:


df['spam'].value_counts()


# In[28]:


seaborn.countplot(x='spam',data=df)


# In[29]:


df.isnull().sum()


# In[30]:


X= df.text.values
y= df.spam.values


# In[31]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X_vectorized=cv.fit_transform(X)
X_vectorized.toarray()


# In[32]:


#Dataset splitting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_vectorized,y,test_size=.25,random_state=1)


# In[33]:


from sklearn.naive_bayes import MultinomialNB

#Create a Gaussian Classifier
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
pred=mnb.predict(X_test)


# In[34]:


print("Accuracy score: ", accuracy_score(y_test,pred))


# In[35]:


confusion_matrix(y_test,pred)


# In[36]:


print(classification_report(y_test,pred))


# In[37]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
seaborn.heatmap(pd.DataFrame(confusion_matrix(y_test,pred)), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[ ]:




