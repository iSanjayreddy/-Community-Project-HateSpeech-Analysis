#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importind libraries required
import pandas as pd
import numpy as np


# In[2]:


#impporting the data 
data = pd.read_csv("twitter.csv")


# In[3]:


#checking the data 
data


# In[4]:


data.describe()


# In[5]:


#cchecking whether any null entries are present or not
data.isnull().sum()


# In[6]:


#Labeling the tweets
data["labels"]  = data["class"].map({0: "Hate speech",
                                       1:"Offensive speech",
                                       2:"Neither hate nor offensive"})


# In[7]:


data


# In[8]:


data = data[["tweet","labels"]]


# In[9]:


data


# In[10]:


#Data preprocessing-(removing ! & @)
import re
import nltk
import string


# In[11]:


import nltk
nltk.download('stopwords')


# In[12]:


from nltk.corpus import stopwords
stopwords = stopwords.words("english")


# In[13]:


stemmer = nltk.SnowballStemmer("english")


# In[14]:


#data cleaning 
def clean(text):
    text = str(text).lower()
    text = re.sub('https?://\+-|www.Smh.', ' ', text)
    text = re.sub('\.img[#*&]',' ',text)
    text = re.sub('[%s]'%re.escape(string.punctuation),' ',text)
    text = re.sub('\n','',text)
    #stopwords removal
    text={word for word in text.split(' ') if word not in stopwords}
    text = ' '.join(text)
    #stemming
    text = {stemmer.stem(word) for word in text.split(' ')}
    text= ' '.join(text)
    return text


# In[15]:


data["tweet"] = data["tweet"].apply(clean)


# In[16]:


data


# In[17]:


data.tail(10)


# In[18]:


data


# In[19]:


#storing the clean data in np arrays
X = np.array(data["tweet"])
Y = np.array(data["labels"])


# In[20]:


Y


# In[21]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# In[22]:


cv = CountVectorizer()
X = cv.fit_transform(X)


# In[23]:


X


# In[24]:


X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.33,random_state=42)


# In[25]:


Y_train


# In[26]:


X_train


# In[27]:


X_test


# In[28]:


#machine learning model intiation
from sklearn.tree import DecisionTreeClassifier


# In[29]:


dt = DecisionTreeClassifier()
dt.fit(X_train,Y_train)


# In[30]:


Y_pred = dt.predict(X_test)


# In[31]:


#confusion matrix and accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
cm


# In[32]:


import seaborn as sns
import matplotlib.pyplot as ply
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


sns.heatmap(cm, annot = True, fmt = ".1f",cmap="YlGnBu" )


# In[34]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_pred)


# In[35]:


sample = " bitch i am stylist black shirt pink tishirt mother"
sample = clean(sample)
sample


# In[36]:


data1 = cv.transform([sample]).toarray()


# In[37]:


data1


# In[38]:


dt.predict(data1)


# In[41]:


s2= "hello i am saketh %&^$:"
s2=clean(s2)
s2


# In[44]:


data2=cv.transform([s2]).toarray()
data2


# In[46]:


dt.predict(data2)

