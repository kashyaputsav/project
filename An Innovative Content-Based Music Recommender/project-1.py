#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


music = pd.read_csv('rym_top_5000_all_time.csv')


# In[3]:


music.head()


# 

# In[4]:


music.head(1)


# In[5]:


music.head(10)['Descriptors']


# In[6]:


# Ranking 
# Album 
# Artist name 
# Geners 
# Average Rating
# we taken these upper tags as keywords for predication of songs.


# In[7]:


music = music[['Ranking', 'Album','Artist Name', 'Genres','Average Rating', 'Descriptors']]


# In[8]:


music.info()


# In[9]:


music.head()
# This is main datas that we have to worked


# In[10]:


music.isnull().sum()


# In[11]:


music.dropna(inplace= True)


# In[12]:


music.duplicated().sum()


# In[13]:


music.head()


# In[ ]:





# In[14]:


def convert(obj):
    l = []
    for i in(obj):
        li = list(obj.split(","))
    return li    


# In[15]:


music.head()


# In[16]:


new_df2 = music.head()


# In[17]:


#I convert this float value into string value
music['Average Rating'] = music['Average Rating'].astype(str)
  
print()
  
# lets find out the data
# type after changing
print(music.dtypes)
  
# print dataframe. 
music


# In[18]:


music['Genres'] = music['Genres'].apply(convert)


# In[19]:


#music['Album'] = music['Album'].apply(convert)


# In[20]:


music['Artist Name'] = music['Artist Name'].apply(convert)


# In[21]:


music['Descriptors'] = music['Descriptors'].apply(convert)


# In[22]:


# now i convert all string value of column into list so that i concatinate all value 
# to use as a tag.
music['Average Rating'] = music['Average Rating'].apply(convert)


# In[23]:


#music['Album'] = music['Album'].apply(lambda x:[i.replace(" ","") for i in x])


# In[24]:


music['Genres'] = music['Genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[25]:


music['Artist Name'] = music['Artist Name'].apply(lambda x:[i.replace(" ","") for i in x])


# In[26]:


music['Descriptors'] = music['Descriptors'].apply(lambda x:[i.replace(" ","") for i in x])


# In[ ]:





# In[27]:


music['tags'] = music['Artist Name'] + music['Genres'] + music['Average Rating']  + music['Descriptors']


# In[28]:


new_df = music[[] ]


# In[29]:


#This is tag value that i created for filter songs.
music['tags'].head(1)


# In[30]:


#after concatination i changed tags list into string
new_df = music[['Album','tags']]


# In[31]:


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))


# In[32]:


new_df.head()


# In[33]:


new_df.head(1)


# In[34]:


#convert all into lower codes
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


# In[35]:


new_df.head(2)


# In[36]:


# now I use vectorisation concept to detrmine reltion by 
#vector distance between tags


# In[37]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000,stop_words= 'english')


# In[38]:


import nltk


# In[39]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[40]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return" ".join(y)


# In[41]:


#stem library example
['loved', 'love', 'loving']
ps.stem('loving')


# In[42]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[43]:


#to convert these array to matrix
vector = cv.fit_transform(new_df['tags']).toarray()


# In[44]:


cv.fit_transform(new_df['tags']).toarray().shape


# In[45]:


vector


# In[46]:


cv.get_feature_names()


# In[47]:


from sklearn.metrics.pairwise import cosine_similarity


# In[48]:


similarity = cosine_similarity(vector)


# In[49]:


similarity[0]


# In[50]:


# with the help of this code we fetch five albums which is very similar
# our slected album 
sorted(list(enumerate(similarity[0])),reverse = True, key= lambda x:x[1])[1:6]


# In[51]:


#new_df[new_df['Album'] == 'Kid A'].index[0]


# In[52]:


def recommend(music):
    music_index = new_df[new_df['Album'] == music].index[0]
    distances = similarity[music_index]
    music_list = sorted(list(enumerate(distances)),reverse = True, key= lambda x:x[1])[1:6]
    
    for i in music_list:
        print(new_df.iloc[i[0]].Album)


# In[53]:


recommend('OK Computer')


# In[ ]:





# In[ ]:




