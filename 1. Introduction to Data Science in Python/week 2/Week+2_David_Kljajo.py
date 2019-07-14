
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # The Series Data Structure

# In[1]:


import pandas as pd
get_ipython().magic('pinfo pd.Series')


# In[2]:


animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals)


# In[3]:


numbers = [1, 2, 3]
pd.Series(numbers)


# In[4]:


animals = ['Tiger', 'Bear', None]
pd.Series(animals)


# In[5]:


numbers = [1, 2, None]
pd.Series(numbers)


# In[6]:


import numpy as np
np.nan == None


# In[7]:


np.nan == np.nan


# In[8]:


np.isnan(np.nan)


# In[9]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s


# In[10]:


s.index


# In[11]:


s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])
s


# In[12]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])
s


# # Querying a Series

# In[13]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s


# In[14]:


s.iloc[3]


# In[15]:


s.loc['Golf']


# In[16]:


s[3]


# In[17]:


s['Golf']


# In[18]:


sports = {99: 'Bhutan',
          100: 'Scotland',
          101: 'Japan',
          102: 'South Korea'}
s = pd.Series(sports)


# In[19]:


s[0] #This won't call s.iloc[0] as one might expect, it generates an error instead


# In[20]:


s = pd.Series([100.00, 120.00, 101.00, 3.00])
s


# In[21]:


total = 0
for item in s:
    total+=item
print(total)


# In[22]:


import numpy as np

total = np.sum(s)
print(total)


# In[23]:


#this creates a big series of random numbers
s = pd.Series(np.random.randint(0,1000,10000))
s.head()


# In[24]:


len(s)


# In[25]:


get_ipython().run_cell_magic('timeit', '-n 100', 'summary = 0\nfor item in s:\n    summary+=item')


# In[26]:


get_ipython().run_cell_magic('timeit', '-n 100', 'summary = np.sum(s)')


# In[27]:


s+=2 #adds two to each item in s using broadcasting
s.head()


# In[28]:


for label, value in s.iteritems():
    s.set_value(label, value+2)
s.head()


# In[29]:


get_ipython().run_cell_magic('timeit', '-n 10', 's = pd.Series(np.random.randint(0,1000,10000))\nfor label, value in s.iteritems():\n    s.loc[label]= value+2')


# In[30]:


get_ipython().run_cell_magic('timeit', '-n 10', 's = pd.Series(np.random.randint(0,1000,10000))\ns+=2')


# In[31]:


s = pd.Series([1, 2, 3])
s.loc['Animal'] = 'Bears'
s


# In[33]:


original_sports = pd.Series({'Archery': 'Bhutan',
                             'Golf': 'Scotland',
                             'Sumo': 'Japan',
                             'Taekwondo': 'South Korea'})
cricket_loving_countries = pd.Series(['Australia',
                                      'Barbados',
                                      'Pakistan',
                                      'England'], 
                                   index=['Cricket',
                                          'Cricket',
                                          'Cricket',
                                          'Cricket'])
all_countries = original_sports.append(cricket_loving_countries)


# In[34]:


original_sports


# In[35]:


cricket_loving_countries


# In[36]:


all_countries


# In[37]:


all_countries.loc['Cricket']


# # The DataFrame Data Structure

# In[38]:


import pandas as pd
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df.head()


# In[39]:


df.loc['Store 2']


# In[40]:


type(df.loc['Store 2'])


# In[41]:


df.loc['Store 1']


# In[42]:


df.loc['Store 1', 'Cost']


# In[43]:


df.T


# In[44]:


df.T.loc['Cost']


# In[45]:


df['Cost']


# In[46]:


df.loc['Store 1']['Cost']


# In[47]:


df.loc[:,['Name', 'Cost']]


# In[48]:


df.drop('Store 1')


# In[49]:


df


# In[50]:


copy_df = df.copy()
copy_df = copy_df.drop('Store 1')
copy_df


# In[51]:


get_ipython().magic('pinfo copy_df.drop')


# In[52]:


del copy_df['Name']
copy_df


# In[53]:


df['Location'] = None
df


# # Dataframe Indexing and Loading

# In[54]:


costs = df['Cost']
costs


# In[55]:


costs+=2
costs


# In[56]:


df


# In[57]:


get_ipython().system('cat olympics.csv')


# In[58]:


df = pd.read_csv('olympics.csv')
df.head()


# In[59]:


df = pd.read_csv('olympics.csv', index_col = 0, skiprows=1)
df.head()


# In[60]:


df.columns


# In[61]:


for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver' + col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#' + col[1:]}, inplace=True) 

df.head()


# # Querying a DataFrame

# In[62]:


df['Gold'] > 0


# In[63]:


only_gold = df.where(df['Gold'] > 0)
only_gold.head()


# In[64]:


only_gold['Gold'].count()


# In[65]:


df['Gold'].count()


# In[66]:


only_gold = only_gold.dropna()
only_gold.head()


# In[67]:


only_gold = df[df['Gold'] > 0]
only_gold.head()


# In[68]:


len(df[(df['Gold'] > 0) | (df['Gold.1'] > 0)])


# In[69]:


df[(df['Gold.1'] > 0) & (df['Gold'] == 0)]


# # Indexing Dataframes

# In[70]:


df.head()


# In[71]:


df['country'] = df.index
df = df.set_index('Gold')
df.head()


# In[72]:


df = df.reset_index()
df.head()


# In[73]:


df = pd.read_csv('census.csv')
df.head()


# In[74]:


df['SUMLEV'].unique()


# In[75]:


df=df[df['SUMLEV'] == 50]
df.head()


# In[ ]:


columns_to_keep = ['STNAME',
                   'CTYNAME',
                   'BIRTHS2010',
                   'BIRTHS2011',
                   'BIRTHS2012',
                   'BIRTHS2013',
                   'BIRTHS2014',
                   'BIRTHS2015',
                   'POPESTIMATE2010',
                   'POPESTIMATE2011',
                   'POPESTIMATE2012',
                   'POPESTIMATE2013',
                   'POPESTIMATE2014',
                   'POPESTIMATE2015']
df = df[columns_to_keep]
df.head()


# In[76]:


df = df.set_index(['STNAME', 'CTYNAME'])
df.head()


# In[77]:


df.loc['Michigan', 'Washtenaw County']


# In[78]:


df.loc[ [('Michigan', 'Washtenaw County'),
         ('Michigan', 'Wayne County')] ]


# # Missing values

# In[79]:


df = pd.read_csv('log.csv')
df


# In[80]:


get_ipython().magic('pinfo df.fillna')


# In[81]:


df = df.set_index('time')
df = df.sort_index()
df


# In[82]:


df = df.reset_index()
df = df.set_index(['time', 'user'])
df


# In[83]:


df = df.fillna(method='ffill')
df.head()


# In[ ]:




