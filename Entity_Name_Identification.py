#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import re
import nltk


# In[19]:


data = pd.read_excel(r'F:\DeeP_LearNinG\Assignment\Book2.xlsx')


# In[20]:


data.head(1)


# In[21]:


#Consider only NARRATION column for analysis
fdata = data['NARRATION 2']


# In[22]:


fdata.shape


# In[23]:


#Creating a list of company or entity surname name
a = ['ltd','limited','LTD','LIMITED','PVT','PRIVATE','pvt','private','inc','INC','BANK','bank','SERVICE','service','INDUSTRIES',]


# In[24]:


#Applying logic to identify the company or entity name from the columns
l = []
for j in fdata:
    for i in a:
        if i in j.split():
            l.append(j)
            break


# In[25]:


#Convert result to cave
df = pd.DataFrame (l,columns=['NARRATION 2'])
print (df)


# In[245]:


#Reference save the file
df.to_csv(r'C:\Users\BISWA\Desktop\Project\Uncleansed_data.csv', index = False)


# In[246]:


#Consider clean data to more analysis
clean_data = df['NARRATION 2']


# In[281]:


clean_data.head


# In[248]:


#Using Regular Expression to remove all special charactor
k=[]
for i in clean_data:
    cleanString = re.sub('\W+',' ', i )
    k.append(cleanString)


# In[249]:


#Convert clean data to data frame
df = pd.DataFrame (k,columns=['NARRATION 2'])
print (df)


# In[4]:


#save clean data for our reference
df.to_csv(r'C:\Users\BISWA\Desktop\Project\cleandata_data_1.csv', index = False)


# In[5]:


cdata = pd.read_csv(r'C:\Users\BISWA\Desktop\Project\cleandata_data_1.csv')


# In[6]:


cdata.shape


# In[7]:


col = cdata['NARRATION 2']


# In[8]:


b = []
for i in cdata['NARRATION 2']:
    Name = cdata['NARRATION 2'].str[-30:]
    Name


# In[11]:


Name


# In[12]:


cdata.columns


# In[13]:


cdata['Company name'] = Name


# In[14]:


#Removing integer value from the company name
cdata['Company name'] = cdata['Company name'].str.replace('\d+', '')


# In[28]:


df['Compay Name']= cdata['Company name']


# In[29]:


df


# In[30]:


#Save the final Data with company name
df.to_csv(r'C:\Users\BISWA\Desktop\Project\Entity_Name.csv', index = False)


# # Similar way we can do for individual name extraction.
# 1.	Collect Indian surname from Google.
# 2.	Create a list with all the surname.
# 3.	Iterate the list in to our NARRATION column.
# 4.	If the surname is available in the column it will consider for further analysis.
# 5.	After filter coluns from the raw day we need to follow the same method how we used to extract the entity name.

# # Using nltk

# In[287]:


import spacy


# In[310]:


nlp = spacy.load('en_core_web_sm')


# In[311]:


def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
    else:
        print("No Entity found")


# In[324]:


doc = nlp('$ $ $ - $ * $ *P $ 83604383? SERVEL INDIA PRIVATE LIMITED } $')


# In[329]:


print(doc[12])


# In[337]:


show_ents(doc)


# # This library consider only English language for recognise the name  and big Organisation in the world so we can add our entity in one time to our code which will help to identify  in upcoming dataset.

# In[319]:


from spacy.tokens import Span


# In[332]:


#Adding our entity name in span 
ORG=doc.vocab.strings[u"ORG"]
new_ent= Span(doc,12,15,label=ORG)
doc.ents = list(doc.ents)+[new_ent]


# In[336]:


show_ents(doc)


# # We need collect all indian company and entity name to add into span 
# after that spacy library will work fine for next dataset.

# In[ ]:




