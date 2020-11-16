#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the necessary library
import pandas as pd
import numpy as np
import re
import nltk


# In[2]:


data = pd.read_excel(r'F:\DeeP_LearNinG\Assignment\Book2.xlsx')


# In[3]:


data.head(4)


# In[4]:


#Consider only NARRATION 2 column for analysis.
Analysis_Column = data['NARRATION 2']


# In[5]:


Analysis_Column.shape


# In[6]:


#Creating list of company & entity last string.
a = ['ltd','limited','LTD','LIMITED','PVT','PRIVATE','pvt','private','inc','INC','BANK','bank','SERVICE','service','INDUSTRIES','LLP','TRADERS',
     'HOSPITAL','TRADERS','SOFTWAR','SOLUTION','PROPETITES','ENTERPRISES','DISTRIBUTORS','CORPORATION','CONSULTING','COMPUTERS',
    'FOUNDATION','PHARMECY','TRADE','AGENCIES','MEASUREMENT','FENCING','SYSTEMS','ENTERP','Outsources','infotech','Computers','BROADBAND']


# In[7]:


#Applying logic to filter the company or entity name from the columns.
l = []
for j in Analysis_Column:
    for i in a:
        if i in j.split():
            l.append(j)
            break


# In[8]:


#Copy of Original Dataset.
Raw_Data = pd.DataFrame (l,columns=['NARRATION 2'])


# In[9]:


#Convert filter result to data frame
df = pd.DataFrame (l,columns=['NARRATION 2'])


# In[10]:


#As per observation collecting stopwords from dataset.
stopwords = ["/chclchl","/gai $ ici1","/chclchl","/chlccl","/cplchl","/gailchcl","CHA ","$ -SAS","$ PCA:","ABB ","NUN","DC/ ","/P2A/ ","OE P ","OSS ","CDCC1","/NT $","_  ","02 ", "16 ", "18:","19:","CC 06","CA-","CC-","P2A $"
            "48/","AS-","CTF2","CC $","CC1","CC2","CMS/","CPLCHCL","$ CTD",'$ CTE','$ CTF','l NF/',"IB:","IB-","IB $","/c","5c","$ fm","MD $","/chclchl","08/02","and Investments-","E60","$   27 ","$ -SASH","/ORBC/","1/GAI $"," $ P2A $",
            "l NF/","9K2E","8K","N2","*P","_ $","SSI $ SHI","/ICICICPLCHCL","KT1","CC Int $"," $ NYK","SHI $ SSI-","$ 16:","/48/","1 IT-"," $ / P2A /","$ /SK/"]
    
#Removing stopwords from the dataset.
for k in stopwords:
    for index, row in df.iterrows():
        if k in df.loc[index, 'NARRATION 2']:
            df.loc[index,'NARRATION 2'] = df.loc[index,'NARRATION 2'].replace(k, " ")


# In[11]:


#Reference save the file
#df.to_csv(r'C:\Users\BISWA\Desktop\Project\Uncleansed_data.csv', index = False)


# In[12]:


#Consider filter data to more analysis
Filter_data = df['NARRATION 2']


# In[13]:


Filter_data.head


# In[14]:


#Using Regular Expression to remove all special charactors.
k=[]
for i in Filter_data:
    cleanString = re.sub('\W+',' ', i )
    k.append(cleanString)


# In[15]:


#Convert clean data to data frame.
Data = pd.DataFrame (k,columns=['NARRATION 2'])


# In[16]:


#Create a function to get the length of integer value count from data frame.
def count_digits(count):
    return sum(item.isdigit() for item in count)

Data['Count']= Data['NARRATION 2'].apply(count_digits) 


# In[17]:


Data.columns


# In[18]:


#Delete integer value when more than 2 digit.
Data.loc[Data['Count'] >2, 'NARRATION 2'] = Data['NARRATION 2'].str.replace('\d+', '')


# In[19]:


#Consider only company and entity name details. 
Data['NARRATION 2'] = Data['NARRATION 2'].str.split('LIMITED').str[0] + 'LIMITED'


# In[20]:


Data.columns


# In[21]:


#From data frame Drop count column.
Data. drop(['Count'], axis=1, inplace=True)


# In[22]:


#save clean data for more Analysis
Data.to_csv(r'C:\Users\BISWA\Desktop\Project\Clean_Data.csv', index = False)


# In[23]:


Cleandata = pd.read_csv(r'C:\Users\BISWA\Desktop\Project\Clean_Data.csv')


# In[24]:


Cleandata.shape


# In[25]:


Cleandata.columns


# In[26]:


#Removing white space from data Frame.
Cleandata = Cleandata.applymap(lambda x: x.strip() if isinstance(x, str) else x)


# In[27]:


Cleandata['NARRATION 2']


# In[28]:


#Adding extract entity name column into original data frame.
Raw_Data['Compay & Entity Names']= Cleandata['NARRATION 2']


# In[29]:


#Aliment of the company names.
for index, row in Raw_Data.iterrows():
    if 'LIMITED' in Raw_Data.loc[index, 'NARRATION 2']:
        Raw_Data.loc[index,'Compay & Entity Names'] = Raw_Data.loc[index,'Compay & Entity Names'] 
    else:
        Raw_Data.loc[index,'Compay & Entity Names'] = Raw_Data.loc[index,'Compay & Entity Names'].replace("LIMITED", " ")


# In[30]:


#Save the final result with original Data Frame.
Raw_Data.to_csv(r'C:\Users\BISWA\Desktop\Project\Entity_Name_New.csv', index = False)


# # Similar way we can do for individual name extraction.
# 1.	Collect Indian surname from Google.
# 2.	Create a list with all the surnames.
# 3.	Iterate the list in to our NARRATION column.
# 4.	If the surname is available in the column it will consider for further analysis.
# 5.	After filter coluns from the raw day we need to follow the same method how we used to extract the entity name.

# # Using nltk

# In[31]:


import spacy


# In[32]:


nlp = spacy.load('en_core_web_sm')


# In[33]:


def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
    else:
        print("No Entity found")


# In[34]:


doc = nlp('$ $ $ - $ * $ *P $ 83604383? Google.com } $')


# In[35]:


print(doc[12])


# In[36]:


show_ents(doc)


# # This library consider only English language for recognise the name  and big Organisation in the world so we can add our entity in one time to our code which will help to identify  in upcoming dataset.

# In[37]:


#from spacy.tokens import Span


# In[38]:


#Adding our entity name in span 
ORG=doc.vocab.strings[u"ORG"]
new_ent= Span(doc,12,15,label=ORG)
doc.ents = list(doc.ents)+[new_ent]


# In[39]:


show_ents(doc)


# # We need collect all indian company and entity name to add into span 
# after that spacy library will work fine for next dataset.

# In[ ]:




