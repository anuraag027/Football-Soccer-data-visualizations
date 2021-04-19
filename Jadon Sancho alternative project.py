#!/usr/bin/env python
# coding: utf-8

# In[743]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[744]:


#Read the complete dataset
df = pd.read_csv('D:\Data Analysis TP\Jadon Sancho alternate project\FBRef Data.csv')


# In[745]:


df.drop(['Rk','Matches','OG'],axis=1,inplace=True)


# In[746]:


#Get the name from Player column
df['Player'] = df['Player'].apply(lambda x: x.split('\\')[0])
og_df = df.copy()


# In[747]:


#Remove the text columns so only numeric columns remain. This is to later scale the fully numeric DF 
df.drop(df.iloc[:, :8], inplace = True, axis = 1)


# In[748]:


#All data values to be converted to numeric from strings
df[:] = df[:].apply(pd.to_numeric)


# In[749]:


#Import model for scaling values and scale the DF
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)


# In[750]:


#Get the scaled values
scaled_features = scaler.transform(df)


# In[751]:


#Create a new DF with scaled values and assign column names from the main DF
df_feat = pd.DataFrame(scaled_features,columns=df.columns)


# In[752]:


#Create K-Means model
km = KMeans(n_clusters=10)


# In[753]:


#Drop the last row from DF which has null values
df_feat.drop(df_feat.tail(1).index,inplace=True) # drop last row


# In[754]:


#Fit the new DF with scaled values to the model
km.fit(df_feat)


# In[755]:


#Drop last row "Salvador Ferrer" as there is no data (null values) for him. Confirmed from FBRef
#Now number of DF rows should match K-Means model labels
og_df.drop(og_df.tail(1).index,inplace=True)


# In[756]:


#Create a new column called Cluster and assign the predicted label values to it
og_df['Cluster'] = km.labels_


# In[757]:


#Check the cluster assigned to Jadon Sancho
og_df[og_df['Player'] == 'Jadon Sancho']


# In[758]:


#Get only year from age from original DF and change it to numeric
og_df['Age'] = og_df['Age'].apply(lambda x: x.split('-')[0])
og_df['Age'] = og_df['Age'].apply(pd.to_numeric)


# In[795]:


#Create new DF with same cluster as Sancho, Position as FWMD or MFFW and age under 25
newdf = og_df[(og_df['Cluster'] == 3) & ((og_df['Pos'] == 'FWMF')|(og_df['Pos'] == 'MFFW')) & (og_df['Age'] <= 25)]

#Create figure
f, ax = plt.subplots(figsize=(13,10))
ax.set_title('K Means')

#Plot Shot Creating Actions per 90 versus Goal Creating Actions per 90
sns.scatterplot(newdf['SCA90'],newdf['GCA90▼'],alpha=0.8)

#Reset the index
newdf.reset_index(inplace=True)
newdf.drop(['index'],axis=1,inplace=True)

#Set the labels for X and Y axes and set the title
ax.set(xlabel = 'Shot Creating Actions per 90', ylabel = 'Goal Creating Actions per 90',title='Using K-Means Clustering to find U-25 players (Top 5 Leagues only) similar to Jadon Sancho')

#Limit figure size
plt.xlim(newdf['SCA90'].min()-0.45,newdf['SCA90'].max()+0.42)
plt.ylim(newdf['GCA90▼'].min()-0.1,newdf['GCA90▼'].max()+0.1)

#Set x axis ticks
plt.xticks(np.arange(2,7,1))

#Add the annotations to show player names in plot
for i in range(len(newdf['Cluster'])):
    if newdf['Player'][i] == 'Mikel Oyarzabal':
        ax.text(newdf['SCA90'][i], newdf['GCA90▼'][i]-0.02,newdf['Player'][i], horizontalalignment='center',size='medium', color='black')
    else:
        ax.text(newdf['SCA90'][i], newdf['GCA90▼'][i]+0.015,newdf['Player'][i], horizontalalignment='center',size='medium', color='black')

ax.text(1.9,0.27,"Twitter: Anuraag@027 | Data: Statsbomb via FBRef",size=11,fontfamily='Calibri')
plt.savefig("D:\Data Analysis TP\similar_to_Sancho.png",dpi=600)


# In[815]:


#Plot all players and annotate the players with same cluster as Sancho

f, ax = plt.subplots(figsize=(13,10))
ax.set_title('K Means')
sns.scatterplot(og_df['SCA90'],og_df['GCA90▼'],alpha=0.8,hue=og_df['Cluster'],palette='coolwarm')

ax.set(xlabel = 'Shot Creating Actions per 90', ylabel = 'Goal Creating Actions per 90',title='Using K-Means Clustering to find players (Top 5 Leagues only) similar to Jadon Sancho - all positions')

for i in range(len(og_df['Cluster'])):
    if (og_df['Cluster'][i] == 3):
        ax.text(og_df['SCA90'][i], og_df['GCA90▼'][i]-0.02,og_df['Player'][i], horizontalalignment='center',size='medium', color='black',weight='semibold')

#Set x axis ticks
plt.xticks(np.arange(0,8.5,1))
#Set y axis ticks
plt.yticks(np.arange(0,1.4,0.2))        
        
ax.text(-0.4,-0.2,"Twitter: Anuraag@027 | Data: Statsbomb via FBRef",size=12,fontfamily='Calibri')
plt.savefig("D:\Data Analysis TP\similar_to_Sancho_all_players.png",dpi=600)


# In[812]:


#PLOT FOR OPEN PLAY PASSES VS DRIBBLES LEADING TO SHOTS

#Create figure
f, ax = plt.subplots(figsize=(13,10))
ax.set_title('K Means')

#Plot Shot Creating Actions per 90 versus Goal Creating Actions per 90
sns.scatterplot(newdf['PassLive'],newdf['Drib'],alpha=0.8)

#Set the labels for X and Y axes and set the title
ax.set(xlabel = 'Open Play Passes Leading To Shots per 90', ylabel = 'Successful Dribbles Leading to Shots per 90',title='Using K-Means Clustering to find U-25 players (Top 5 Leagues only) similar to Jadon Sancho')

#Limit figure size
plt.xlim(newdf['PassLive'].min()-0.45,newdf['PassLive'].max()+0.42)
plt.ylim(newdf['Drib'].min()-0.1,newdf['Drib'].max()+0.1)

#Set x axis ticks
plt.xticks(np.arange(0,5.5,0.5))
#Set y axis ticks
plt.yticks(np.arange(0,0.9,0.1))

#Add the annotations to show player names in plot
for i in range(len(newdf['Cluster'])):
    if newdf['Player'][i] == 'Mikel Oyarzabal':
        ax.text(newdf['PassLive'][i], newdf['Drib'][i]-0.02,newdf['Player'][i], horizontalalignment='center',size='medium', color='black')
    else:
        ax.text(newdf['PassLive'][i], newdf['Drib'][i]+0.015,newdf['Player'][i], horizontalalignment='center',size='medium', color='black')

ax.text(-0.16,-0.1,"Twitter: Anuraag@027 | Data: Statsbomb via FBRef",size=12,fontfamily='Calibri')
plt.savefig("D:\Data Analysis TP\similar_to_Sancho_2.png",dpi=600)

