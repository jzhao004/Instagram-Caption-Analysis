#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import numpy as np
import re
from datetime import datetime
import emoji
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


# In[6]:


# Import datasets
with open('jjunqi_20191218_part_2\media.json', encoding="utf8") as file1:
    data1 = json.loads(file1.read())

with open('jjunqi_20191218_part_3\media.json', encoding="utf8") as file2:
    data2 = json.loads(file2.read())


# In[11]:


# Merge datasets and remove unnecessary variables
data = [{k:j[k] for k in {'caption', 'taken_at'}} for j in data1['photos']] + [{k:j[k] for k in {'caption', 'taken_at'}} for j in data1['videos']] + [{k:j[k] for k in {'caption', 'taken_at'}} for j in data2['photos']] + [{k:j[k] for k in {'caption', 'taken_at'}} for j in data2['videos']]
data = pd.DataFrame(data)
data = data.drop_duplicates()

data = data.reset_index(drop=True)
data.head()


# In[12]:


# Reformat taken_at
data['taken_at']=[datetime.strptime(re.sub('T',' ', taken_at), '%Y-%m-%d %H:%M:%S') for taken_at in data['taken_at']]


# In[13]:


# Extract year, month, day, and time from taken_at
data['year']=data['taken_at'].dt.year
data['month']=data['taken_at'].dt.month
data['day']=data['taken_at'].dt.day
data['time']=data['taken_at'].dt.time


# In[14]:


# Extract posts posted between 2014 and 2019
data = data[data['taken_at'].dt.year.isin(range(2014,2020))]
print('The data contains information on', data.shape[0], 'posts')


# In[15]:


# Convert captions to lower case
data['caption'] = [caption.lower() for caption in data['caption']]


# In[16]:


# Remove parts of captions that are reposted
data['caption'] = [''.join(caption.split('#repost', 1)[0]) for caption in data['caption']]
data['caption'] = [''.join(caption.split('#repost', 1)[0]) for caption in data['caption']]

data['caption'] = [''.join(caption.split('#incomeorangeaid', 1)[0]) for caption in data['caption']]


# In[17]:


# Create variable to indicate if post contains caption
data['iscaption'] =  [caption is not '' for caption in data['caption']]


# In[18]:


# Extract emojis from captions
data['emojis'] = [' '.join(re.findall('(:[a-zA-Z_-]+(?:[a-zA-Z]):)', emoji.demojize(caption))) for caption in data['caption']]
data['caption'] = [re.sub('(:[a-zA-Z_-]+(?:[a-zA-Z]):)', '', emoji.demojize(caption)) for caption in data['caption']]


# In[19]:


# Extract hashtags and mentions from captions
data['hashtags'] = [' '.join([word for word in caption.split() if word.startswith('#')]) for caption in data['caption']]
data['mentions'] = [' '.join([word for word in caption.split() if word.startswith('@')]) for caption in data['caption']]
data['caption'] = [' '.join([word for word in caption.split() if not (word.startswith('#') | word.startswith('@'))]) for caption in data['caption']]


# In[20]:


# Extract chinese characters from captions
data['caption_ch'] = [' '.join(re.findall('[\u4e00-\u9FFF]', caption)) for caption in data['caption']]
data['caption'] = [re.sub('[\u4e00-\u9FFF]', '', caption) for caption in data['caption']]
data = data.rename(columns={'caption': 'caption_en'})


# In[22]:


# View data
data


# In[23]:


# Data Analysis
# Plot no. of Instagram posts by year
nposts_year = data['year'].value_counts()

fig = plt.figure(figsize=(6,4))
ax = plt.axes()

year = nposts_year.index
nposts = nposts_year.values
ax.bar(year,nposts)
ax.axhline(nposts.mean(), color='grey', linewidth=1)

ax.set_title('No. of Instagram posts by year')
ax.set_ylabel('No. of posts')
ax.set_xticks(year)

plt.show()


# In[25]:


# Plot no. of Instagram posts by month
nposts_year_month = data.pivot_table(values='taken_at', index='month', columns='year', aggfunc='count', fill_value=0)

fig = plt.figure(figsize=(10,4))
plt.suptitle('No. of Instagram posts by month between 2014-2019')

ax1 = plt.subplot(1,2,1)
ax1.plot(nposts_year_month.index, nposts_year_month.mean(axis=1))

ax1.set_xlabel('Month')
ax1.set_ylabel('Average no. of posts')
ax1.set_xticks(nposts_year_month.index)
ax1.set_ylim(0,50)

ax2 = plt.subplot(1,2,2)
ax2.plot(nposts_year_month)
ax2.legend(nposts_year_month.columns)

ax2.set_xlabel('Month')
ax2.set_ylabel('No. of posts')
ax2.set_xticks(nposts_year_month.index)
ax2.set_ylim(0,50)

plt.show()


# In[26]:


# Create variables for no. of words (omitting Chinese characters), hashtags, and emojis in captions
data['nwords'] = [len(caption.split()) for caption in data['caption_en']]
data['nhashtags'] = [len(hashtag.split()) for hashtag in data['hashtags']]
data['nemojis'] = [len(emoji.split()) for emoji in data['emojis']]


# In[27]:


# Find % of posts with captions
nposts = data.shape[0]
nposts_caption = sum(data['iscaption'])

print(round(nposts_caption*100/nposts,2), '% of posts contain captions.')


# In[28]:


fig = plt.figure(figsize=(6,4))
ax = plt.axes()

nposts_caption_year = data.groupby('year')['iscaption'].sum()
nposts_nocaption_year = data.groupby('year')['iscaption'].count() - data.groupby('year')['iscaption'].sum()

ax1 = plt.bar(nposts_caption_year.index, nposts_caption_year.values)
ax2 = plt.bar(nposts_nocaption_year.index, nposts_nocaption_year.values, bottom=nposts_caption_year.values)

ax.set_title('No. of posts with/without caption by year ')
ax.set_xlabel('Year')
ax.set_ylabel('No. of posts')
ax.legend((ax1[0], ax2[0]), ('Caption', 'No Caption'))

plt.show()


# In[29]:


# Plot distribution of caption lengths (in words)

fig = plt.figure(figsize=(6,4))
ax = plt.axes()

len_caption = data[data['iscaption']]['nwords'].value_counts()
ax.bar(len_caption.index,len_caption.values)

ax.set_title('Distribution of caption lengths (in words)')
ax.set_xlabel('No. of words')
ax.set_ylabel('No. of posts')

plt.show()


# In[30]:


# Plot average caption lengths (in words) by year
fig = plt.figure(figsize=(6,4))
ax = plt.axes()

mlen_caption_year = data[data['iscaption']].groupby(['year'])['nwords'].mean()
ax.bar(mlen_caption_year.index,mlen_caption_year.values)

ax.set_title('Average caption lengths (in words) by year')
ax.set_xlabel('Year')
ax.set_ylabel('No. of words')

plt.show()


# In[31]:


# Find % of captions containing emojis and hashtags
nposts_caption_emojis = sum(np.array(data['emojis']) != '')
nposts_caption_hashtags = sum(np.array(data['hashtags']) != '')
nposts_caption_mentions = sum(np.array(data['mentions']) != '')

print(round(nposts_caption_emojis*100/nposts_caption,2),'% of captions contain emojis and',
      round(nposts_caption_hashtags*100/nposts_caption,2),'% of captions contain hashtags')


# In[32]:


# Plot distribution of emojis and hashtags in captions
fig = plt.figure(figsize=(12,4))
plt.suptitle('Distribution of emojis, hashtags, and mentions in captions')

ax1 = plt.subplot(1,2,1)
nemojis_caption = data[data['iscaption']]['nemojis'].value_counts()
ax1.bar(nemojis_caption.index, nemojis_caption.values)

ax1.set_xlabel('No. of emojis')
ax1.set_ylabel('No. of posts')
ax1.set_xticks(nemojis_caption.index)

ax2 = plt.subplot(1,2,2)
nhashtags_caption = data[data['iscaption']]['nhashtags'].value_counts()
ax2.bar(nhashtags_caption.index, nhashtags_caption.values)

ax2.set_xlabel('No. of hashtags')
ax2.set_ylabel('No. of posts')
ax2.set_xticks(nhashtags_caption.index)

plt.show()


# In[33]:


# Plot average no. of emojis and hashtags in captions by year
fig = plt.figure(figsize=(12,4))
plt.suptitle('Average no. of emojis and hashtags in caption by year')

ax1 = plt.subplot(1,2,1)

mnemojis_caption_year = data[data['iscaption']].groupby(['year'])['nemojis'].mean()
ax1.bar(mnemojis_caption_year.index,mnemojis_caption_year.values)

ax1.set_xlabel('Year')
ax1.set_ylabel('Average no. of emojis')

ax2 = plt.subplot(1,2,2)

mnhashtags_caption_year = data[data['iscaption']].groupby(['year'])['nhashtags'].mean()
ax2.bar(mnhashtags_caption_year.index,mnhashtags_caption_year.values)

ax2.set_xlabel('Year')
ax2.set_ylabel('Average no. of hashtags')

plt.show()


# In[34]:


# Find the most used emojis in captions
emojis_caption = pd.Series(" ".join(data['emojis'][data['emojis']!='']).split())
top_emojis_caption = emojis_caption.value_counts().nlargest(5)
top_emojis_caption = pd.DataFrame({'Emoji': [emoji.emojize(x) for x in list(top_emojis_caption.index)],
                                   "Freq (%)": (top_emojis_caption.values*100/emojis_caption.count()).round(2)}).set_index(pd.Series([1,2,3,4,5]))
top_emojis_caption


# In[35]:


# Find the most used emojis in captions by year
top_emojis_caption_year = {}
for year in sorted(set(data['year'])):
    emojis_caption_year = pd.Series(" ".join(data[data['year']==year]['emojis'][data['emojis']!='']).split())
    top_emojis_caption = emojis_caption_year.value_counts().nlargest(5)
    top_emojis_caption_year.update({year: [emoji.emojize(x) for x in list(top_emojis_caption.index)]})

top_emojis_caption_year = pd.DataFrame(top_emojis_caption_year).set_index(pd.Series([1,2,3,4,5]))
top_emojis_caption_year


# In[36]:


# Find most used words in captions
# Prepare caption data
words_caption = data['caption_en'][data['caption_en']!='']
words_caption = [re.sub(r'[^a-z\s]+', ' ',caption) for caption in words_caption] # Remove numbers and punctuation
words_caption = [re.sub(r'\b[a-z]\b', ' ',caption) for caption in words_caption] # Remove single characters
words_caption = [caption.strip() for caption in words_caption] # Remove spaces at the start and end of strings
words_caption = ' '.join(words_caption) # Merge captions
words_caption = re.sub(' +', ' ',words_caption) # Remove extra spaces


# In[37]:


# Plot wordcloud
wordcloud = WordCloud(max_words=100, width=800, height=800, background_color='white', stopwords=STOPWORDS).generate(words_caption)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

plt.show()


# In[38]:


# Find most used words in captions by year
# Prepare caption data
def words_caption_wordcloud(year):
    words_caption = data['caption_en'][data['caption_en']!=''][data['year']==year]
    words_caption = [re.sub(r'[^a-z\s]+', ' ',caption) for caption in words_caption] # Remove numbers and punctuation
    words_caption = [re.sub(r'\b[a-z]\b', ' ',caption) for caption in words_caption] # Remove single characters
    words_caption = [caption.strip() for caption in words_caption] # Remove spaces at the start and end of strings
    words_caption = ' '.join(words_caption) # Merge captions
    words_caption = re.sub(' +', ' ',words_caption) # Remove extra spaces

    wordcloud = WordCloud(max_words=50, width=600, height=600, background_color='white', stopwords=STOPWORDS).generate(words_caption)
    return wordcloud


# In[39]:


# Plot wordclouds
fig = plt.figure(figsize=(18, 12))
index = 1

for year in sorted(set(data['year'])):
    ax = fig.add_subplot(2,3,index)
    index = index + 1
    wordcloud = words_caption_wordcloud(year)

    ax.imshow(wordcloud)
    ax.set_title(str(year))

    ax.axis('off')


# In[ ]:
