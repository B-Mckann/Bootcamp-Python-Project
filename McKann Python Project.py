#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import tools to work with data frame
import pandas as pd
import numpy as np

# convert csv to data frame
df = pd.read_csv('C:\\Users\\britt\\Downloads\\Happiness dataset\\output.csv')


# # Is GDP a good indicator of a high happiness ranking?

# In[ ]:


# pull top 25 happiest countries
top25_df = df.head(25)

# sort by GDP per capita
top25_economysort = top25_df.sort_values(by=['Economy (GDP per Capita)'])
print(top25_economysort[['Country','Happiness Rank','Economy (GDP per Capita)']])


# In[ ]:


# What is the max GDP per capita and what is that countries happiness rank?
econsort = df.sort_values(by=['Economy (GDP per Capita)'], ascending=False)
econsort.head(1)

print(econsort[['Country','Happiness Rank','Economy (GDP per Capita)']].head(25))


# In[ ]:


# Take mean and compare it to the GDP of the top 25
Econmean = df[['Economy (GDP per Capita)']].mean()
Econstd = df[['Economy (GDP per Capita)']].std()

top25_df[['Economy Z_Score25']] = (top25_df[['Economy (GDP per Capita)']] - Econmean) / Econstd

print(top25_df[['Country','Happiness Rank','Economy (GDP per Capita)','Economy Z_Score25']])


# In[ ]:


# Plot the happiness rank by z-score
df[['Economy Z_Score']] = (df[['Economy (GDP per Capita)']] - Econmean) / Econstd

df.plot.scatter(x='Happiness Rank', y='Economy Z_Score')


# Notes:
# There is a trend between GDP per capita and the happiness ranking of a country. Nearly all those in the top 25 are a standard deviation higher than the mean of all the countries measured. From the scatter plot, a correlation is present, but it is not extraordinarily clear as there is still quite a range from the line of best fit.

# # What is the biggest predictor of ranking low in the Happiness Rank?

# In[ ]:


# pull bottom 50
df_copy = df[df['Happiness Rank'].notna()]
bottom50 = df_copy[-50:]
# count number of countries per region
bottom50[['Region','Country']].groupby('Region').count()


# In[ ]:


# compare to number countries per region in the dataset
df[['Region','Country']].groupby('Region').count()


# In[ ]:


# bottom 50 number of countries per region
bottom50_count = pd.DataFrame(bottom50[['Region','Country']].groupby('Region').count())
b50_count = bottom50_count.reset_index()

# countries per region whole list with Happiness rankings
reg_country_count = df[['Region','Country']].groupby('Region').count()
region_country_count = reg_country_count.reset_index()


# In[ ]:


# calculation the proportion of countries in a region in the bottom 50 of happiness
t_dict = {}

for row in b50_count[['Region']].values:
    region = row[0]
    t_dict[region] = b50_count[b50_count['Region'] == region]['Country'].values[0] / region_country_count[region_country_count['Region'] == region]['Country'].values[0]
    


# In[ ]:


# add to b50_count table
b50_count['Proportion of region'] = t_dict.values()
b50_count.rename(columns={'Country':'Country Count'}) 


# In[ ]:


# Look at other factors taken into consideration when calculating the happiness ranking
# build table to add features

b50_factors = bottom50[['Region','Country','Happiness Rank','Health (Life Expectancy)','Freedom','Trust (Government Corruption)']].sort_values(by=['Region'])

print(b50_factors)


# In[ ]:


# Find mean and STD of health
health_mean = df[['Health (Life Expectancy)']].mean()
health_std = df[['Health (Life Expectancy)']].std()

# Find mean and STD of Freedom
free_mean = df[['Freedom']].mean()
free_std = df[['Freedom']].std()

# Find mean and STD of Trust
trust_mean = df[['Trust (Government Corruption)']].mean()
trust_std = df[['Trust (Government Corruption)']].std()


# In[ ]:


# find the z-scores for each feature

# Health z-score
b50_factors[['Health (Life Expectancy) z-score']] = (b50_factors[['Health (Life Expectancy)']] - health_mean) / health_std

# Freedom z-score
b50_factors[['Freedom z-score']] = (b50_factors[['Freedom']] - free_mean) / free_std

# Trust z-score
b50_factors[['Trust (Government Corruption) z-score']] = (b50_factors[['Trust (Government Corruption)']] - trust_mean) / trust_std


# In[ ]:


# View z-scores in table

print(b50_factors[['Region','Country','Health (Life Expectancy) z-score','Freedom z-score','Trust (Government Corruption) z-score']])


# In[ ]:


# Plot scatterplots of bottom 50 countries by Health, Freedom, and Trust of Government
b50_factors.plot.scatter(x='Happiness Rank', y='Health (Life Expectancy) z-score')

b50_factors.plot.scatter(x='Happiness Rank', y='Freedom z-score')

b50_factors.plot.scatter(x='Happiness Rank', y='Trust (Government Corruption) z-score')


# In[ ]:


# Compare it to the whole set of data

# find the z-scores for each feature

# Health z-score
df[['Health (Life Expectancy) z-score']] = (df[['Health (Life Expectancy)']] - health_mean) / health_std

# Freedom z-score
df[['Freedom z-score']] = (df[['Freedom']] - free_mean) / free_std

# Trust z-score
df[['Trust (Government Corruption) z-score']] = (df[['Trust (Government Corruption)']] - trust_mean) / trust_std


# In[ ]:


# plot the whole list for each additional feature
df.plot.scatter(x='Happiness Rank', y='Health (Life Expectancy) z-score')

df.plot.scatter(x='Happiness Rank', y='Freedom z-score')

df.plot.scatter(x='Happiness Rank', y='Trust (Government Corruption) z-score')


# Notes:
# It appears that countries of Sub-Saharan Africa are generally more unhappy than other regions. Upon further examination of features, it is clear that Health (Life Expectancy) holds the strongest correlation. When looking at the last 50 on the Happiness Rank, the plots are generally inconclusive. When the scope is broadened to the whole list, there is a trend in all three categories. Having higher life expectancy and higher freedom, people tend to be happier. Surprisingly, trust/government corruption was relatively average for most of the sample, including the lowest ranking countries in happiness.
