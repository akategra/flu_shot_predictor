#!/usr/bin/env python
# coding: utf-8

# In[228]:


import pandas as pd
import numpy as np


# In[384]:


df = pd.read_csv(r'C:\Users\Akash Memon\Documents/training_set_features.csv')


# In[230]:


df.drop('employment_occupation' , axis=1, inplace=True)


# In[231]:


df.drop('employment_industry' , axis=1, inplace=True)


# In[232]:


age_group = pd.get_dummies(df['age_group'], drop_first=False)
age_group.drop('65+ Years', axis = 1, inplace=True)


# In[233]:


test_data = df.drop('age_group', axis =1 , inplace=True)
test_data = pd.concat([age_group, df], axis =1 )


# In[234]:


df['education'].unique()


# In[235]:


age = pd.get_dummies(df['education'], drop_first=False)


# In[236]:


age.drop('Some College', axis = 1, inplace=True)


# In[237]:


test_data = pd.concat([age, test_data], axis = 1)
test_data.drop('education', axis = 1, inplace = True)


# In[238]:


test_data['race'].unique()


# In[239]:


race = pd.get_dummies(df['race'], drop_first=False)


# In[240]:


race.drop('Other or Multiple', axis = 1, inplace=True)
test_data.drop('race', axis = 1, inplace=True)
test_data = pd.concat([race, test_data], axis =1 )


# In[241]:


sex = pd.get_dummies(df['sex'], drop_first=False)


# In[242]:


test_data.drop('sex', axis = 1, inplace=True)
test_data = pd.concat([sex, test_data], axis =1 )


# In[243]:


income_poverty = pd.get_dummies(df['income_poverty'], drop_first=True)


# In[244]:


test_data.drop('income_poverty', axis = 1, inplace=True)
test_data = pd.concat([income_poverty, test_data], axis =1 )


# In[245]:


test_data['marital_status'].unique()


# In[246]:


marital_status = pd.get_dummies(df['marital_status'], drop_first=False)


# In[247]:


marital_status.drop('Not Married', axis =1 , inplace=True)
test_data.drop('marital_status', axis = 1, inplace=True)
test_data = pd.concat([marital_status, test_data], axis =1 )


# In[248]:


test_data['rent_or_own'].unique()


# In[249]:


rent_or_own = pd.get_dummies(df['rent_or_own'], drop_first=False)


# In[250]:


rent_or_own.drop("Rent", axis =1, inplace=True)


# In[251]:


test_data.drop('rent_or_own', axis = 1, inplace=True)
test_data = pd.concat([rent_or_own, test_data], axis =1 )


# In[252]:


test_data['employment_status'].unique()


# In[253]:


employment_status = pd.get_dummies(df['employment_status'], drop_first=False)


# In[254]:


employment_status.drop('Unemployed', axis =1 , inplace=True)
test_data.drop('employment_status', axis = 1, inplace=True)
test_data = pd.concat([employment_status, test_data], axis =1 )


# In[255]:


test_data.rename(columns = {'Own':'owns_house'}, inplace=True)


# In[256]:


test_data['hhs_geo_region'].unique()


# In[257]:


test_data.drop('hhs_geo_region', axis=1, inplace=True)


# In[258]:


test_data['census_msa'].unique()


# In[259]:


census_msa = pd.get_dummies(df['census_msa'], drop_first=True)


# In[260]:


test_data.drop('census_msa', axis = 1, inplace=True)
test_data = pd.concat([census_msa, test_data], axis =1 )


# In[262]:


respondent_id = test_data['respondent_id']
test_data.drop('respondent_id', axis = 1, inplace=True)
test_data.insert(0, 'respondent_id', respondent_id)
#pre-processing


# In[263]:


from sklearn.model_selection import train_test_split


# In[265]:


features = test_data.iloc[:,1:45]


# In[266]:


labels = pd.read_csv(r'C:\Users\Akash Memon\Documents/training_set_labels.csv')
h1n1_labels = labels.iloc[:,1:2]
for column in features.columns:
    features[column].fillna(test_data[column].median(), inplace=True)
#handle missing values


# In[267]:


x_train, x_test, y_train, y_test = train_test_split(features, h1n1_labels, test_size = 0.33);


# In[268]:


from sklearn.ensemble import RandomForestClassifier


# In[269]:


h1n1_random_forest = RandomForestClassifier(n_estimators=250, random_state=42)
h1n1_random_forest.fit(x_train, y_train.values.ravel())


# In[276]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[278]:


h1n1_probs = h1n1_random_forest.predict_proba(x_test)


# In[279]:


print(h1n1_probs)


# In[280]:


h1n1_probs = h1n1_probs[:, 1]
h1n1_auc = roc_auc_score(y_test, h1n1_probs)


# In[281]:


h1n1_auc


# In[282]:


seasonal_labels = labels.iloc[:,2:3]
seasonal_labels.head()


# In[283]:


x_train, x_test, y_train, y_test = train_test_split(features, seasonal_labels, test_size = 0.33);


# In[284]:


seasonal_random_forest = RandomForestClassifier(n_estimators=250, random_state=42)
seasonal_random_forest.fit(x_train, y_train.values.ravel())


# In[285]:


seasonal_probs = seasonal_random_forest.predict_proba(x_test)


# In[286]:


print(seasonal_probs)


# In[287]:


seasonal_probs = seasonal_probs[:, 1]
seasonal_auc = roc_auc_score(y_test, seasonal_probs)


# In[288]:


print(seasonal_auc)


# In[338]:


test_features = pd.read_csv(r'C:\Users\Akash Memon\Documents/test_set_features.csv')


# In[339]:


test_features.drop('employment_occupation' , axis=1, inplace=True)


# In[340]:


test_features.drop('employment_industry' , axis=1, inplace=True)


# In[341]:


age_group = pd.get_dummies(test_features['age_group'], drop_first=False)
age_group.drop('65+ Years', axis = 1, inplace=True)


# In[342]:


test_data = test_features.drop('age_group', axis =1 , inplace=True)
test_data = pd.concat([age_group, test_features], axis =1 )


# In[343]:


age = pd.get_dummies(test_features['education'], drop_first=False)


# In[344]:


age.drop('Some College', axis = 1, inplace=True)


# In[345]:


test_data = pd.concat([age, test_data], axis = 1)
test_data.drop('education', axis = 1, inplace = True)


# In[346]:


race = pd.get_dummies(test_features['race'], drop_first=False)


# In[347]:


race.drop('Other or Multiple', axis = 1, inplace=True)
test_data.drop('race', axis = 1, inplace=True)
test_data = pd.concat([race, test_data], axis =1 )


# In[348]:


sex = pd.get_dummies(test_features['sex'], drop_first=False)


# In[349]:


test_data.drop('sex', axis = 1, inplace=True)
test_data = pd.concat([sex, test_data], axis =1 )


# In[350]:


income_poverty = pd.get_dummies(test_features['income_poverty'], drop_first=True)


# In[351]:


test_data.drop('income_poverty', axis = 1, inplace=True)
test_data = pd.concat([income_poverty, test_data], axis =1 )


# In[352]:


test_data['marital_status'].unique()


# In[353]:


marital_status = pd.get_dummies(test_features['marital_status'], drop_first=False)


# In[354]:


marital_status.drop('Not Married', axis =1 , inplace=True)
test_data.drop('marital_status', axis = 1, inplace=True)
test_data = pd.concat([marital_status, test_data], axis =1 )


# In[355]:


test_data['rent_or_own'].unique()


# In[356]:


rent_or_own = pd.get_dummies(test_features['rent_or_own'], drop_first=False)


# In[357]:


rent_or_own.drop("Rent", axis =1, inplace=True)


# In[358]:


test_data.drop('rent_or_own', axis = 1, inplace=True)
test_data = pd.concat([rent_or_own, test_data], axis =1 )


# In[359]:


test_data['employment_status'].unique()


# In[360]:


employment_status = pd.get_dummies(test_features['employment_status'], drop_first=False)


# In[361]:


employment_status.drop('Unemployed', axis =1 , inplace=True)
test_data.drop('employment_status', axis = 1, inplace=True)
test_data = pd.concat([employment_status, test_data], axis =1 )


# In[362]:


test_data.rename(columns = {'Own':'owns_house'}, inplace=True)


# In[363]:


test_data['hhs_geo_region'].unique()


# In[364]:


test_data.drop('hhs_geo_region', axis=1, inplace=True)


# In[365]:


test_data['census_msa'].unique()


# In[366]:


census_msa = pd.get_dummies(test_features['census_msa'], drop_first=True)


# In[367]:


test_data.drop('census_msa', axis = 1, inplace=True)
test_data = pd.concat([census_msa, test_data], axis =1 )


# In[368]:


respondent_id = test_data['respondent_id']
test_data.drop('respondent_id', axis = 1, inplace=True)


# In[369]:


for column in test_data.columns:
    test_data[column].fillna(test_data[column].median(), inplace=True)


# In[370]:


h1n1_final_probs = h1n1_random_forest.predict_proba(test_data)


# In[371]:


print(h1n1_final_probs)


# In[372]:


seasonal_final_probs = seasonal_random_forest.predict_proba(test_data)


# In[373]:


print(seasonal_final_probs)


# In[375]:


submission_format = pd.read_csv(r'C:\Users\Akash Memon\Documents/submission_format.csv')


# In[377]:


submission_format["h1n1_vaccine"] = h1n1_final_probs[:, 1]
submission_format["seasonal_vaccine"] = seasonal_final_probs[:, 1]
submission_format["h1n1_outcome"] = [[] for _ in range(len(submission_format))]
submission_format["seasonal_outcome"] = [[] for _ in range(len(submission_format))]


# In[378]:


for i in range(len(submission_format)):
    if(submission_format.at[i, 'h1n1_vaccine'] < 0.20):
        submission_format.at[i, 'h1n1_outcome'] = 'Highly Unlikely'
    elif(0.20 < submission_format.at[i, 'h1n1_vaccine'] < 0.40):
        submission_format.at[i, 'h1n1_outcome'] = 'Unlikely'
    elif(0.4 < submission_format.at[i, 'h1n1_vaccine'] < 0.6):
        submission_format.at[i, 'h1n1_outcome'] = 'Uncertain'
    elif(0.6 < submission_format.at[i, 'h1n1_vaccine'] < 0.8):
        submission_format.at[i, 'h1n1_outcome'] = 'Likely'
    else: submission_format.at[i, 'h1n1_outcome'] = 'Highly Likely'
    
        
    if(submission_format.at[i, 'seasonal_vaccine'] < 0.20):
        submission_format.at[i, 'seasonal_outcome'] = 'Highly Unlikely'
    elif(0.2 < submission_format.at[i, 'seasonal_vaccine'] < 0.4):
        submission_format.at[i, 'seasonal_outcome'] = 'Unlikely'
    elif(0.4 < submission_format.at[i, 'seasonal_vaccine'] < 0.6):
        submission_format.at[i, 'seasonal_outcome'] = 'Uncertain'
    elif(0.6 < submission_format.at[i, 'seasonal_vaccine'] < 0.8):
        submission_format.at[i, 'seasonal_outcome'] = 'Likely'
    else: submission_format.at[i, 'seasonal_outcome'] = 'Highly Likely'


# In[379]:


#submission_format.to_csv('my_submission.csv', index = False)


# In[380]:


test12 = pd.read_csv(r'C:\Users\Akash Memon\Documents/test_set_features.csv')


# In[381]:


test12 = pd.concat([submission_format, test12], axis=1)


# In[382]:


test12.to_csv('testAI.csv', index=False)

