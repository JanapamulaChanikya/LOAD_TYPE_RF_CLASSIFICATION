#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r"C:\Users\janap\Downloads\load_data.csv")


# In[4]:


df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d-%m-%Y %H:%M')


# In[5]:


print(df.info())
print(df.head())


# In[6]:


for col in df.columns:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)


# In[7]:


le = LabelEncoder()
df['Load_Type_enc'] = le.fit_transform(df['Load_Type'])


# In[8]:


df['Hour'] = df['Date_Time'].dt.hour
df['Month'] = df['Date_Time'].dt.month


# In[9]:


features = ['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh',
            'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM', 'Hour', 'Month']
X = df[features]
y = df['Load_Type_enc']



# In[10]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[11]:


last_month = df['Date_Time'].dt.month.max()
train_mask = df['Date_Time'].dt.month < last_month
test_mask = df['Date_Time'].dt.month >= last_month


# In[12]:


X_train = X_scaled[train_mask.values]  
X_test = X_scaled[test_mask.values]
y_train = y[train_mask]
y_test = y[test_mask]


# In[13]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[14]:


y_pred = model.predict(X_test)


# In[15]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))


# In[16]:


print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))


# In[17]:


importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)


# In[18]:


plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Feature Importances from Random Forest")
plt.show()


# In[19]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

