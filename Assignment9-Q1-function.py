#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

import warnings #These two lines will ignore the warning messages
warnings.simplefilter("ignore")  

data0=pd.read_excel('Lecture9-IBM-Data-Clean.xlsx')

data0.head().transpose() # In order to show all the columns, you can use transpose()


# In[ ]:


data0 = pd.concat([data0, pd.get_dummies(data0["BusinessTravel"], prefix = "BusinessTravel"), 
                      pd.get_dummies(data0["Department"], prefix = "Department"), 
                      pd.get_dummies(data0["EducationField"], prefix = "EducationField"), 
                      pd.get_dummies(data0["Gender"], prefix = "Gender"), 
                      pd.get_dummies(data0["JobRole"], prefix = "JobRole"), 
                      pd.get_dummies(data0["MaritalStatus"], prefix = "MaritalStatus"),
                      pd.get_dummies(data0["OverTime"], prefix = "OverTime")], axis = 1)

data0 = data0.drop(["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"], axis = 1)

data0["Attrition"] = data0["Attrition"].apply(lambda x: 1 if x == "Yes" else 0) #apply() apply a if function to transform Attrition from Yes/No to 1/0. 

data0.head().transpose()


# In[ ]:


from sklearn.model_selection import train_test_split

X = data0.drop(['Attrition'], axis = 1) #not all axis=1 indicate columns, check the function definition. 
y = data0['Attrition']          

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train.shape, X_test.shape)


# ### - Logistic Regression

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
logreg = GridSearchCV(model,param_grid={"penalty": ['l1','l2']})

logreg.fit(X_train, y_train)

y_pred_log = logreg.predict(X_test)

print("Logistic regression accuracy for test set:", logreg.score(X_test, y_test))
print("\nClassification report:")
print(classification_report(y_test, y_pred_log))
print(logreg.best_estimator_.coef_)
print(logreg.best_estimator_.intercept_)
print(X_train.columns)

