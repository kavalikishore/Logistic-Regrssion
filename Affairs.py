import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as smf
Affairs=pd.read_csv("E:\\ExcelR Assignments\\Assignments\\logistic Regression\\affairs.csv")
#I have a dataset containing family information of married couples, which have around 10 variables & 600+ observations. 
#Independent variables are ~ gender, age, years married, children, religion etc.
#I have one response variable which is number of extra marital affairs.
 #Now, I want to know what all factor influence the chances of extra marital affair.
#Since extra marital affair is a binary variable (either a person will have or not), 
#so we can fit logistic regression model here to predict the probability of extra marital affair.
#install.packages('AER')
#data(Affairs,package="AER")
Affairs.describe()
#to check null values..
Affairs.isnull().sum()#no null values in this dataset
#to check for linearity using pairplot
import seaborn as sns
sns.pairplot(Affairs) #pair plot to see the correlation
plt.show()
##Unnamed not required
Affairs.drop(["Unnamed: 0"],inplace=True,axis=1)

Affairs.columns
Affairs.affairs.value_counts()
#for categorizing in 0 and 1 for logistic_regression
Affairs["Att_val"] = np.zeros(601)
# converting the affairs to binary variable
Affairs.loc[Affairs.affairs >= 1,"Att_val"] = 1
Affairs.drop(["affairs"],axis=1,inplace=True)
Affairs.drop(["Att_val"],axis=1,inplace=True)
#create Dummy Variables
Affairs.gender=pd.get_dummies(Affairs.gender)
Affairs.children=pd.get_dummies(Affairs.children)
Affairs.columns
##Logistic Regression Model
import statsmodels.formula.api as sm
logit_model = sm.logit('Att_val~age+yearsmarried+religiousness+rating',data = Affairs).fit()
logit_model.summary()##gender,Children,education, and occupation not haiving significant..so remove those variables and build the model
#Out[97]: 
#<class 'statsmodels.iolib.summary.Summary'>
"""
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                Att_val   No. Observations:                  601
Model:                          Logit   Df Residuals:                      596
Method:                           MLE   Df Model:                            4
Date:                Fri, 06 Sep 2019   Pseudo R-squ.:                 0.08887
Time:                        06:18:02   Log-Likelihood:                -307.68
converged:                       True   LL-Null:                       -337.69
                                        LLR p-value:                 2.874e-12
=================================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
---------------------------------------------------------------------------------
Intercept         1.9308      0.610      3.164      0.002       0.735       3.127
age              -0.0353      0.017     -2.032      0.042      -0.069      -0.001
yearsmarried      0.1006      0.029      3.445      0.001       0.043       0.158
religiousness    -0.3290      0.089     -3.678      0.000      -0.504      -0.154
rating           -0.4614      0.089     -5.193      0.000      -0.635      -0.287
=================================================================================
"""
##calculating prdictions
y_pred=logit_model.predict(Affairs)

Affairs["pred_prob"]=y_pred##creating new columns for storing predicted class of affairs
Affairs["output"]=np.zeros(601)##filling all the cells with zeros
#taking threshold value as 0.5 and above the prob value will be treated 
#as correct value
Affairs.loc[y_pred>0.5,"output"]=1
Affairs.drop(["ouput"],inplace=True,axis=1)

##confuccion matrix
Confussion_matrix=pd.crosstab(Affairs.Att_val,Affairs.output)
Confussion_matrix
Confussion_matrix
#Out[114]: 
#output   0.0  1.0
#Att_val          
0.0      432 #TN  19 #FP
1.0      128#FN   22#TP

from sklearn import metrics
print(metrics.accuracy_score(Affairs.Att_val,Affairs.output))#calu accuracy value,0.7554076539101497
print(metrics.precision_score(Affairs.Att_val,Affairs.output))#cal precision value,0.5365853658536586
print(metrics.recall_score(Affairs.Att_val,Affairs.output))#cal recall Value,0.14666666666666667

accuracy = (432+22)/(432+22+128+19)
accuracy#0.755
precision = 22/(128+22)
precision#0.1466
recall = 22/(22+19)
recall#0.5365
f1_score = 2*(precision*recall/(precision+recall))  
f1_score#0.230.
classification_error=1-accuracy
classification_error#0.24459
print(accuracy,precision,recall,f1_score)
#0.7554076539101497 0.14666666666666667 0.5365853658536586 0.23036649214659685

# ROC curve 
from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, thresholds = metrics.roc_curve(Affairs.Att_val,Affairs.output)


# the above function is applicable for binary classification class 
fig,ax=plt.subplots()
plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")

cutoff_dataframe=pd.DataFrame({"fpr":fpr,"tpr":tpr,"thresholds":threshold})
cutoff_dataframe
...Out[142]: 
        fpr       tpr  cutoff
0  0.000000  0.000000     2.0
1  0.042129  0.146667     1.0
2  1.000000  1.000000     0.0

def evaluate_threshold(threshold):
    print("sensitivity:",tpr[thresholds>threshold][-1])
    print("specificity:",fpr[thresholds>threshold][-1])

evaluate_threshold(0.50)
 
roc_auc = metrics.auc(fpr, tpr)
roc_auc##005522
