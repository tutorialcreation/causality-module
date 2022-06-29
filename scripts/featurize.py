#!/usr/bin/env python
# coding: utf-8

# # Feature Selection

# In the following cells, we will select a group of variables, the most predictive ones, to build our machine learning models. Feature Selection is the process where you automatically or manually select those features which contribute most to your prediction variable or output in which you are interested in. Having irrelevant features in your data can decrease the accuracy of the models and make your model learn based on irrelevant features. The selection of relevant features may also get benefitted from the right domain knowledge.
# 
# ### Why do we need to select variables?
# 
# 1. For production: Fewer variables mean smaller client input requirements (e.g. customers filling out a form on a website or mobile app), and hence less code for error handling. This reduces the chances of bugs.
# 2. For model performance: Fewer variables mean simpler, more interpretable, less over-fitted models

# In this part we will select feature with different methods that are 
# 
# #### 1. Feature selection with correlation 
# #### 2. Univariate feature selection (Chi-square)
# #### 3. Recursive feature elimination (RFE) with random forest
# #### 4. Recursive feature elimination with cross validation(RFECV) with random forest
# #### 5. Tree based feature selection with random forest classification
# #### 6. L1-based feature selection (LinearSVC)
# #### 7. Tree-based feature selection (ExtraTrees)
# #### 8. Vote based feature selection
# 
# We will use random forest classification in order to train our model and predict.

# #### Import libraries

# In[113]:


# importing the libraries
from functools import reduce
# linear algebra
import numpy as np 
# data processing, CSV file I/O
import pandas as pd 
# data visualization library
import seaborn as sns  
import matplotlib.pyplot as plt
from pandas import DataFrame
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
dataset = pd.read_csv("../data/data.csv")


# #### Drop unnecessary columns in a dataset

# In[3]:
class Featurize:
    def __init__(self) -> None:
        pass

    def get_xy(self,data:pd.DataFrame,list_drp:list):
            """
            set the x and y column
            
            args:
                data(pd.DataFrame): the dataFrame which we are extracting the x and y
            
            returns:
                y and X in form of pandas series
            
            """
            y = data.diagnosis # M or B 
            X = data.drop(list_drp,axis = 1 )
            return y,X




    # #### 1) Feature selection with correlation

    # As it can be seen in map heat figure radius_mean, perimeter_mean and area_mean are correlated with each other so we will use only area_mean. If you ask how i choose area_mean as a feature to use, well actually there is no correct answer, I just look at swarm plots and area_mean looks like clear for me but we cannot make exact separation among other correlated features without trying. So lets find other correlated features and look accuracy with random forest classifier.
    # 
    # Compactness_mean, concavity_mean and concave points_mean are correlated with each other.Therefore I only choose concavity_mean. Apart from these, radius_se, perimeter_se and area_se are correlated and I only use area_se. radius_worst, perimeter_worst and area_worst are correlated so I use area_worst. Compactness_worst, concavity_worst and concave points_worst so I use concavity_worst. Compactness_se, concavity_se and concave points_se so I use concavity_se. texture_mean and texture_worst are correlated and I use texture_mean. area_worst and area_mean are correlated, I use area_mean.

    # #### Drop high correlated columns in a dataset

    # In[20]:


    def by_correlation(self,x,drop_list_cor):
        """
        selects the features by correlation
        
        args:
            x (pd.DataFrame):a dataframe of the independent variables
            drop_list_cor (list): a list of the columns believed to have high correlation
            
        returns:
            a dataframe demonstrating correlation among
        """
        x_1 = x.drop(drop_list_cor,axis = 1 )        # do not modify x, we will use it later 
        x_1.head()
        selected_feature_corr=x_1.columns
        fs_corr = np.ones(len(x_1.columns)).astype(int)
        fs_corr = DataFrame(fs_corr, columns = ["Corr"], index=x_1.columns)
        f,ax = plt.subplots(figsize=(14, 14))
        sns.heatmap(x_1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
        return x_1,fs_corr


    # In[23]:


    # After drop correlated features, as it can be seen in below correlation matrix, there are no more correlated features. Actually, I know and you see there is correlation value 0.9 but lets see together what happen if we do not drop it.

    # Well, we choose our features but did we choose correctly ? Lets use random forest and find accuracy according to chosen features.

    # In[75]:


    def test_rf(self,x,y):
        """
        find correlation tests by random forest
        
        args:
            x (pd.DataFrame): a dataframe of the independent variables
            y (pd.DataFrame): a dataframe of the dependent variable
        
        returns:
            a heatmap showing rf results
            
        """
        # split data train 70 % and test 30 %
        x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3, random_state=42)

        #random forest classifier with n_estimators=10 (default)
        clf_rf = RandomForestClassifier(random_state=43)      
        clr_rf = clf_rf.fit(x_train,y_train)

        ac = accuracy_score(y_test,clf_rf.predict(x_test))
        print('Accuracy is: ',ac)
        cm = confusion_matrix(y_test,clf_rf.predict(x_test))
        sns.heatmap(cm,annot=True,fmt="d")
        return x_train, x_test, y_train, y_test,clr_rf


    # In[76]:




    # Accuracy is almost 95% and as it can be seen in confusion matrix, we make few wrong prediction. Now lets see other feature selection methods to find better results.

    # #### 2. Univariate feature selection (Chi-square)

    # In univariate feature selection, we will use SelectKBest that removes all but the k highest scoring features. 

    # In this method we need to choose how many features we will use. For example, will k (number of features) be 5 or 10 or 15? The answer is only trying or intuitively. I do not try all combinations but I only choose k = 10 and find best 10 features.

    # In[45]:


    def by_chi2(self,x_train,y_train):
        """
        selects features based on the chisquared method
        
        args:
            x_train (pd.DataFrame): dataframe of the training x dataset
            y_train (pd.DataFrame): dataframe of the training y dataset
        
        returns:
            result of the chi2 in a dataframe
        """
        # find best scored 10 features
        select_feature = SelectKBest(chi2, k=10).fit(x_train, y_train)
        np.set_printoptions(suppress=True)
        print('Score list:', select_feature.scores_)
        pd.options.display.float_format = '{:.2f}'.format
        fs_chi2 = pd.DataFrame(select_feature.scores_, columns = ["Chi_Square"], index=x_train.columns)
        fs_chi2 = fs_chi2.reset_index()
        fs_chi2 = fs_chi2.sort_values('Chi_Square',ascending=0)
        return select_feature,fs_chi2


    # In[53]:




    # #### 3. Recursive feature elimination (RFE) with random forest

    # Basically, it uses one of the classification methods (random forest in our example), assign weights to each of features. Whose absolute weights are the smallest are pruned from the current set features. That procedure is recursively repeated on the pruned set until the desired number of features
    # 
    # Like previous method, we will use 10 features. However, which 10 features will we use ? We will choose them with RFE method.

    # In[58]:


    def by_rfe(self,x_train,y_train):
        """
        select by recursive feature elimination
        
        args:
            x_train (pd.DataFrame): a pandas dataframe for trarining x dataset
            y_train (pd.DataFrame): a pandas dataframe for training y dataset
            
        returns:
            a datafram of the extracted features
        """
        
        # Create the RFE object and rank each pixel
        clf_rf_3 = RandomForestClassifier()      
        rfe = RFE(estimator=clf_rf_3, n_features_to_select=10, step=1)
        rfe = rfe.fit(x_train, y_train)
        # let's print the number of total and selected features
        fs_rfe = DataFrame(rfe.support_, columns = ["RFE"], index=x_train.columns)
        fs_rfe = fs_rfe.reset_index()
        # this is how we can make a list of the selected features
        # let's print some stats
        print('total features: {}'.format((x_train.shape[1])))
        print('selected features: {}'.format(len(x_train.columns[rfe.support_])))
        print('Chosen best 10 feature by rfe:',x_train.columns[rfe.support_])
        return fs_rfe,rfe


    # In[59]:




    # Chosen 10 best features by rfe is texture_mean, area_mean, smoothness_mean, concavity_mean, area_se, concavity_se, fractal_dimension_se, concavity_worst,symmetry_worst,fractal_dimension_worst. They are similar with previous (selectkBest) method. Therefore we do not need to calculate accuracy again. Shortly, we can say that we make good feature selection with rfe and selectkBest methods. However as you can see there is a problem, okey I except we find best 10 feature with two different method and these features are almost same but why it is 10. Maybe if we use best 5 or best 12 feature we will have better accuracy. Therefore lets see how many feature we need to use with rfecv method.

    # #### 4.  Recursive feature elimination with cross validation(RFECV) with random forest

    # Now we will not only find best features but we also find how many features do we need for best accuracy.

    # In[62]:


    def by_rfecv(self,x_train,y_train):
        """
        select by recursive feature elimination
        
        args:
            x_train (pd.DataFrame): a pandas dataframe for trarining x dataset
            y_train (pd.DataFrame): a pandas dataframe for training y dataset
            
        returns:
            a datafram of the extracted features
        """
        # The "accuracy" scoring is proportional to the number of correct classifications
        clf_rf_4 = RandomForestClassifier() 
        rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
        rfecv = rfecv.fit(x_train, y_train)
        # let's print the number of total and selected features
        fs_rfecv = DataFrame(rfecv.support_, columns = ["RFECV"], index=x_train.columns)
        fs_rfecv = fs_rfecv.reset_index()
        # this is how we can make a list of the selected features
        # let's print some stats
        print('total features: {}'.format((x_train.shape[1])))
        print('selected features: {}'.format(len(x_train.columns[rfecv.support_])))
        print('Optimal number of features :', rfecv.n_features_)
        print('Best features by rfecv:',x_train.columns[rfecv.support_])
        return rfecv,fs_rfecv


    # In[65]:


    rfecv,fs_rfecv = by_rfecv(x_train,y_train)
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score of number of selected features")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


    # Finally, we find best 15 features that are 'texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean',
    #        'symmetry_mean', 'fractal_dimension_mean', 'area_se', 'smoothness_se',
    #        'concavity_se', 'symmetry_se', 'fractal_dimension_se',
    #        'smoothness_worst', 'concavity_worst', 'symmetry_worst',
    #        'fractal_dimension_worst' for best classification. Lets look at best accuracy with plot.

    # Lets look at what we did up to this point. Lets accept that this data is very easy to classification. However, our first purpose is actually not finding good accuracy. Our purpose is learning how to make feature selection and understanding data. 

    # #### 5. Tree based feature selection and random forest classification 

    # In random forest classification method there is a featureimportances attributes that is the feature importances (the higher, the more important the feature). !!! To use feature_importance method, in training data there should not be correlated features. Random forest choose randomly at each iteration, therefore sequence of feature importance list can change.

    # In[83]:


    def by_rf(self,x_train,y_train):
        """
        select by recursive feature elimination
        
        args:
            x_train (pd.DataFrame): a pandas dataframe for trarining x dataset
            y_train (pd.DataFrame): a pandas dataframe for training y dataset
            
        returns:
            a datafram of the extracted features
        """
        clf_rf_5 = RandomForestClassifier()      
        clr_rf_5 = clf_rf_5.fit(x_train,y_train)
        importances = clr_rf_5.feature_importances_
        clf_rf = RandomForestClassifier(random_state=43)   
        clr_rf = clf_rf.fit(x_train,y_train)

        std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
                    axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(x_train.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest

        plt.figure(1, figsize=(14, 13))
        plt.title("Feature importances")
        plt.bar(range(x_train.shape[1]), importances[indices],
            color="g", yerr=std[indices], align="center")
        plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
        plt.xlim([-1, x_train.shape[1]])
        plt.show()
        print(importances[indices])
        # let's print the number of total and selected features
        # let's print some stats
        print('total features: {}'.format((x_train.shape[1])))
        #print('Chosen optimal features by rf:',selected_feature_rf[1:10])
        fs_rf = DataFrame(clr_rf_5.feature_importances_, columns = ["RF"], index=x_train.columns)
        fs_rf = fs_rf.reset_index()
        fs_rf = fs_rf.sort_values('RF',ascending=0)
        return fs_rf


    # As you can seen in plot above, after 6 best features importance of features decrease. Therefore we can focus these 6 features. 

    # In[84]:



    # #### 6. L1-based feature selection (LinearSVC)

    # In[92]:


    def by_l1(self,x_train,y_train):
        """
        select by recursive feature elimination
        
        args:
            x_train (pd.DataFrame): a pandas dataframe for trarining x dataset
            y_train (pd.DataFrame): a pandas dataframe for training y dataset
            
        returns:
            a datafram of the extracted features
        """
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False,max_iter=2000).fit(x_train, y_train) 
        model = SelectFromModel(lsvc, prefit=True) 
        x_new = model.transform(x_train) 
        print(x_train.columns[model.get_support()]) 
        # let's print the number of total and selected features
        fs_l1 = DataFrame(model.get_support(), columns = ["L1"], index=x_train.columns)
        fs_l1 = fs_l1.reset_index()
        # this is how we can make a list of the selected fes
        selected_feature_lsvc = x_train.columns[model.get_support()]

        # let's print some stats
        print('total features: {}'.format((x_train.shape[1])))
        print('selected features: {}'.format(len(selected_feature_lsvc)))
        print('Best features by lsvc:',x_train.columns[model.get_support()])
        return fs_l1,selected_feature_lsvc,model


    # In[93]:


    def by_trees(self,x_train,y_train,model):
        """
        select by recursive feature elimination
        
        args:
            x_train (pd.DataFrame): a pandas dataframe for trarining x dataset
            y_train (pd.DataFrame): a pandas dataframe for training y dataset
            
        returns:
            a datafram of the extracted features
        """
        # Build a forest and compute the impurity-based feature importances
        clf = ExtraTreesClassifier(n_estimators=32,random_state=0)
        clf.fit(x_train, y_train)
        clf.feature_importances_ 
        importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                    axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(x_train.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the impurity-based feature importances of the forest
        plt.figure(1, figsize=(14, 13))
        plt.title("Feature importances")
        plt.bar(range(x_train.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
        plt.xlim([-1, x_train.shape[1]])
        plt.show()        
        # let's print the number of total and selected features
        # this is how we can make a list of the selected features
        selected_feature_extraTrees = x_train.columns[model.get_support()]
        # let's print some stats
        print('total features: {}'.format((x_train.shape[1])))
        print('selected features: {}'.format(len(selected_feature_extraTrees)))
        print('Best features by ExtraTrees:',x_train.columns[model.get_support()])
        fs_extratrees=DataFrame(clf.feature_importances_, columns = ["Extratrees"], index=x_train.columns)
        fs_extratrees = fs_extratrees.reset_index()
        fs_extratrees = fs_extratrees.sort_values(['Extratrees'],ascending=0)
        return fs_extratrees


    # In[96]:




    # #### 8. Vote based feature selection
    # #### Combine all together

    # In[100]:


    def combine_vote(self,dfs):
        """
        combine all features selected to make a vote
        
        args:
            dfs (list): a list of all features
        
        returns:
            score of features that occur in all
        """
        final_results = reduce(lambda left,right: pd.merge(left,right,on='index'), dfs)
        columns = ['Chi_Square', 'RF', 'Extratrees']
        score_table = pd.DataFrame({},[])
        score_table['index'] = final_results['index']
        for i in columns:
            score_table[i] = final_results['index'].isin(list(final_results.nlargest(10,i)['index'])).astype(int)

        #score_table['Corr'] = final_results['Corr'].astype(int)
        score_table['RFE'] = final_results['RFE'].astype(int)
        score_table['RFECV'] = final_results['RFECV'].astype(int)
        score_table['L1'] = final_results['L1'].astype(int)
        score_table['final_score'] = score_table.sum(axis=1)   
        scores = score_table.sort_values('final_score',ascending=0)
        return scores


    # In[103]:




    # In[48]:


    # ExtraTrees features


    # #### Multicollinearity - VIF (Addon)

    # In[105]:


    def calculate_vif(self,features):
        """
        return variation inflation
        
        args:
            features (pd:DataFrame): a dataframe of the training dataset
        
        returns:
            variance inflation value
        """
        vif = pd.DataFrame()
        vif["Features"] = features.columns
        vif["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
        return(vif)


    # In[111]:


    def vif(self,x_train):
        """
        return variation inflation
        
        args:
            features (pd:DataFrame): a dataframe of the training dataset
        
        returns:
            variance inflation dataframe
        """
        vif = calculate_vif(x_train)
        while vif['VIF'][vif['VIF'] > 10].any():
            remove = vif.sort_values('VIF',ascending=0)['Features'][:1]
            x_train.drop(remove,axis=1,inplace=True)
            vif = calculate_vif(x_train)
        return vif


    # In[112]:




    # #### Feature Extraction using PCA

    # We will use principle component analysis (PCA) for feature extraction. Before PCA, we need to normalize data for better performance of PCA.

    # In[116]:


    def by_pca(self,x,y):
        """
        return principal compenonent analysis of features
        
        args:
            x (pd:DataFrame): a dataframe of the x dataset
            y (pd:DataFrame): a dataframe of the y dataset
    
        returns:
            plot of the principal component analysis
        """
        # split data train 70 % and test 30 %
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        #normalization
        x_train_N = (x_train-x_train.mean())/(x_train.max()-x_train.min())
        x_test_N = (x_test-x_test.mean())/(x_test.max()-x_test.min())

        pca = PCA()
        pca.fit(x_train_N)

        plt.figure(1, figsize=(14, 13))
        plt.clf()
        plt.axes([.2, .2, .7, .7])
        plt.plot(pca.explained_variance_ratio_, linewidth=2)
        plt.axis('tight')
        plt.xlabel('n_components')
        plt.ylabel('explained_variance_ratio_')


    # In[117]:




    # According to variance ration, 3 component can be chosen.