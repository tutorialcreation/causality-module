#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from causalnex.structure.notears import from_pandas
from causalnex.structure import StructureModel
from causalnex.network import BayesianNetwork
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.discretiser import Discretiser
from causalnex.discretiser.discretiser_strategy import ( DecisionTreeSupervisedDiscretiserMethod )
from causalnex.evaluation import classification_report,roc_auc

import networkx as nx
import joblib

from IPython.display import Image
import warnings
warnings.filterwarnings('ignore')

BETA = 10
RATIO = 0.30
FEATURES='fs_voted'


class CausalNet:
    def __init__(self) -> None:
        pass

        
    def get_features(self):
        """
        this method gets a set of features from our previous feature analysis
        """
        features = {
            # 1. Features selected from correlations
            "fs_corr" : ['texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean','symmetry_mean',
                            'fractal_dimension_mean', 'texture_se', 'area_se','smoothness_se', 'concavity_se',
                            'symmetry_se', 'fractal_dimension_se','smoothness_worst', 'concavity_worst', 
                            'symmetry_worst', 'fractal_dimension_worst'],

            # 2. Univariate feature selection SelectKBest, chi2
            "fs_chi2" : ['texture_mean', 'area_mean', 'concavity_mean', 'symmetry_mean', 'area_se', 
                            'concavity_se', 'smoothness_worst', 'concavity_worst', 'symmetry_worst', 
                            'fractal_dimension_worst'],

            # 3. Recursive feature elimination (RFE) with random forest
            "fs_rfe" : ['texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'area_se', 
                    'smoothness_se', 'concavity_se', 'smoothness_worst', 'concavity_worst', 'symmetry_worst'],

            # 4. Recursive feature elimination with cross validation(RFECV) with random forest
            "fs_rfecv" : ['texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean','fractal_dimension_mean'
                        , 'area_se', 'concavity_se', 'concavity_worst', 'symmetry_worst'],

            # 5. Tree based feature selection with random forest classification
            "fs_rf" : ['texture_mean', 'area_mean', 'concavity_mean', 'area_se', 'concavity_se', 
                    'fractal_dimension_se', 'smoothness_worst','concavity_worst', 'symmetry_worst', 
                    'fractal_dimension_worst'],

            # 6. ExtraTree based feature selection 
            "fs_extraTree" : ['texture_mean', 'area_mean', 'concavity_mean', 'fractal_dimension_mean', 'area_se', 
                            'concavity_se','smoothness_worst', 'concavity_worst', 
                            'symmetry_worst','fractal_dimension_worst'],

            # 7. L1 feature selection (LinearSVC)
            "fs_l1" : ['texture_mean', 'area_mean', 'area_se'],

            # 8. Vote based feature selection
            "fs_voted" : ['texture_mean',  'area_mean',  'smoothness_mean',  'concavity_mean',  
                            'fractal_dimension_mean',  'area_se',  'concavity_se',  'smoothness_worst',  
                            'concavity_worst',  'symmetry_worst',  'fractal_dimension_worst'],

            
        }
        return features


    # In[3]:




    # In[4]:


    def load_data(self,filename,ratio):
        """
        returns complete dataset as split dataset
        
        args:
            model_full (pd.DataFrame): a pandas dataframe containing complete dataset
        
        returns:
            split dataset in form of train and test datasets
        """
        global feature_names, response_name, n_features, model_full  
        model_full = pd.read_csv(filename)
        
        # we change the class values (at the column number 2) from B to 0 and from M to 1
        model_full.iloc[:,1].replace('B', 0,inplace=True)
        model_full.iloc[:,1].replace('M', 1,inplace=True)
        response_name = ['diagnosis']
        drop_list = ['Unnamed: 32','id','diagnosis']
        model_full_x= model_full.drop(drop_list,axis = 1)
        X = model_full_x
        y = model_full.diagnosis
        
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size = ratio,
                                                            random_state = 12345)
        return X_train, y_train, X_test, y_test


    # In[5]:




    # In[42]:


    def scale_data(self,x_train,y_train,features):
        """
        scales the data obtained for causal networking
        """
        y = pd.DataFrame(y_train,columns=['diagnosis'])
        x_train = x_train.join(y)
        X_train = x_train[features[FEATURES]]
        # scaling data
        cols = X_train.columns
        print('------columns------')
        print(cols)
        print('--------------------')
        sc = MinMaxScaler()
        X_train = pd.DataFrame(sc.fit_transform(X_train), columns=cols)
        X_test = pd.DataFrame(sc.transform(X_test), columns=cols)
        X_train['diagnosis'] = x_train['diagnosis']
        print('Size of data:')
        print ('The train data has {0} rows and {1} columns'.format(X_train.shape[0],X_train.shape[1]))
        print ('----------------------------')
        print ('The test data has {0} rows and {1} columns'.format(X_test.shape[0],X_test.shape[1]))
        df_feat = X_train
        df_feat = df_feat.dropna()
        return df_feat


    # In[7]:




    # In[11]:




    def draw_network(self,SM):
        """
        drawing of causal networks using nx
        """
        plt.figure(figsize=(18,10))
        pos = nx.spring_layout(SM, k=60)

        edge_width = [ d['weight']*0.3 for (u,v,d) in SM.edges(data=True)]
        #nx.draw_networkx_labels(SM, pos, fontsize=16, font_family="Yu Gothic", font_weight="bold")
        nx.draw_networkx_labels(SM, pos, font_family="Yu Gothic", font_weight="bold")
        nx.draw_networkx(SM,
                        pos,
                        node_size=4000,
                        arrowsize=20,
                        alpha=0.6,
                        edge_color='b',
                        width=edge_width)
        return SM

    # In[145]:


    def vis_sm(self,sm):
        """
        for visualizing structural models
        """
        viz = plot_structure(
                    sm,
                    graph_attributes={"scale": "1.0"},
                    all_node_attributes=NODE_STYLE.WEAK,
                    all_edge_attributes=EDGE_STYLE.WEAK
            )
        return Image(viz.draw(format='png'))


    # In[146]:


    

    # # Data test

    # In[15]:


    def jaccard_similarity(self,g, h):
        """
        for calculating the jacard similarity index
        and therefore grants us the opportunities of comparing two
        different graphs to help us make better conclusions
        """
        i = set(g).intersection(h)
        return round(len(i) / (len(g) + len(h) - len(i)), 3)


    # In[148]:


    # remove weak network
    
    # In[102]:


    # 60% of the data


    # In[39]:


    

    # In[111]:


    # 70% of the data


    
    

    def get_edges(self,sm):
        """
        get the edges and initialize the model
        """
        bn = BayesianNetwork(sm)
        edge_list = list(bn.edges)
        return edge_list,bn


    # In[19]:




    # In[21]:


    # Bayesian Networks in CausalNex support only discrete distributions.
    # So change to continuous value to discrete values and distributions
    # library for make dsicrete value

    def return_copy(self,df_feat,col):
        df_c = df_feat.copy()

        for i in range(len(col)):
            c = col[i]
            df_c[c] = Discretiser(method="fixed",
                                numeric_split_points=[df_c[c].quantile(0.5)]).transform(df_c[c].values)
        return df_c


    # In[22]:





    # In[26]:


    def discretise_data(self,features,df_feat):
        """
        this function transforms data into discrete data
        """
        tree_discretiser = DecisionTreeSupervisedDiscretiserMethod(
            mode='single',
            tree_params={'max_depth': 3, 'random_state': 27},
        )
        tree_discretiser.fit(
            feat_names=features,
            dataframe=df_feat,
            target_continuous=True,
            target='diagnosis',
        )
        discretised_data = df_feat.copy()
        for col in features:
            discretised_data[col] = tree_discretiser.transform(df_feat[[col]])
        return discretised_data


    # In[27]:




    # In[32]:


    def model_causalnet(self,bn,discretised_data):
        """
        models a causal network
        """
        train, test = train_test_split( discretised_data, train_size=0.8, test_size=0.2, random_state=27)
        bn = BayesianNetwork(bn.structure)
        bn = bn.fit_node_states(discretised_data)
        bn = bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")
        return bn


    # In[33]:




    def view_predictions(self,bn2, test,limit):
        """
        a function to help viewing how the outcomes look like manually
        """
        predictions = bn2.predict(test, "diagnosis");
        pred=pd.DataFrame()
        pred['predictions']=[i for i in predictions['diagnosis_prediction']]
        pred['true'] = [i for i in test['diagnosis']]
        return pred.head(limit)


    # In[73]:



    def get_tree(self,k_list):
        """
        Return the edges of a tree given the number of children at each level
        """
        n = 1
        edges_radial = []
        for level in range(0, len(k_list)):
            k = k_list[level]
            edges_radial.extend(
                [(f"A{level}_{i // k}", f"A{level + 1}_{i}") for i in range(k * n)]
            )
            n = k * n
        return edges_radial



    # In[78]:


    def draw_tree(self):
        """
        helps in the drawing of trees
        """
        layouts = [
            ("dot","Order nodes hierarchly. Great to spot the dependencies of a causal network.",[2, 3, 3]),
            ("neato", "Spring model. Great default tool if the graph is not too large", [2, 2, 4, 3, 2]),
            ("sfdp", "A different style of spring model", [2, 2, 4, 3, 2]),
            ("twopi", "Radial layout", [2, 2, 5, 3]),
        ]

        for layout, description, k_list in layouts:
            g_tree = StructureModel(self.get_tree(k_list))
            viz = plot_structure(g_tree)

            print(f"{layout}: {description}")
            image_binary = viz.draw(format="png", prog=layout)
            display(Image(image_binary, width=500))


    # In[206]:




    # In[82]:


    def draw_graph(self,sm):
        """
        draws a graph based on certain levels of nodes and edges,
        to produce visual perspective of final graph
        
        """
        
        graph_attributes = {
            "splines": "spline",  # I use splies so that we have no overlap
            "ordering": "out",
            "ratio": "fill",  # This is necessary to control the size of the image
            "size": "16,9!",  # Set the size of the final image. (this is a typical presentation size)
            "label": "The structure of our\n \t Diabetes model",
            "fontcolor": "#FFFFFFD9",
            "fontname": "Helvetica",
            "fontsize": 100,
            "labeljust": "l",
            "labelloc": "t",
            "pad": "1,1",
            "dpi": 200,
            "nodesep": 0.8,
            "ranksep": ".5 equally",
        }

        # Making all nodes hexagonal with black coloring
        node_attributes = {
            node: {
                "shape": "hexagon",
                "width": 2.2,
                "height": 2,
                "fillcolor": "#000000",
                "penwidth": "10",
                "color": "#4a90e2d9",
                "fontsize": 35,
                "labelloc": "c",
            }
            for node in sm.nodes
        }

        # Splitting two words with "\n"
        for node in sm.nodes:
            print(node)
            up_idx = [i for i, c in enumerate(node)][-1]
            node_attributes[node]["label"] = node[:up_idx] + "\n" + node[up_idx:]

        # Target nodes (ones with "Cost" in the name) are colored differently
        for node in sm.nodes:
            if "diagnosis" in node:  # We color nodes with "cost" in the name with a orange colour.
                node_attributes[node]["fillcolor"] = "#DF5F00"

        # Customising edges
        edge_attributes = {
            (u, v): {
                "penwidth": w * 20 + 2,  # Setting edge thickness
                "weight": int(5 * w),  # Higher "weight"s mean shorter edges
                "arrowsize": 2 - 2.0 * w,  # Avoid too large arrows
                "arrowtail": "dot",
            }
            for u, v, w in sm.edges(data="weight")
        }


        viz = plot_structure(
            sm,
            prog="dot",
            graph_attributes=graph_attributes,
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
        )
        f = "../images/final_model.jpg"
        viz.draw(f)
        return Image(f)


    # In[83]:


