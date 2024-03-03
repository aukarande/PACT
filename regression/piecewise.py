from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor

class PiecewiseRegression():
    #piecewise regression according to clusters
    def __init__(self, numClusters, featureKeys, clusteringKeys = None) -> None:
        #pass in the position (column index) of the desired keys for kmeans
        self.numClusters = numClusters
        self.km = KMeans(n_clusters = self.numClusters, max_iter = 1000, algorithm = "elkan")
        self.models = {} #models for each cluster
        self.featureKeys = featureKeys
        self.allLabels = []
        self.clusteringKeys = featureKeys if clusteringKeys == None else clusteringKeys
        self.scaler = MinMaxScaler(feature_range=(0, 100))

    def fit(self, X, Y):
        #fit k_means
        X = X.to_numpy() if isinstance(X, pd.DataFrame) == True else X
        xScaled = pd.DataFrame(self.scaler.fit_transform(X), columns=self.featureKeys)
        self.km.fit(xScaled[self.clusteringKeys])
        predictedClusters = self.km.predict(xScaled[self.clusteringKeys])
        self.allLabels = np.unique(predictedClusters)
        
        X = self.scaler.transform(X)
        
        for i in range(len(self.allLabels)):
            # print(self.allLabels[i], len(X[predictedClusters == self.allLabels[i]]), X.shape)
            self.models[self.allLabels[i]] = LinearRegression()
            self.models[self.allLabels[i]].fit(X[predictedClusters == self.allLabels[i]], Y[predictedClusters == self.allLabels[i]])

    def predict(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) == True else X
        xScaled = pd.DataFrame(self.scaler.transform(X), columns=self.featureKeys)

        X = self.scaler.transform(X)
        
        predictedClusters = self.km.predict(xScaled[self.clusteringKeys])
        labels = self.allLabels
        prediction = np.zeros((X.shape[0], 1))
        for i in range(len(labels)):
            if (sum(predictedClusters == labels[i]) != 0): # check if the cluster exist in the prediction 
                prediction[predictedClusters == labels[i]] = self.models[labels[i]].predict(X[predictedClusters == labels[i]])
        return prediction
    
    def scaleClusterData(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) == True else X
        xScaled = pd.DataFrame(self.scaler.transform(X), columns=self.featureKeys)
        predictedClusters = self.km.predict(xScaled[self.clusteringKeys])
        return self.scaler.transform(X), self.allLabels, predictedClusters