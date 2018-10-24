import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import warnings
import operator
from  __builtin__ import any as b_any
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MeanShift
from sklearn.cluster import DBSCAN, SpectralClustering, AffinityPropagation
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist, pdist
from sklearn import preprocessing
import random
from sklearn.metrics.pairwise import pairwise_distances
import math
import pyproj
import seaborn.apionly as sns
from scipy import stats
from fiona.crs import from_epsg
#import pysal
import shapely
#from pysal.esda.getisord import G_Local
import statsmodels.api as sm
from difflib import SequenceMatcher


def dunn(c, distances):

    unique_cluster_distances = np.unique(min_cluster_distances(c, distances))
    max_diameter = max(diameter(c, distances))

    if np.size(unique_cluster_distances) > 1:
        return unique_cluster_distances[1] / max_diameter
    else:
        return unique_cluster_distances[0] / max_diameter

def min_cluster_distances(c, distances):
    """Calculates the distances between the two nearest points of each cluster"""
    min_distances = np.zeros((max(c) + 1, max(c) + 1))
    for i in np.arange(0, len(c)):
        if c[i] == -1: continue
        for ii in np.arange(i + 1, len(c)):
            if c[ii] == -1: continue
            if c[i] != c[ii] and distances[i, ii] > min_distances[c[i], c[ii]]:
                min_distances[c[i], c[ii]] = min_distances[c[ii], c[i]] = distances[i, ii]
    return min_distances

def diameter(c, distances):
    """Calculates cluster diameters (the distance between the two farthest data points in a cluster)"""
    diameters = np.zeros(max(c) + 1)
    for i in np.arange(0, len(c)):
        if c[i] == -1: continue
        for ii in np.arange(i + 1, len(c)):
            if c[ii] == -1: continue
            if c[i] != -1 or c[ii] != -1 and c[i] == c[ii] and distances[i, ii] > diameters[c[i]]:
                diameters[c[i]] = distances[i, ii]
    return diameters


def computeSilhouetteDunnScores(data, cases):
    dunn_score=[]
    silhouette_avg =[]
    
    range_n_clusters = np.arange(2,cases,1)
    
    for clusters in range_n_clusters:
        # Fit the model
        km = KMeans(n_clusters = clusters, random_state = 14)
        # Predict labels
        cluster_labels = km.fit_predict(data)
        # Compute silhouette and dunn scores
        dunn_score.append(dunn(cluster_labels, pairwise_distances(data)))
        silhouette_avg.append(silhouette_score(data, cluster_labels))
    return(dunn_score, silhouette_avg)


# Get Dunn and Silhouette scores


# dunnScore, silScore = computeSilhouetteDunnScores(norm_ts, 11)

# Save them
# dfScores = pd.DataFrame({'Dunn':dunnScore, 'Silhouette':silScore})
# dfScores.to_csv(r'..\output\scores_'+ typology +'.csv')


def elbow(data,K):
#data is your input as numpy form
#K is a list of number of clusters you would like to show.
    # Run the KMeans model and save all the results for each number of clusters
    '''
    Function that calculates and plots the 'elbow' for kmeans clustering.
    '''
    KM = [KMeans(n_clusters=k).fit(data) for k in K]

    # Save the centroids for each model with a increasing k
    centroids = [k.cluster_centers_ for k in KM]

    # For each k, get the distance between the data with each center. 
    D_k = [cdist(data, cent, 'euclidean') for cent in centroids]

    # But we only need the distance to the nearest centroid since we only calculate dist(x,ci) for its own cluster.
    globals()['dist'] = [np.min(D,axis=1) for D in D_k]

    # Calculate the Average SSE.
    avgWithinSS = [sum(d)/data.shape[0] for d in dist]

    # Include Dunn index
    dunn_score = [dunn(KMeans(n_clusters=k).fit_predict(data), pairwise_distances(data)) for k in range(K[1], max(K)+1)]

    # Find optimal value
    clusDict = {key: i for i, key in enumerate(dunn_score)}


    # Elbow curve
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(K, avgWithinSS, '-*', color = 'navy', label='SSE')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Average within-cluster SSE')
    #ax1.legend(loc=1)
    ax2 = ax1.twinx()
    ax2.bar(range(K[1],max(K)+1), dunn_score, color = 'navy', alpha=0.2, label='Dunn index')
    for i, v in enumerate(dunn_score):
        ax2.text(i+1.8, v+0.03, str(round(v,2)), color='black',size=12)#, fontweight='bold')
    ax2.set_ylabel('Dunn Index')
    ax2.axis('off')
    ax1.grid(True,ls='--',lw=1, alpha=0.9)

    ax2.legend(loc=1)
    plt.title('Elbow plot and Dunn index for KMeans clustering')
    #plt.savefig('..\output\elbowOF.png')
    plt.show()

    return max(clusDict.iteritems(), key=operator.itemgetter(0))[1] + 2


def visualizeClusters(optClus, norm_ts, df, typology):
    kmOpt = KMeans(n_clusters = optClus, random_state = 14)

    df['cluster'] = kmOpt.fit_predict(norm_ts)

    # Dataframes for viz
    dfViz = df[['site11', 'site12', 'site13', 'site14', 'site15', 'site16', 'cluster']]
    dfViz1 = dfViz[dfViz.cluster == 0]
    dfViz1 = dfViz1.drop('cluster', axis=1)
    dfViz1.reset_index(inplace=True)
    dfViz1 = dfViz1.drop('index',axis=1)
    dfViz1Pl = np.asarray(dfViz1)

    dfViz2 = dfViz[dfViz.cluster == 1]
    dfViz2 = dfViz2.drop('cluster', axis=1)
    dfViz2.reset_index(inplace=True)
    dfViz2 = dfViz2.drop('index',axis=1)
    dfViz2Pl = np.asarray(dfViz2)


    
    plt.figure(figsize=(15,9))
    sns.tsplot(data=dfViz1Pl, err_style="boot_kde", lw=3, ls='--', estimator=np.median,color=sns.color_palette("viridis")[0], legend=True, 
               condition='cluster 0 (n=%i)'%len(dfViz1Pl))
    sns.tsplot(data=dfViz2Pl, err_style="boot_kde", lw=3, estimator=np.median,color=sns.color_palette("viridis")[2], legend=True, 
               condition='cluster 1 (n=%i)'%len(dfViz2Pl))
    plt.tick_params(axis='both',bottom='off',top='off',left='on',right='off')
    plt.xticks(range(6), ['2011', '2012', '2013', '2014', '2015', '2016'])
    plt.ylabel('Site EUI (kWh/sq.m.)', size=20)
    plt.xlabel('Year', size=20)
    plt.title("Time series' clusters (%s)"%typology, size=26)
    plt.grid(True,ls='--',lw=1, alpha=0.8)
    plt.legend(fontsize=16)
    plt.show()  

    return df


def contigT(df, var):
    """Creates the contigency table for Fisher's exact test."""
    return [[df[var][df.cluster == 0].sum(), df[var][df.cluster == 1].sum()],
        [len(df[var][df.cluster == 0]) - df[var][df.cluster == 0].sum(), 
         len(df[var][df.cluster == 1]) - df[var][df.cluster == 1].sum()]]     

def fisherTests(df, typology):

    featuresToTestFisher = ['is_LL87', 'has_boiler', 'majorRealEstate', 'NYC_CC', 'in_Manhattan', 'not_in_Manhattan', 
                            'top5organizations','top5providers']

    bldCl0 = df[df.cluster == 0].reset_index(drop=True)
    bldCl1 = df[df.cluster == 1].reset_index(drop=True)
    
    if typology == 'Office':
        featuresToTestFisher.append('has_datacenter')
    fisherStat = []
    fisherPvalue = []
    ratioCluster0 = []
    ratioCluster1 = []
    feats =[]
    for feature in featuresToTestFisher:
        print "Fisher's test for feature -" + feature + "- (p-value): ", stats.fisher_exact(contigT(df, feature))[1]
        feats.append(feature)
        ratioCluster0.append(bldCl0[feature].sum()/(len(bldCl0)+0.0))
        ratioCluster1.append(bldCl1[feature].sum()/(len(bldCl1)+0.0))
        fisherStat.append(stats.fisher_exact(contigT(df, feature))[0])
        fisherPvalue.append(stats.fisher_exact(contigT(df, feature))[1])
    
    dfFish = pd.DataFrame({'Feature': feats, 'Cluster 0 ratio': ratioCluster0, 'Cluster 1 ratio': ratioCluster1,  
                           'Fisher statistic': fisherStat, 'Fisher p-value': fisherPvalue})
    return dfFish


def mannwhitneyTests(df, typology):
    if typology == 'Office':
        featuresToTestMann = ['site11', 'site16', 'ess11', 'ess16', 'area', 'pcdens11', 'pcdens16', 'ophours11', 'ophours16', 
                         'workdens11', 'workdens16', 'NumFloors', 'YearBuilt', 'valuePerM2']
    else:
        featuresToTestMann = ['site11', 'site16', 'ess16', 'area', 'totLaund11', 'totLaund16', 'units11', 
                         'unitDens11', 'unitDens16', 'NumFloors', 'YearBuilt', 'valuePerM2']
    mannStat = []
    mannPvalue = []
    medianCluster0 = []
    medianCluster1 = []
    feats = []
    for feature in featuresToTestMann:
        print "Mann-Whitney test for feature -" +feature +"- (p-value):", stats.mannwhitneyu(df[feature][df.cluster==0], 
                                                                                           df[feature][df.cluster==1])[1]
        medianCluster0.append(df[feature][df.cluster==0].median())
        medianCluster1.append(df[feature][df.cluster==1].median())
        mannStat.append(stats.mannwhitneyu(df[feature][df.cluster==0], df[feature][df.cluster==1])[0])
        mannPvalue.append(stats.mannwhitneyu(df[feature][df.cluster==0], df[feature][df.cluster==1])[1])
        feats.append(feature)
    dfMann = pd.DataFrame({'Feature': feats,'Cluster 0 median': medianCluster0,'Cluster 1 median': medianCluster1, 
                           'Mann-Whitney statistic': mannStat,'Mann-Whitney p-value': mannPvalue})
    
    return dfMann
