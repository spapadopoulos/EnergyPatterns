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
from scipy import stats
from sklearn.metrics.pairwise import pairwise_distances
import warnings
import pyproj
import seaborn.apionly as sns
from scipy import stats
import geopandas as gpd
from fiona.crs import from_epsg
#import pysal
import shapely
#from pysal.esda.getisord import G_Local
import statsmodels.api as sm
from difflib import SequenceMatcher