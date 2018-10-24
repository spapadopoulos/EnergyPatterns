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
import math
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

def loadData():
    ll87 = pd.read_csv(r'../data/LL87_BBL.csv')
    ll87 = ll87[(ll87.Year == 2013) | (ll87.Year == 2014)]

    # Load boiler data
    boil = pd.read_csv(r'../data/boilers_parsed.csv')
    boil_bbl = list(boil.bbl)

    feats = ['BBL10_2011', 'Total_Floor_Space_Sq_Ft_2011', 'Facility_Type_2011', 
                 'Service_and_Product_Provider_2011', 'Metering_Configuration_2011', 'Current_Rating_1_100_2011',
                 'Current_Total_GHG_Emissions_MtCO2e_2011', 'Current_Total_Site_Energy_Use_kBtu_2011', 'Weather_Normalized_Site_EUI_kBtu/Sq_Ft_2011', 
                 'Electricity_Use_kBtu_2011','Natural_Gas_Use_kBtu_2011', 'District_Steam_Use_kBtu_2011', 
                 'Fuel_Oil_2_Use_kBtu_2011','Fuel_Oil_4_Use_kBtu_2011', 'Fuel_Oil_56_Use_kBtu_2011',
                 'Current_Weather_Normalized_Source_Energy_Intensity_kBtu/Sq_Ft_2011','Multifamily_Home__Laundry_in_common_area_2011','Multifamily_Home__Laundry_in_each_unit_2011',
                 'Multifamily_Home__Government_Subsidized_Housing?_Y=1__N=0_2011', 'Multifamily_Home__Number_of_units_2011', 'Office__PC_Density_2011', 
                 'Office__Weekly_operating_hours_2011','Office__Workers_Density_2011','Total_GHG_Emissions_MtCO2e_2012',
                 'Weather_Normalized_Site_EUI_kBtu/ft_2_2012','Weather_Normalized_Source_EUI_kBtu/ft_2_2012', 'Weather_Normalized_Site_EUI_kBtu/ft_2_2013', 
                 'Weather_Normalized_Source_EUI_kBtu/ft_2_2013', 'Total_GHG_Emissions_Metric_Tons_CO2e_2013','Weather_Normalized_Site_EUI_kBtu/ft_2_2014',
                 'Weather_Normalized_Source_EUI_kBtu/ft_2_2014', 'Total_GHG_Emissions_Metric_Tons_CO2e_2014','Weather_Normalized_Site_EUI_kBtu/ft_2_2015', 
                 'Weather_Normalized_Source_EUI_kBtu/ft_2_2015', 'Total_GHG_Emissions_Metric_Tons_CO2e_2015', 
                 'ENERGY_STAR_Score_2016', 'Weather_Normalized_Site_EUI_kBtu/ft_2_2016', 'Site_Energy_Use_kBtu_2016',
                 'Weather_Normalized_Source_EUI_kBtu/ft_2_2016','Fuel_Oil_#2_Use_kBtu_2016', 'Fuel_Oil_#4_Use_kBtu_2016', 'Fuel_Oil_#5_&_6_Use_kBtu_2016', 
                 'District_Steam_Use_kBtu_2016', 'Natural_Gas_Use_kBtu_2016','Electricity_Use___Grid_Purchase_kBtu_2016',
                 'Total_GHG_Emissions_Metric_Tons_CO2e_2016', 'Data_Center___Gross_Floor_Area_ft_2_2016', 'Multifamily_Housing___Number_of_Laundry_Hookups_in_All_Units_2016',
                 'Multifamily_Housing___Number_of_Laundry_Hookups_in_Common_Areas_2016','Multifamily_Housing___Total_Number_of_Residential_Living_Units_2016', 
                 'Office___Gross_Floor_Area_ft_2_2016','Office___Number_of_Computers_2016','Office___Number_of_Workers_on_Main_Shift_2016', 
                 'Office___Weekly_Operating_Hours_2016', 'Office___Worker_Density_Number_per_1_000_ft_2_2016',
                 'Organization_2016','Borough','ZipCode', 'BldgArea','NumFloors', 'AssessLand','AssessTot','YearBuilt', 'BuiltFAR', 'XCoord','YCoord']

    # load LL84 data
    df = pd.read_csv('../data/ll84_11_16_BBL_BIN_pluto_v2.csv', usecols=feats)

    new_cols = ['BBL', 'area', 'FacilityType', 'provider', 'meter', 'ess11', 'ghg11', 'totener11', 'site11','ElecUse11',
                    'NGUse11','SteamUse11', 'fuel211','fuel411','fuel5611', 'eui11','comLaund11','unitLaund11','GovSubsidized','units11',
                    'pcdens11', 'ophours11','workdens11', 'ghg12', 'site12', 'eui12', 'site13', 'eui13', 'ghg13', 'site14', 
                    'eui14', 'ghg14', 'site15', 'eui15', 'ghg15', 'ess16', 'site16', 'totener16','eui16',
                    'fuel216', 'fuel416', 'fuel5616', 'SteamUse16', 'NGUse16','ElecUse16', 'ghg16', 'dataCenterArea','unitLaund16','comLaund16',
                    'units16','officeGFA16','computers16','workers16', 'ophours16', 'workdens16', 'organization','Borough', 'ZipCode', 'BldgArea', 'NumFloors', 
                    'AssessLand', 'AssessTot', 'YearBuilt', 'BuiltFAR', 'XCoord','YCoord'] 

    df.columns = new_cols
    return df, ll87, boil_bbl


def featureEngineering(df, ll87, boil_bbl, typology):

    # Convert units
    df.area = df.area * 0.092903 # ft2 to m2
    df.BldgArea = df.BldgArea * 0.092903 
    df.officeGFA16 = df.officeGFA16 * 0.092903 
    df.site11 = df.site11 * 3.1545913 # kWh/m2
    df.site12 = df.site12 * 3.1545913
    df.site13 = df.site13 * 3.1545913
    df.site14 = df.site14 * 3.1545913
    df.site15 = df.site15 * 3.1545913
    df.site16 = df.site16 * 3.1545913

    df.pcdens11 = (df.pcdens11*100)/92.903
    df.workdens11 = (df.workdens11*100)/92.903 # workers/100m2
    df.workdens16 = (df.workdens16*100)/92.903

    df = df[df.YearBuilt > 1800]
    df['totLaund11'] = (df['comLaund11'] + df['unitLaund11'])/df.area*100 # laundry/100m2
    df['totLaund16'] = (df['comLaund16'] + df['unitLaund16'])/df.area*100

    # Calculate computer density
    df['pcdens16'] = df.computers16/df.officeGFA16*100 # computers/100m2

    # Calculate value/sqft
    df['valuePerM2'] = df.AssessTot/df.BldgArea

    # Calculate unit density
    df['unitDens11']= df.units11/df.area*1000 # Units/1000m2
    df['unitDens16']= df.units16/df.area*1000

    df = df[(df.FacilityType == typology)]

    df['ZipCode'] = pd.to_numeric(df['ZipCode'], errors='coerce')
    df = df[np.isfinite(df['ZipCode'])]
    df['ZipCode']=df['ZipCode'].map(lambda x:int(x))

    # Find properties that reported in LL87 in 2013 or 2014

    def is_LL87(x):
        if x in list(ll87.BBL):
            val = 1
        else:
            val = 0
        return val

    df['is_LL87'] = df.BBL.apply(is_LL87)


    # Find properties with #4 or #6 boilers
    def has_boiler(x):
        if x in boil_bbl:
            val = 1
        else:
            val = 0
        return val

    df['has_boiler'] = df.BBL.apply(has_boiler)

    if typology == 'Office':
        carbChal = ['Durst Organization', 'Fisher Brothers', 'Forest City Ratner Companies', 'Hines', 'Industry City',
                      'Normandy Real Estate Partners', 'Related Companies', 'Rockefeller Group', 'Rudin Management Company, Inc.',
                      'RXR Realty', 'Silverstein Properties, Inc.', 'SL Green Realty Corp.','Vornado Realty Trust']
    else:
        carbChal = ['A&E Real Estate', 'AKAM Associates, Inc.', 'Charles H. Greenthal Management Corp.',
                      'Community League of the Heights', 'Douglas Elliman Property Management',
                      'FirstService Residential', 'Harlem Congregations for Community Improvement, Inc.',
                      'Lott Community Development Corporation', 'Marion Scott Real Estate, Inc.', 'Milford Management',
                      'New Holland Residences', 'New York City Housing Authority (NYCHA)', 'Prestige Management',
                      'RiseBoro Community Partnership', 'Riverbay Corporation at Co-op City', 'Rose Associates',
                      'Selfhelp Community Services Inc.', 'Solstice Residential', 'StuyTown Property Services','Urban American']


    def similar(a, b):
        """Returns the similarity ratio between two strings"""
        return SequenceMatcher(None, a, b).ratio()

    
    uniqueOrganization = list(df.organization.unique())
    uniqueProvider = list(df.provider.unique())


    def match_provider(x):
        """Matches similar energy service providers names"""
        for i in uniqueProvider:
            if similar(x, i)>0.7:
                val = i
                break
            else:
                val = x     
        return val  

    def match_organization(x):
        """Matches similar organization names"""
        for i in uniqueOrganization:
            if similar(x, i)>0.7:
                val = i
                break
            else:
                val = x       
        return val  

        
    def is_nycCC(x):
        """Get buildings managed by a company participating in NYC Carbon Challenge."""

        if b_any(x[0:15].lower() in comp.lower() for comp in carbChal)==True:
            val = 1
        else:
            val = 0
        return val
    

    # Identify buildings participating in NYC Carbon Challenge, 
    df['NYC_CC']  = df.organization.apply(is_nycCC)
    
    # Standardize organization and energy service provider information
    df['organizationNEW'] = df.organization.apply(match_organization)
    df['providerNEW'] = df.provider.apply(match_provider)
    
    
    # Find properties managed by a major real estate firm
    
    groupedRE = df.groupby('organizationNEW').BBL.count()
    groupedProv = df.groupby('providerNEW').BBL.count()
    
    top5organ = list(groupedRE.sort_values(ascending=False).index[0:5])
    top5providers = list(groupedProv.sort_values(ascending=False).index[1:6])
    
    majorRE = groupedRE[groupedRE>len(df)*0.05] # Select firms with more than 5% market share 
    #majorRE = groupedRE[groupedRE>10]

    def top5_ORG(x):
        """Get buildings managed by the top 5 mostly encountered developers"""
        if x in top5organ:
            val = 1
        else:
            val = 0
        return val

    def top5_PROV(x):
        """Get buildings managed by the top 5 mostly encountered energy service providers"""
        if x in top5providers:
            val = 1
        else:
            val = 0
        return val
    
    df['top5organizations']  = df.organizationNEW.apply(top5_ORG)
    df['top5providers']  = df.providerNEW.apply(top5_PROV)
    
    def major_real_estate(x):
        """Get buildings managed by a company that represents more than 10 buildings"""
        if x in majorRE.index:
            val = 1
        else:
            val = 0
        return val

    df['majorRealEstate']  = df.organizationNEW.apply(major_real_estate)


    # Flag properties in Manhattan

    def is_Manhattan(x):
        if x == "MN":
            val = 1
        else:
            val = 0
        return val
    
    def is_not_Manhattan(x):
        if x != "MN":
            val = 1
        else:
            val = 0
        return val

    df['in_Manhattan']  = df.Borough.apply(is_Manhattan)
    df['not_in_Manhattan']  = df.Borough.apply(is_not_Manhattan)

    def has_datacenter(x):
    	"""Flags commercial buildings with data center space"""
        if math.isnan(x):
            val = 0
        else:
            val = 1
        return val

    df['has_datacenter']  = df.dataCenterArea.apply(has_datacenter)


    def trans_loc15(x):
    	"""Standardizes coordinates to be reported as int"""
        try:
            val = int(x)
            return val
        except:
            return np.nan

    df['XCoord'] = df.XCoord.apply(trans_loc15)
    df['YCoord'] = df.YCoord.apply(trans_loc15)

    # Transform to lat/lon from NYS coordinates
    lat=[]
    lon=[]
    NYSP1983 = pyproj.Proj(init="ESRI:102718", preserve_units=True)
    for i in df.index:
        x, y = (df.XCoord[i], df.YCoord[i])
        lat.append(NYSP1983(x, y, inverse=True)[0])
        lon.append(NYSP1983(x, y, inverse=True)[1])

    df['lat'] = lat
    df['lon'] = lon

    # difference of features 2010-2015
    df['percent_diff_eui_11_16'] = ((df.eui16-df.eui11)/df.eui11)*100 # Negative values means improvement in EUI
    df['percent_diff_site_11_16'] = ((df.site16-df.site11)/df.site11)*100
    df['percent_diff_ghg_11_16'] = ((df.ghg16-df.ghg11)/df.ghg11)*100 
    
    df.totener11 = df.totener11 * 0.293071
    df.totener16 = df.totener16 * 0.293071
    df['energy_diff'] = df.totener11 - df.totener16

    return df



def getTimeSeries(df):
	"""Get normalized EUI times series"""

	df_eui_ts = df[['site11','site12','site13','site14','site15', 'site16']]
	df_eui_ts.reset_index(inplace=True)
	df_eui_ts = df_eui_ts.drop('index',axis=1)

	def normal_ts(series):
	    """Normalize time series"""
	    norm_series = (series - min(series))/(max(series) - min(series))
	    return norm_series

	norm_ts = df_eui_ts.apply(normal_ts, axis=1)
	return norm_ts