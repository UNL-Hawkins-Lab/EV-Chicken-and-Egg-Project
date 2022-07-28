import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
#pysqldf = lambda q: sqldf(q, globals())
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings(action='once')
warnings.simplefilter(action='once', category=FutureWarning)


# # The Electric Vehicle and Charging Station Problem
# 
# Electric vehicle ownership is often referenced as exhibiting a "chicken and egg" behavior arising from the supply and demand relationship. Individual demand for electric vehicles is influenced by the available supply of charging points. Consumers are unwilling to purchase vehicles due to range anxiety and a perceived lack of charging stations. Suppliers are not incentivized to provide charging stations unless there is sufficient demand to warrant their cost. There is a clear role for public policy in such situations. The government deems electric vehicles as a solution to a public ill (i.e., climate change) and can incentivize either suppliers by providing installation subsidies or consumers by installing charging stations. While the problem has been recognized in the literature [@melliger2018], empirical analysis is minimal.
# 
# An important consideration to the analysis is how electric mobility system may differ from one based on fossil fuels. In the conventional private mobility model, the individual owns the vehicle and purchases fuel from centralized and privately owned refueling stations. In contrast, electric vehicles may be charged in the home using previously existing infrastructure. The presence of charging points in the home begs the questions 1) if (or to what extent) out-of-home charging stations are required for travel? and 2) to what extent is range anxiety a perception versus a reality?
# 
# According to the Bureau of Transportation Statistics, 98% of trips made in the US are less than 50 miles [@vehicletechnologyoffice2022]. Given that most battery-electric vehicles (BEVs) have a range greater than 200 miles [@elfalan2021], it is feasible to make most trips on a single charge. However, long-distance trips (over 50 miles) comprise 30% of total vehicle-miles traveled (VMT) [@aultman-hall2018]. There is clearly a need for out-of-home charging stations to accommodate these trips. Even if most trips can be accommodated by in-home charging, the vehicle purchase decision will be influenced by consideration of these longer trips that require charging stations [@silvia2016]. Additionally, Wolbertus et al. [@wolbertus2018] find that there is still a demand for charging stations in places where public daytime charging is the only option, such as at the workplace.
# 
# # Data Sources
# 
# We use a combination of open-source and purchased data in our analysis. The two key input datasets are charging station locations provided by the Alternative Fuel Data Center (AFDC) and electric vehicle registrations provided by Experian Inc. The vehicle registration dataset comprises a 10-year panel at 2-year increments (2012, 2014, 2016, 2018, 2020). Total vehicle registrations are recorded by county, make, model, year, and other vehicle characteristics for the United States.
# 
# After filtering out protectorates and other state-state locations, several county codes in the Experian data remain that are missing registration totals in a subsset of years (192/3172, or about 6%). We remove county codes with more than one missing year, leaving 186 for which registration totals are interpolated from adjacent years. Many of these county codes are for remote areas with low populations (e.g., the Aleutian Islands in Alaska and much of Idaho).

# In[2]:


#  https://gist.github.com/rogerallen/1583593
us_state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}


# In[3]:


def impute_veh_regs(df):
    geo_ct = df.GEOID.value_counts()==4
    cty_list = list(np.unique(df[df.GEOID.isin(geo_ct[geo_ct].index)].GEOID.values)) # record count is 4 for the county
    yr_set = set([2012,2014,2016,2018,2020])
    for cty in cty_list:
        temp = df[df.GEOID==cty]
        miss_yr = (yr_set-set(temp.YEAR)).pop()
        row = temp.iloc[0].copy() # initialize new row to the first row in dataset
        row.loc["YEAR"] = miss_yr
        if(miss_yr==2012):
            row[~row.index.isin(["GEOID","YEAR"])] = np.reshape(temp.loc[temp.YEAR==2014,~temp.columns.isin(["GEOID","YEAR"])].values - (temp.loc[temp.YEAR==2016,~temp.columns.isin(["GEOID","YEAR"])].values - temp.loc[temp.YEAR==2014,~temp.columns.isin(["GEOID","YEAR"])].values),-1)
        elif(miss_yr==2020):
            row[~row.index.isin(["GEOID","YEAR"])] = np.reshape(temp.loc[temp.YEAR==2018,~temp.columns.isin(["GEOID","YEAR"])].values + (temp.loc[temp.YEAR==2018,~temp.columns.isin(["GEOID","YEAR"])].values - temp.loc[temp.YEAR==2016,~temp.columns.isin(["GEOID","YEAR"])].values),-1)
        else:
            row[~row.index.isin(["GEOID","YEAR"])] = np.reshape((temp.loc[temp.YEAR==(miss_yr+2),~temp.columns.isin(["GEOID","YEAR"])].values + temp.loc[temp.YEAR==(miss_yr-2),~temp.columns.isin(["GEOID","YEAR"])].values)/2,-1)                                       
        df = pd.concat((df,row.to_frame().T),ignore_index=True)
    return(df)


# In[4]:


def fill_station_lags(df, id_col, yr_col, ch_col):
    cty_list = np.unique(df[:,id_col])
    for cty in cty_list:
        temp = df[df[:,id_col]==cty]
        yr_list = temp[:,yr_col]
        yr_set = set(yr_list)
        i = 0
        for yr in range(2012,2021,1):
            if(yr>np.min(yr_list)): # if the year is after the first installation year (prior years will be autofilled as zero when merged with registration data)
                if(yr not in yr_list): # if the year isn't in the list of years for the county
                    row = temp[i].copy()
                    row[yr_col] = yr
                    row[ch_col:ch_col+2] = 0 # no stations installed in this year
                    row_mat = row[np.newaxis]
                    df = np.vstack((df,row_mat))
                else:
                    i+=1 # update previous row id if the year is in temp df
    return(df)


# load in state population totals
df_pop = pd.read_csv("../Data/Census/PopByState/state_pop_est_2019.csv")
df_pop['STATE'] = df_pop['Geographic Area'].map(us_state_to_abbrev)

# load charging station data
df_ch_stn = pd.read_csv("../Data/Transport/alt_fuel_stations_w_county.csv")
print("df_ch_stn all",df_ch_stn.shape[0])
print("df_ch_stn no private",df_ch_stn.shape[0])
df_ch_stn = df_ch_stn[pd.notnull(df_ch_stn["Open Date"])]
print("df_ch_stn (after removing nan open year",df_ch_stn.shape[0])
df_ch_stn = df_ch_stn.assign(YEAR=pd.to_datetime(df_ch_stn.loc[:,'Open Date']).dt.year.astype(int))
#df_ch_stn = df_ch_stn.drop(columns="Open Date")
df_ch_stn = df_ch_stn[df_ch_stn.STATEFP<=56]
id_col = df_ch_stn.columns.get_loc("GEOID")
yr_col = df_ch_stn.columns.get_loc("YEAR")
ch_col = df_ch_stn.columns.get_loc("EVSE-L01")
df_ch_stn = pd.DataFrame(data=fill_station_lags(df_ch_stn.values, id_col, yr_col,ch_col), columns=df_ch_stn.columns)
grp_ch_stn = df_ch_stn.groupby(["GEOID","YEAR"]).sum()[["EVSE-L01","EVSE-L02","EVSE-L03"]].reset_index()
grp_ch_stn["cEVSE-L01"]=grp_ch_stn[["GEOID","EVSE-L01"]].groupby("GEOID").cumsum()
grp_ch_stn["cEVSE-L02"]=grp_ch_stn[["GEOID","EVSE-L02"]].groupby("GEOID").cumsum()
grp_ch_stn["cEVSE-L03"]=grp_ch_stn[["GEOID","EVSE-L03"]].groupby("GEOID").cumsum()
grp_ch_stn["cEVSE"]=grp_ch_stn["cEVSE-L01"]+grp_ch_stn["cEVSE-L02"]+grp_ch_stn["cEVSE-L03"]

# load vehicle registrations data
df_veh_reg = pd.read_parquet("../Data/Transport/Experian Registrations/sum_registrations.parquet")
df_veh_reg.rename(columns={"year":"YEAR"},inplace=True)
# fill nan with zero and aggregate electricity columns
df_veh_reg.fillna(0, inplace=True)
df_veh_reg = impute_veh_regs(df_veh_reg)
df_veh_reg = df_veh_reg.assign(bev=df_veh_reg["24kw Electric~Electric"]+df_veh_reg["60kw Electric~Electric"]+df_veh_reg["85kw Electric~Electric"]+df_veh_reg["90kw Electric~Electric"]+df_veh_reg["Electric"]+df_veh_reg["Electric Fuel System"])
df_veh_reg = df_veh_reg.assign(pev=(df_veh_reg["bev"]+df_veh_reg["Plug-In Hybrid"]))
# percent bev is bev/total vehicles
df_veh_reg = df_veh_reg.assign(per_bev=df_veh_reg["bev"] / df_veh_reg["All"])
# percent pev is (bev+phev)/total vehicles
df_veh_reg = df_veh_reg.assign(per_pev=df_veh_reg["pev"] / df_veh_reg["All"])

# combine registration and charging data by year and county code
df = df_veh_reg.merge(grp_ch_stn.loc[:,["GEOID","YEAR","cEVSE-L01","cEVSE-L02","cEVSE-L03","cEVSE"]],how="left",on=["GEOID","YEAR"])
# add additional columns for lagged charging station counts
df["lag_year"] = df.YEAR-1
df = df.merge(grp_ch_stn.loc[:,["GEOID","YEAR","cEVSE-L01","cEVSE-L02","cEVSE-L03","cEVSE"]],how="left",left_on=["GEOID","lag_year"],right_on=["GEOID","YEAR"],suffixes=("","_lag"))
df["GEOID"] = df["GEOID"].astype(int)

# read in demographic data by county and add to main dataframe
df_pop11 = pd.read_csv("../Data/Census/county_pop_race_age_2011_2015.csv")
df_pop15 = pd.read_csv("../Data/Census/county_pop_race_age_2015_2019.csv") 
df_dem = pd.read_csv("../Data/Census/county_race_income.csv")
df_dem["Percent_Minority"] = (df_dem["ADK5E004"]+df_dem["ADK5E005"]+df_dem["ADK5E007"]+df_dem["ADK5E012"]+df_dem["ADK5E008"])/df_dem["ADK5E001"]
df_dem.loc[df_dem.YEAR==2012,"ABDPE001"] = df_dem.loc[df_dem.YEAR==2012,"ABDPE001"]*1.08 # adjust to 2019$ rather than 2014$ https://data.bls.gov/cgi-bin/cpicalc.pl?cost1=1.00&year1=201411&year2=201911
df_dem14 = df_dem[df_dem.YEAR==2012]
df_dem14 = df_dem14.assign(YEAR=2014)         
df_dem18 = df_dem[df_dem.YEAR==2016]
df_dem18 = df_dem18.assign(YEAR=2018)  
df_dem20 = df_dem[df_dem.YEAR==2020]
df_dem20 = df_dem20.assign(YEAR=2020)  
df_dem = pd.concat((df_dem,df_dem14,df_dem18,df_dem20))

# read in egrid data and add to main dataframe
df_egrid = pd.read_csv("../Data/eGrid/egrid_data_co2.csv")

# read in elections data and add to main dataframe
df_elec = pd.read_csv("../Data/Elections/county_pres_2020.csv")
df_elec = df_elec[df_elec.PARTY=="DEMOCRAT"]
df_elec["percent_dem"] = df_elec.CAND_VOTES/df_elec.TOT_VOTES

# join demographics to main dataframe for 2011 to 2015
df11 = df.loc[df.YEAR<2015,:].merge(df_pop11, how="left", on="GEOID")
# update data for YEAR 2015 forward to use the 2015-2019 data
df15 = df.loc[df.YEAR>=2015,:].merge(df_pop15, how="left", on="GEOID")
df = pd.concat((df11,df15),axis=0)
df = df.merge(df_dem,how="left",on=["GEOID","YEAR"])
# some data are assigned county codes that don't appear in the population dataset. They should be removed for analysis.
df.dropna(axis=0,subset="ALUBE001", inplace=True)
# join egrid and elections data to main dataframe
df = df.merge(df_elec, how="left",on="GEOID")
df = df.merge(df_egrid, how="left", left_on=["STATEA","YEAR"], right_on=["FIPSST","YEAR"])

# calculate per capita statistics (per 100,000 inhabitants)
df["bev_cap"] = (df["bev"]/df["ALUBE001"])*100000
df["pev_cap"] = (df["pev"]/df["ALUBE001"])*100000
df["cEVSE-L01_cap"] = (df["cEVSE-L01"]/df["ALUBE001"])*100000
df["cEVSE_L02_cap"] = (df["cEVSE-L02"]/df["ALUBE001"])*100000
df["cEVSE_L03_cap"] = (df["cEVSE-L03"]/df["ALUBE001"])*100000
df["cEVSE_cap"] = (df["cEVSE"]/df["ALUBE001"])*100000
df["cEVSE_cap_lag"] = (df["cEVSE_lag"]/df["ALUBE001"])*100000

df.fillna(0,inplace=True) # careful using a blanket fillna statement on a dataframe
df.sort_values(by=["GEOID","YEAR"],inplace=True)
temp = df.GEOID.value_counts()==5 # data available for all years. Some remote areas and reservations do not have data available for all years
df = df[df.GEOID.isin(temp[temp].index.get_level_values(0).values)]
temp = df.groupby("GEOID").sum()[["cEVSE_cap","pev_cap"]]
temp = temp[(temp.cEVSE_cap>0)&(temp.pev_cap>0)]
df = df[df.GEOID.isin(temp.index.get_level_values(0).values)] # filter out counties where there are no PEVs or charging stations in any analysis year

# Mediation effects

# In[7]:


from causal_curve import Mediation
med = Mediation(
    bootstrap_draws=500,
    bootstrap_replicates=100,
    spline_order=3,
    n_splines=5,
    random_seed=111,
    verbose=True,
    treatment_grid_num=6
)

med.fit(
    T=df["cEVSE_cap"],
    M=df["percent_dem"],
    y=df["pev_cap"],
)

med_results = med.calculate_mediation(ci = 0.95)