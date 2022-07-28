import pandas as pd
import datetime
import numpy as np


def impute_veh_regs(df):
    geo_ct = df.GEOID.value_counts()==4
    cty_list = list(np.unique(df[df.GEOID.isin(geo_ct[geo_ct].index)].GEOID.values)) # record count is 4 for the county
    yr_set = set([2012,2014,2016,2018,2020])
    for cty in cty_list:
        temp = df[df.GEOID==cty]
        miss_yr = (yr_set-set(temp.year)).pop()
        row = temp.iloc[0].copy() # initialize new row to the first row in dataset
        row.loc["year"] = miss_yr
        if(miss_yr==2012):
            row[~row.index.isin(["GEOID","year"])] = np.reshape(temp.loc[temp.year==2014,~temp.columns.isin(["GEOID","year"])].values - (temp.loc[temp.year==2016,~temp.columns.isin(["GEOID","year"])].values - temp.loc[temp.year==2014,~temp.columns.isin(["GEOID","year"])].values),-1)
        elif(miss_yr==2020):
            row[~row.index.isin(["GEOID","year"])] = np.reshape(temp.loc[temp.year==2018,~temp.columns.isin(["GEOID","year"])].values + (temp.loc[temp.year==2018,~temp.columns.isin(["GEOID","year"])].values - temp.loc[temp.year==2016,~temp.columns.isin(["GEOID","year"])].values),-1)
        else:
            row[~row.index.isin(["GEOID","year"])] = np.reshape((temp.loc[temp.year==(miss_yr+2),~temp.columns.isin(["GEOID","year"])].values + temp.loc[temp.year==(miss_yr-2),~temp.columns.isin(["GEOID","year"])].values)/2,-1)                                       
        df = pd.concat((df,row.to_frame().T),ignore_index=True)
    return(df)

# load vehicle registrations data
df_veh_reg = pd.read_parquet("../Data/Transport/Experian Registrations/sum_registrations.parquet")
print(df_veh_reg.shape)
# fill nan with zero and aggregate electricity columns
df_veh_reg.fillna(0, inplace=True)
# impute vehicle data if not available for a year (only when there is just 1 missing year)
df_veh_reg = impute_veh_regs(df_veh_reg)
print(df_veh_reg.shape)