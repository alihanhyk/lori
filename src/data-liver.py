from datetime import datetime
import dill
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
import random

df = pd.read_csv('data/liver.csv', na_values=' ', low_memory=False)

df.INIT_DATE = pd.to_datetime(df.INIT_DATE, format='%m/%d/%Y')
df.TX_DATE = pd.to_datetime(df.TX_DATE, format='%m/%d/%Y')
df.END_DATE = pd.to_datetime(df.END_DATE, format='%m/%d/%Y')
df.DEATH_DATE = pd.to_datetime(df.DEATH_DATE, format='%m/%d/%Y')

df = df[df.INIT_DATE >= datetime(2019,1,1)]
df = df[df.END_DATE <= datetime(2020,1,1)]
df = df[df.INIT_AGE >= 18]

df = df[df.DON_TY.isna() | (df.DON_TY == 'C')]
df = df[df.MULTIORG.isna() | (df.MULTIORG != 'Y')]
df = df[df.PREV_TX_ANY.isna() | (df.PREV_TX_ANY != 'Y')]

df.GENDER = df.GENDER.replace({'F': 1, 'M': 0})
df.HCV_NAT = df.HCV_NAT.replace({'P': 1, 'N': 0, 'ND': 0, 'U': np.nan})
df.PREV_AB_SURG_TRR = df.PREV_AB_SURG_TRR.replace({'Y': 1, 'N': 0, 'U': np.nan})
df.PREV_AB_SURG_TCR = df.PREV_AB_SURG_TCR.replace({'Y': 1, 'N': 0, 'U': np.nan})
df.ENCEPH_TX = df.ENCEPH_TX.replace({1: 0, 2: 1, 3: 1, 4: np.nan})
df.INIT_ENCEPH = df.INIT_ENCEPH.replace({1: 0, 2: 1, 3: 1, 4: np.nan})
df.ASCITES_TX = df.ASCITES_TX.replace({1: 0, 2: 1, 3: 1, 4: np.nan})
df.INIT_ASCITES = df.INIT_ASCITES.replace({1: 0, 2: 1, 3: 1, 4: np.nan})
df.DIAB = df.DIAB.replace({1: 0, 2: 1, 3: 1, 4: 1, 5: np.nan, 998: np.nan})
df['COD_CAD_DON_1'] = df.COD_CAD_DON.replace({1: 1, 2: 0, 3: 0, 4: 0, 999: np.nan})
df['COD_CAD_DON_2'] = df.COD_CAD_DON.replace({1: 0, 2: 1, 3: 0, 4: 0, 999: np.nan})
df['COD_CAD_DON_3'] = df.COD_CAD_DON.replace({1: 0, 2: 0, 3: 1, 4: 0, 999: np.nan})
df['COD_CAD_DON_4'] = df.COD_CAD_DON.replace({1: 0, 2: 0, 3: 0, 4: 1, 999: np.nan})
df.DIABETES_DON = df.DIABETES_DON.replace({'Y': 1, 'N': 0, 'U': np.nan})
df.ABO = df.ABO.replace({'UNK': np.nan})
df.ABO_DON = df.ABO_DON.replace({'UNK': np.nan})

df['INIT_DATE_YEAR'] = pd.DatetimeIndex(df.INIT_DATE).year
df['INIT_BILIRUBIN_X_SERUM_SODIUM'] = df.INIT_BILIRUBIN * df.INIT_SERUM_SODIUM

df['M1_DURATION_RAW'] = (df.DEATH_DATE - df.INIT_DATE) / np.timedelta64(1, 'D')
df['M1_DURATION'] = np.minimum(5*12*30, df.M1_DURATION_RAW)
df['M1_EVENT'] = df.M1_DURATION_RAW < 5*12*30
df.M1_EVENT = df.M1_EVENT.replace({True: 1, False: 0})

df['TX_WAITING_TIME'] = (df.TX_DATE - df.INIT_DATE) / np.timedelta64(1, 'D')
df['ABO_COMPATIBILITY'] = df.ABO == df.ABO_DON
df.ABO_COMPATIBILITY = df.ABO_COMPATIBILITY.replace({True: 1, False: 0})
df['HCV_NAT_X_DIABETES_DON'] = df.HCV_NAT * df.DIABETES_DON
df['HCV_NAT_X_AGE_DON'] = df.HCV_NAT * df.AGE_DON
df['AGE_X_CREAT_TX'] = df.AGE * df.CREAT_TX

df['M2_DURATION'] = np.minimum(5*12*30, df.PTIME)
df['M2_EVENT'] = df.PTIME < 5*12*30
df.M2_EVENT = df.M2_EVENT.replace({True: 1, False: 0})

# m1_vars = ['INIT_AGE', 'GENDER', 'INIT_SERUM_CREAT', 'INIT_BILIRUBIN', 'INIT_INR', 'INIT_SERUM_SODIUM', 'INIT_DATE_YEAR', 'INIT_BILIRUBIN_X_SERUM_SODIUM']
m1_vars = ['INIT_AGE', 'GENDER', 'INIT_SERUM_CREAT', 'INIT_BILIRUBIN', 'INIT_INR', 'INIT_SERUM_SODIUM', 'INIT_BILIRUBIN_X_SERUM_SODIUM']

m1_df = df[m1_vars + ['M1_DURATION', 'M1_EVENT']]
for var in m1_vars + ['M1_DURATION', 'M1_EVENT']:
    m1_df = m1_df[~m1_df[var].isna()]

m1_cph = CoxPHFitter()
m1_cph.fit(m1_df, 'M1_DURATION', 'M1_EVENT')

m2_vars = ['AGE', 'GENDER', 'HCV_NAT', 'CREAT_TX', 'TBILI_TX', 'INR_TX', 'INIT_SERUM_SODIUM', 'ALBUMIN_TX', 'PREV_AB_SURG_TRR', 'ENCEPH_TX', 'ASCITES_TX', 'TX_WAITING_TIME', 'DIAB', 'AGE_DON', 'COD_CAD_DON_1', 'COD_CAD_DON_2', 'COD_CAD_DON_3', 'BMI_DON_CALC', 'DIABETES_DON', 'ABO_COMPATIBILITY', 'HCV_NAT_X_DIABETES_DON', 'HCV_NAT_X_AGE_DON', 'AGE_X_CREAT_TX']

m2_df = df[m2_vars + ['M2_DURATION', 'M2_EVENT']]
for var in m2_vars + ['M2_DURATION', 'M2_EVENT']:
    m2_df = m2_df[~m2_df[var].isna()]

m2_cph = CoxPHFitter()
m2_cph.fit(m2_df, 'M2_DURATION', 'M2_EVENT')

# print(m1_df.index.size)
# print(m2_df.index.size)

df.CREAT_TX = df.CREAT_TX.fillna(df.INIT_SERUM_CREAT)
df.INIT_SERUM_CREAT = df.CREAT_TX
df.TBILI_TX = df.TBILI_TX.fillna(df.INIT_BILIRUBIN)
df.INIT_BILIRUBIN = df.TBILI_TX
df.INIT_BILIRUBIN_X_SERUM_SODIUM = df.INIT_BILIRUBIN * df.INIT_SERUM_SODIUM
df.INR_TX = df.INR_TX.fillna(df.INIT_INR)
df.INIT_INR = df.INR_TX
df.ALBUMIN_TX = df.ALBUMIN_TX.fillna(df.INIT_ALBUMIN)
df.PREV_AB_SURG_TRR = df.PREV_AB_SURG_TRR.fillna(df.PREV_AB_SURG_TCR)
df.ENCEPH_TX = df.ENCEPH_TX.fillna(df.INIT_ENCEPH)
df.ASCITES_TX = df.ASCITES_TX.fillna(df.INIT_ASCITES)

for var in ['AGE', 'GENDER', 'HCV_NAT', 'CREAT_TX', 'TBILI_TX', 'INR_TX', 'INIT_SERUM_SODIUM', 'ALBUMIN_TX', 'PREV_AB_SURG_TRR', 'ENCEPH_TX', 'ASCITES_TX', 'TX_WAITING_TIME', 'DIAB', 'ABO']:
    df = df[~df[var].isna()]

df_tx = df.copy()
for var in ['AGE_DON', 'COD_CAD_DON_1', 'COD_CAD_DON_2', 'COD_CAD_DON_3', 'BMI_DON_CALC', 'DIABETES_DON', 'ABO_DON']:
    df_tx = df_tx[~df_tx[var].isna()]

# print(df.index.size)
# print(df_tx.index.size)

def f(ind, row):
    df1 = df[(df.INIT_DATE <= row.TX_DATE) & (row.TX_DATE <= df.END_DATE)].copy()

    df1.INIT_AGE = df1.INIT_AGE + (row.TX_DATE - df1.INIT_DATE) / np.timedelta64(1, 'Y')
    df1.AGE = df1.INIT_AGE
    for var in ['AGE_DON', 'COD_CAD_DON_1', 'COD_CAD_DON_2', 'COD_CAD_DON_3', 'BMI_DON_CALC', 'DIABETES_DON']:
        df1[var] = row[var]
    df1.TX_WAITING_TIME = (row.TX_DATE - df1.INIT_DATE) / np.timedelta64(1, 'D')
    df1.ABO_COMPATIBILITY = df1.ABO == df1.ABO_DON
    df1.ABO_COMPATIBILITY = df.ABO_COMPATIBILITY.replace({True: 1, False: 0})
    df1.HCV_NAT_X_DIABETES_DON = df1.HCV_NAT * df1.DIABETES_DON
    df1.HCV_NAT_X_AGE_DON = df1.HCV_NAT * df1.DIABETES_DON
    df1.AGE_X_CREAT_TX = df1.AGE * df1.CREAT_TX

    df1['M1'] = m1_cph.predict_expectation(df1)
    df1['M2'] = m2_cph.predict_expectation(df1)
    df1['BENEFIT'] = df1.M2-df1.M1
    df1['NEED'] = -df1.M1
    df1 = df1[['BENEFIT', 'NEED']]
    diff = df1.loc[ind] - df1[df1.index != ind]

    return diff.values

k = 0
data = list()
for ind, row in df_tx.iterrows():
    # if k % 10 == 0:
    #     print(k)
    k += 1
    data.append(f(ind, row))

random.seed(0)
random.shuffle(data)

with open('data/liver.obj', 'wb') as f:
    dill.dump(data, f)
