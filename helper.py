import numpy as np
import pandas as pd


def normalize(arr, t_min=0, t_max=1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return pd.Series(np.array(norm_arr))


def normalize_dataframe(df):
    for col in df.columns:
        df[col] = normalize(df[col])
    return df


def define_membership_function(df, labels, num_clusters=3, memfunc='gaussmf'):
    mf = []
    for col in df.columns:
        if col not in labels:
            mf.append([])
            for cluster in range(num_clusters):
                df_cluster = df[col].sort_values().iloc[cluster:len(df[col]) * (cluster + 1)]
                dicts = ['mean', 'sigma']
                values = [np.mean(df_cluster), np.std(df_cluster)]
                mf[-1].append([memfunc, dict(zip(dicts, values))])
    return np.array(mf)
