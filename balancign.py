import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.utils import resample


df = pd.read_csv('hasil/2018_datatesting.csv', quoting=csv.QUOTE_NONE)
# print(df)
# plt.figure(figsize=(8, 8))
# sns.countplot('Kelas', data=df)
# plt.title('Balanced Classes')
# plt.show()
print(df.Kelas.value_counts())

# # Shuffle the Dataset.
# shuffled_df = df.sample(frac=1,random_state=4)
# #
# # # Put all the fraud class in a separate dataset.
# fraud_df = shuffled_df.loc[shuffled_df['Kelas'] == 1]
# #
# # #Randomly select 492 observations from the non-fraud (majority class)
# non_fraud_df = shuffled_df.loc[shuffled_df['Kelas'] == 0].sample(n=492,random_state=42)
# #
# # # Concatenate both dataframes again
# normalized_df = pd.concat([fraud_df, non_fraud_df])
#

df_majority = df[df.Kelas == 1]
df_minority1 = df[df.Kelas == 2]
df_minority2 = df[df.Kelas == 3]
df_minority3 = df[df.Kelas == 4]
df_minority4 = df[df.Kelas == 5]
# resample()
# Downsample majority class
df_minority_upsampled1 = resample(df_minority1,
                                   replace=True,  # sample without replacement
                                   n_samples=2801)  # reproducible results

df_minority_upsampled2 = resample(df_minority2,
                                   replace=True,  # sample without replacement
                                   n_samples=2801)  # reproducible results

df_minority_upsampled3 = resample(df_minority3,
                                   replace=True,  # sample without replacement
                                   n_samples=2801)  # reproducible results

df_minority_upsampled4 = resample(df_minority4,
                                   replace=True,  # sample without replacement
                                   n_samples=2801)  # reproducible results

# Combine minority class with downsampled majority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled1, df_minority_upsampled2, df_minority_upsampled3, df_minority_upsampled4])

# Display new class counts
print(df_upsampled.Kelas.value_counts())
# #plot the dataset after the undersampling
# plt.figure(figsize=(8, 8))
# sns.countplot('Kelas', data=df_downsampled)
# plt.title('Balanced Classes')
# plt.show()

df_upsampled.to_csv('hasil/2018_upbalancing.csv', encoding='utf-8', index=False)
