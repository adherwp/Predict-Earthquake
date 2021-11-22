import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.utils import resample


df = pd.read_csv('hasil/2018_datatesting.csv', quoting=csv.QUOTE_NONE)
# print(df)
plt.figure(figsize=(8, 8))
sns.countplot('Kelas', data=df)
plt.title('Balanced Classes')
# plt.show()
# print(df.Kelas.value_counts())

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

df_majority1 = df[df.Kelas == 1]
df_majority2 = df[df.Kelas == 2]
df_majority3 = df[df.Kelas == 3]
df_majority4 = df[df.Kelas == 4]
df_minority = df[df.Kelas == 5]
resample()
# Downsample majority class
df_majority_downsampled1 = resample(df_majority1,
                                   replace=False,  # sample without replacement
                                   n_samples=150)  # reproducible results

df_majority_downsampled2 = resample(df_majority2,
                                   replace=False,  # sample without replacement
                                   n_samples=150)  # reproducible results

df_majority_downsampled3 = resample(df_majority3,
                                   replace=False,  # sample without replacement
                                   n_samples=150)  # reproducible results

df_majority_downsampled4 = resample(df_majority4,
                                   replace=False,  # sample without replacement
                                   n_samples=150)  # reproducible results

# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_minority, df_majority_downsampled1, df_majority_downsampled2, df_majority_downsampled3, df_majority_downsampled4])

# Display new class counts
# print(df_downsampled)
# #plot the dataset after the undersampling
# plt.figure(figsize=(8, 8))
# sns.countplot('Kelas', data=df_downsampled)
# plt.title('Balanced Classes')
# plt.show()

df_downsampled.to_csv('hasil/2018_balancing1.csv', encoding='utf-8', index=False)
