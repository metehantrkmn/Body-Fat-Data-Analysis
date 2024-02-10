import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px

#data set'inin okundu kod bloğu
data = pd.read_csv ('./bodyfat.csv')
df = pd.DataFrame (data)
df

#görselleştirme ve ilişki incelemek için vücut kitle endeksinin hesaplanıp veriye eklendiği blok
Height_meter = (df['Height'] * 0.0254).round(2)

Weight_Kg = (df['Weight'] * 0.454).round(2)

BMI_score = (Weight_Kg/Height_meter**2).round()
BMI = pd.DataFrame(BMI_score)
BMI
df.insert(0,'BMI',BMI)
df

#line plot => ortalama vücut kitle endeksi ve yaş ilişkisi
avg_BMI_age = df.groupby('Age')['BMI'].mean().reset_index()
plot = px.line(avg_BMI_age, x='Age', y='BMI',title = 'Average BMI by Age')
plot.show()

#yaş ve vücut yağ oranı dağılımı scatter plot ile
plt.scatter(df['Age'], df['BodyFat'], c='blue', alpha=0.5)
plt.title('Age vs. Body Fat')
plt.xlabel('Age')
plt.ylabel('Body Fat')
plt.show()

#ağırlık ve vücut yağ oranı dağılımı scatter plot ile
plt.scatter(df['Weight'], df['BodyFat'], c='blue', alpha=0.5)
plt.title('Weight vs. Body Fat')
plt.xlabel('Weight')
plt.ylabel('Body Fat')
plt.show()

#yaş ve ağırlık karşılaştırma bar plot ile 
plt.bar(df['Age'], df['Weight'], color='green', alpha=0.7)
plt.title('Age vs. Weight')
plt.xlabel('Age')
plt.ylabel('Weight')
plt.show()

#vücut yağ dağılımı frekans dağılımı histogram ile
plt.hist(df['BodyFat'], bins=20, color='orange', edgecolor='black')
plt.title('Body Fat Dağılımı')
plt.xlabel('Body Fat Percentage')
plt.ylabel('Frequency')
plt.show()

#verideki boy frekans dağılımı histogram ile
plt.hist(df['Height'], bins=20, color='blue', edgecolor='black')
plt.title('Boy Dağılımı')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.show()

#ağırlık frekans dağılımı histogram ile ---
plt.hist(df['Weight'], bins=20, color='yellow', edgecolor='black')
plt.title('Ağırlık Dağılımı')
plt.xlabel('Ağırlık')
plt.ylabel('Frequency')
plt.show()

#veri seti yaş dağılımı histogram ile
plt.hist(df['Age'], bins=15, color='skyblue', edgecolor='black')
plt.title('Yaş Dağılımı')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

#ağırlık boy dağılımı scatter ile
plt.scatter(df['Weight'], df['Height'], c='coral', alpha=0.7)
plt.title('Weight vs. Height')
plt.xlabel('Weight')
plt.ylabel('Height')
plt.show()

#ağırlık, boy, vücut yağ oranı, yaş toplu karşılaştırma
sns.pairplot(df[['Weight', 'Height', 'BodyFat', 'Age']])
plt.suptitle('Weight, Height, BodyFat, ve Age', y=1.02)
plt.show()

#tüm değişkenlerin dağılımı box plot ile
plt.figure(figsize=(20,15))
sns.boxplot (data= df,notch=True,showcaps=True,
             flierprops={'marker':'x'},
             boxprops={'facecolor':(.3,.4,.5,.3)},
             medianprops={"color": "r", "linewidth": 2},whis=(0,80))
plt.show()

#ortalama vücut kitle endeksinin yaşa göre dağılımı line plot
avg_BMI_age = df.groupby('Age')['BMI'].mean().reset_index()
plot = px.line(avg_BMI_age, x='Age', y='BMI',title = 'Ortalama Vücut Kütle Endeksinin Yaşa Göre Dağılımı')
plot.show()

#ortalama vücut kitle endeksi vücut yağına göre dağılımı
avg_BodyFat_age = df.groupby('BodyFat')['BMI'].mean().reset_index()
plot = px.line(avg_BodyFat_age, x='BodyFat', y='BMI',title = 'Ortalama Vücüt Kitle Endeksi Vucut Yağına Göre Dağılımı')
plot.show()

#data içerisindeki değişkenlerin tanımlayıcı istatistik değerlerini veren komut
descriptive_stats = df.describe().round(2)
print(descriptive_stats)

#değişkenlerin varyanslarını veren komut
all_variances = df.var()
print("Overall Variance:")
print(all_variances)


# yaşı 30'dan küçük bireyler
df_younger_than_30 = df[df['Age'] < 30]

# yaşı 30'dan küçük olanların vücut yağı ortalaması ve varyansını veren kod bloğu
var_younger_than_30 = df_younger_than_30['BodyFat'].var()
mean_body_fat_younger_than_30 = df_younger_than_30['BodyFat'].mean()
count_younger_than_30 = df_younger_than_30.shape[0]

print("Mean Body Fat for People Younger Than 30: {:.2f}".format(mean_body_fat_younger_than_30))
print("Number of People Younger Than 30: {}".format(count_younger_than_30))
print("variance of Body Fat for People Younger Than 30: {:.2f}".format(var_younger_than_30))

# yaşı 30'dan büyük olanlar
df_older_than_30 = df[df['Age'] > 30]

# yaşı 30'dan büyük olan bireylerin vücut yağı ortalaması ve varyansı
var_older_than_30 = df_older_than_30['BodyFat'].var()
mean_body_fat_older_than_30 = df_older_than_30['BodyFat'].mean()
count_older_than_30 = df_older_than_30.shape[0]

print("Mean Body Fat for People older Than 30: {:.2f}".format(mean_body_fat_younger_than_30))
print("Number of People older Than 30: {}".format(count_younger_than_30))
print("variance of Body Fat for People older Than 30: {:.2f}".format(var_older_than_30))

#korelasyon analizi için kovaryans değerini bulan kod bloğu
variable1 = data['Weight']
variable2 = data['Height']

covariance = variable1.cov(variable2)

print(f"The covariance between Variable1 and Variable2 is: {covariance}")