import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
df = pd.read_csv('dating.csv')

grouped = df.groupby(df.gender)
df_female = grouped.get_group(0)
df_male = grouped.get_group(1)


attract_fem = mean(df_female['attractive_important'])
sincere_fem = mean(df_female['sincere_important'])
intelligent_fem = mean(df_female['intelligence_important'])
funny_fem = mean(df_female['funny_important'])
ambition_fem = mean(df_female['ambition_important'])
shared_int_fem = mean(df_female['shared_interests_important'])

attract_mal = mean(df_male['attractive_important'])
sincere_mal = mean(df_male['sincere_important'])
intelligent_mal = mean(df_male['intelligence_important'])
funny_mal = mean(df_male['funny_important'])
ambition_mal = mean(df_male['ambition_important'])
shared_int_mal = mean(df_male['shared_interests_important'])

data = [['attractive',attract_fem,attract_mal], ['sincere',sincere_fem,sincere_mal], ['intelligence',intelligent_fem,intelligent_mal],
        ['funny',funny_fem,funny_mal],
        ['ambition',ambition_fem,ambition_mal], ['shared_interests',shared_int_fem,shared_int_mal]]
barData = pd.DataFrame(data, columns=["Attribute", "Female", "Male"])

barData.plot(x="Attribute", y=["Female", "Male"], kind="bar", figsize=(5,3))
plt.xticks(rotation = 0)
plt.savefig('graph1.png')
plt.show()
