import pandas as pd


data = ['cancer','paralysis','heart dieases']
table = pd.Series(data)


animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals)

s = pd.Series([100.00, 120.00, 101.00, 3.00])
#print(s.sum())
##print("sum :", s.sum())

#sports = {'Archery': 'Bhutan','Golf': 'Scotland','Sumo': 'Japan','Taekwondo': 'South Korea'}
#s = pd.Series(sports)
#print(s.loc['Taekwondo'])

import pandas as pd
purchase_1 = pd.Series({'Name': 'Chris','Item Purchased': 'Dog Food','Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn','Item Purchased': 'Kitty Litter','Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod','Item Purchased': 'Bird Seed','Cost': 5.00}) 
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 2', 'Store 3']) 
print(df.head())
print(" ")
print(df['Name'])