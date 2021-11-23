import pandas as pd
from itertools import product
from sklearn.utils.extmath import cartesian
import numpy as np
from CTA_class import CTA_class
import os
import codecs
import matplotlib.pyplot as plt

#data_file_path = "E:\Other\Repos\CTA\Main\CTA-Projekti\Data\Original Data\\CTArekisteri_DATA_2021-05-17_1441.csv"

'''
with codecs.open(data_file_path, 'r+', encoding='utf-8', errors='ignore') as fdata:
    s = fdata.read()
    with open('temp_data.csv', 'w', encoding='utf-8') as f:
        f.write(s)

'''

df = pd.read_csv('temp_data.csv', encoding='utf-8', index_col=0)

df = df.loc[df['tutkimus9'] == 1]
events = df['mi_uap_death_fu_status_2020']
late_revascs = df.loc[:, ['latepci_date', 'latecabg_date']]
late_revascs = late_revascs.notna().astype(int)
late_revascs['either'] = late_revascs.max(axis=1)
late_revascs['event'] = events
print(late_revascs.loc[late_revascs['event'] >= 1].mean(axis=0))
print(late_revascs.loc[late_revascs['event'] <= 0].mean(axis=0))
print(late_revascs.loc[late_revascs['either'] >= 1].mean())
print(late_revascs.loc[late_revascs['either'] <= 0].mean())
'''
df = df.iloc[:, [n for n in range(252, 259)] + [n for n in range(338, 366)]]
df['event'] = events

df = df.loc[:, ['event', 'early_pci', 'early_cabg']]
df = df.astype('float64')
df['any early revasc'] = df.loc[:, ['early_pci', 'early_cabg']].sum(axis=1) >= 1
event_proportion = df['event'].sum()/df.shape[0]
revasc_proportion = df['any early revasc'].sum()/df.shape[0]
event_proportion_with_revasc = df.loc[df['any early revasc'] >= 1]['event'].sum()/df.loc[df['any early revasc'] >= 1].shape[0]
revasc_proportion_with_event = df.loc[df['event'] >= 1]['any early revasc'].sum()/df.loc[df['event'] >= 1].shape[0]

fig, axes = plt.subplots()
vals, _, _ = axes.hist([df['early_ica'], df['early_pci'], df['early_cabg'], df['any early revasc']], bins=[0, 0.5, 1.0], label=['Early ICA', 'Early PCI', 'Early CABG', 'Any'])
axes.legend()
axes.set_xticks([0.25, 0.75])
axes.set_xticklabels(['No revascularization', 'Revascularization done'])
axes.grid(True)
axes.set_title(f'Total proportion of revascularizations: {revasc_proportion*100:.1f}%')
for i, val in enumerate(vals):
    axes.text(0.55 + i/10, val[-1], f'{val[-1]/np.sum(val)*100:.1f}%')
    axes.text(0.05 + i/10, val[0], f'{val[0]/np.sum(val)*100:.1f}%')
plt.savefig('revasc_hist.png', dpi=600)

#todo 2x2 gridi, x-akselilla eventit y-akselilla revascit
'''


