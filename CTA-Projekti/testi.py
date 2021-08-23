import pandas as pd
from itertools import product
from sklearn.utils.extmath import cartesian
import numpy as np
from CTA_class import CTA_class
import os


settings_cta = {
    'basic': True,
    'ctapca': True,
    'only_pet': False
}
settings = {
    'basic': True,
    'pca': True,
    'only_pet': False,
    'ctapca': False
}
pet_settings_cta = {
    'basic': True,
    'ctapca': True,
    'only_pet': True
}
pet_settings = {
    'basic': True,
    'pca': True,
    'only_pet': True,
    'ctapca': False
}
settings_list = [settings, settings_cta, pet_settings, pet_settings_cta]
times = [n for n in range(2, 9)]
times.append(None)

cta_data = CTA_class(pca=True)
pca2_vals = cta_data.data['PCA2']
lower_quantile = np.quantile(pca2_vals, 0.25)
upper_quantile = np.quantile(pca2_vals, 0.75)
pca2_vals = pca2_vals.apply(lambda x: 1 if x <= lower_quantile else 2 if x <= upper_quantile else 3)

columns = []
columns.append('true y')
columns.append('passed time')
to_save = pd.DataFrame(index=cta_data.data.index)
for t in times:
    cta_data = CTA_class(time=t)
    to_save[f'true y {t if t is not None else "unrestricted"} years'] = cta_data.labels
to_save['PCA2'] = pca2_vals

to_save['passed time'] = cta_data.original_data['passed time']
to_save.to_csv(os.getcwd() + f'\\kaplan-meier data.csv')
for setting in settings_list:
    temp_cta_class = CTA_class(**setting)
    to_save = pd.DataFrame(index=temp_cta_class.data.index, columns=[f'pred y {n} years' if n is not None else 'pred y unrestricted years' for n in times])
    for t in times:
        cta_data = CTA_class(time=t, **setting)
        cta_data(write_to_csv=False)
        th = cta_data.binary_threshold['lasso']
        cta_data.predict(cta_data.data_fill_na)
        to_save[f'pred y {t if t is not None else "unrestricted"} years'] = (cta_data.model_predictions['lasso'] >= th).astype(int)
    to_save.to_csv(os.getcwd() + f'\\predictions {"only pet" if setting["only_pet"] else "all"} {"cta" if setting["ctapca"] else ""}.csv')

