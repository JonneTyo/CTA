from CTA_class import CTA_class
from CTA_class import PROCESSED_DATA_DIR
import pandas as pd
import numpy as np

test_data = pd.read_csv('Test data filled.csv', index_col=0)
test_labels = pd.read_csv('Test labels.csv', index_col=0)
potilas = test_data.iloc[0:1, :]

from CTA_final import final_models
from CTA_final import float_to_binary_pred

koulutusvuosi, asetukset, mallin_nimi = final_models['all patients using PET data']
cta = CTA_class(**asetukset)
training_data = pd.read_csv('Training data filled.csv', index_col=0) #Alkuperäisessä datassa puuttuvat arvot ovat nanneja
training_labels = pd.read_csv('Training labels.csv', index_col=0)
cta.label = f'event {koulutusvuosi} years'
cta.X_train = training_data
cta.X_test = test_data
cta.y_train = training_labels[cta.label]
cta.y_test = test_labels[cta.label]
cta.train_models()
cta.predict(cta.X_train)
bin_ths = cta.binary_threshold
cta.predict(potilas)

cta.model_predictions = float_to_binary_pred(cta.model_predictions, bin_ths)
print(cta.model_predictions[mallin_nimi])
print(test_labels.iloc[0:1, :])


