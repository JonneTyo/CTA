import pandas as pd
from data_handler import csv_read
import datetime
END_OF_FOLLOW_UP_DATE = datetime.date(2019, 12, 31)

DATA_DIR = 'Original data\\'
CURRENT_DATA_FILE =  DATA_DIR + 'CTArekisteri_DATA_LABELS_2021-02-17_1146.csv'
csv_headers_original = csv_read(CURRENT_DATA_FILE).columns
field_types = ['perustiedot', 'cta', 'perfuusio', 'seuranta', 'laakitys', 'labrat_ennen', 'labrat_jalkeen', 'ttgene',
               'tutkimukset']

PECTUS_IDS = csv_read(DATA_DIR + 'PECTUS.csv').index

# features to remove
'''ftr = ['Turku ID',  "Previous ID (if there is a former CTA)", "Data collector's name", "Complete?",
       "CTA failed/nondiagnostic (choice=fail)", "PET study failed/nondiagnostic (choice=fail)",
       'CARDIOVASCULAR DEATH (choice=cardiovascular death)', 'Personal identification number ', 'Patient name',
       'NON-CARDIOVASCULAR DEATH (choice=non-cardiovascular death)', 'EXITUS, end of follow-up (date)',
       'EXITUS - date checked (choice=confirmed)', 'EXITUS - comment', 'EXITUS date', 'main cause of death',
       'VALIVAIHEEN_KUOLEMANSYY', 'VALITON_KUOLEMANSYY', 'MYOTAVAIKUTTANEET_TILAT', 'KUOLEMANSYYN_LAHDE',
       'MI2 - Code (dg)', 'MI2 -Event date', 'MI2 - LähtöPVM', 'MI2 - confirmed', 'MI3 - comment', 'MI3 - code (dg)',
       'MI3 - Event date', 'MI3 - LähtöPVM', 'MI3 - confirmed', 'UAP3 - Event date', 'UAP3 - LähtöPVM',
       'UAP3 - confirmed', 'MI1 - Comment', 'MI1 - Code (dg)', 'MI1 - Event date', 'MI1 - LähtöPVM',
       'MI1 - Confirmed', 'MI2 - Comment', 'UAP1 - lähtöpvm', 'UAP1 - confirmed', 'UAP2 - comment',
       'UAP2 - code (dg)', 'UAP2 - Event date', 'UAP2 - lähtöpvm', 'UAP2 - confirmed', 'UAP3 - comment',
       'UAP3 - code (dg)', 'UAP1 - comment', 'UAP1 - code (dg)', 'UAP1 - Event date', 'Complete?',
       ' (choice=cin (Teemu))', ' (choice=prognosis (Teemu))', ' (choice=cmd (IidaS))', ' (choice=adherence (IidaS))',
       ' (choice=3vd (Teemu))', ' (choice=global (Esa))', ' (choice=gender (Wail))', ' (choice=gene (IidaK))',
       'LM_ART (choice=artefact, cannot be analysed)', 'Study indication', 'Following PET perfusion imaging performed',
       'Sending unit (Kts. Lähete)', 'SELOKEN', 'DINIT', 'DINIT ', 'DINIT  ', 'CTA date', 'Complete data on baseline medication',
       'CTA image quality', 'Betablocker', 'Lipid-lowering drug', 'Anti-platelet drug (ASA or other)',
       'Anticoagulant', 'Long-acting nitrate', 'Diuretic', 'ACE-inhibitor', 'ATR-blocker',
       'Calcium channel blocker', 'Antiarrhythmic drug', 'CTA_LOYD', 'Stress ARVIO_GLOBAL']'''


# create a matrix where each column describes an event, such as a requirement for a variable or how to handle missing
# variables and rows describe the variables
# index 'transform' must be of form list > list > tuple/list and val
# missing data doesn't need to be transformed to 0 as that is done later. If missing data should have value
# other than 0 then do it here.
data_handling_functions = ['req', 'between', 'missing', 'transform']
data_handling = {

    'diabetes': [None, None, 4, [[(1,2), 1], [None, 0]]],
    'Study indication': [[1,2,4,5], None, None, None],
    'Chestpain': [None, None, 3, [[[1], 1], [None, 0]]],
    'MI1 - Confirmed': [None, None, 0, [[None, 0]]],
    'Smoking': [None, None, 0, [[(1,3), 1], [None, 0]]],
    'hypertension': [None, None, 0, [[[1], 1], [None, 0]]],
    'dyslipidemia': [None, None, 0, [[[1], 1], [None, 0]]]
}

feature_handling_frame = pd.DataFrame(data=data_handling, index=data_handling_functions)

CUSTOM_HANDLING = {
    'index': {
        'require': {'min_val': 862}
    },
    'diabetes': {
        'missing': {'fill_val': 4},
        'transform': {'combine': [(1, 2)],
                      'missing': 0},
    },
    'Study indication': {
        'require': {'one_of': [1, 2, 4, 5, 11]}
    },
    'Chestpain': {
        'missing': {'fill_val': 3}
    },
    'Smoking': {
        'transform': {'combine': [(1, 3)],
                      'missing': 0}
    }
}

START_DATES = ['CTA date']
END_DATES = ['EXITUS, end of follow-up (date)', 'EXITUS date', 'MI1 - Event date']

