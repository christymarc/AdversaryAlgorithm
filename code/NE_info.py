# code from https://github.com/JHL-HUST/PWWS/blob/dfd9f6b9a8fb9a57e3a0b9207e1dcbe8a624618c/get_NE_list.py
from collections import defaultdict

NE_type_dict = {
    'PERSON': defaultdict(int),  # People, including fictional.
    'NORP': defaultdict(int),  # Nationalities or religious or political groups.
    'FAC': defaultdict(int),  # Buildings, airports, highways, bridges, etc.
    'ORG': defaultdict(int),  # Companies, agencies, institutions, etc.
    'GPE': defaultdict(int),  # Countries, cities, states.
    'LOC': defaultdict(int),  # Non-GPE locations, mountain ranges, bodies of water.
    'PRODUCT': defaultdict(int),  # Object, vehicles, foods, etc.(Not services)
    'EVENT': defaultdict(int),  # Named hurricanes, battles, wars, sports events, etc.
    'WORK_OF_ART': defaultdict(int),  # Titles of books, songs, etc.
    'LAW': defaultdict(int),  # Named documents made into laws.
    'LANGUAGE': defaultdict(int),  # Any named language.
    'DATE': defaultdict(int),  # Absolute or relative dates or periods.
    'TIME': defaultdict(int),  # Times smaller than a day.
    'PERCENT': defaultdict(int),  # Percentage, including "%".
    'MONEY': defaultdict(int),  # Monetary values, including unit.
    'QUANTITY': defaultdict(int),  # Measurements, as of weight or distance.
    'ORDINAL': defaultdict(int),  # "first", "second", etc.
    'CARDINAL': defaultdict(int),  # Numerals that do not fall under another type.
}

class NameEntityList(object):
    # If the original input in IMDB belongs to class 0 (negative)
    imdb_0 = {'PERSON': 'David',
              'NORP': 'Australian',
              'FAC': 'Hound',
              'ORG': 'Ford',
              'GPE': 'India',
              'LOC': 'Atlantic',
              'PRODUCT': 'Highly',
              'EVENT': 'Depression',
              'WORK_OF_ART': 'Casablanca',
              'LAW': 'Constitution',
              'LANGUAGE': 'Portuguese',
              'DATE': '2001',
              'TIME': 'hours',
              'PERCENT': '98%',
              'MONEY': '4',
              'QUANTITY': '70mm',
              'ORDINAL': '5th',
              'CARDINAL': '7',
              }
    # If the original input in IMDB belongs to class 1 (positive)
    imdb_1 = {'PERSON': 'Lee',
              'NORP': 'Christian',
              'FAC': 'Shannon',
              'ORG': 'BAD',
              'GPE': 'Seagal',
              'LOC': 'Malta',
              'PRODUCT': 'Cat',
              'EVENT': 'Hugo',
              'WORK_OF_ART': 'Jaws',
              'LAW': 'RICO',
              'LANGUAGE': 'Sebastian',
              'DATE': 'Friday',
              'TIME': 'minutes',
              'PERCENT': '75%',
              'MONEY': '$',
              'QUANTITY': '9mm',
              'ORDINAL': 'sixth',
              'CARDINAL': 'zero',
              }
    imdb = [imdb_0, imdb_1]
    
NE_list = NameEntityList().imdb