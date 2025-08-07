

# -------------------------
# Land Cover (LC) Mapping
# -------------------------

lc0_class_mapping = {
    'A': 'Artificial land',
    'B': 'Cropland',
    'C': 'Woodland',
    'D': 'Shrubland',
    'E': 'Grassland',
    'F': 'Bareland',
    'G': 'Water',
    'H': 'Wetlands'
}

lc1_class_mapping = {
    # ARTIFICIAL
    'A11': 'Buildings with one to three floors',
    'A12': 'Buildings with more than three floors',
    'A13': 'Greenhouses',
    'A21': 'Non built-up area features',
    'A22': 'Non built-up linear features',
    'A30': 'Other artificial areas',

    # CROPLAND
    'B11': 'Common wheat',
    'B12': 'Durum wheat',
    'B13': 'Barley',
    'B14': 'Rye',
    'B15': 'Oats',
    'B16': 'Maize',
    'B17': 'Rice',
    'B18': 'Triticale',
    'B19': 'Other cereals',
    'B21': 'Potatoes',
    'B22': 'Sugar beet',
    'B23': 'Other root crops',
    'B31': 'Sunflower',
    'B32': 'Rape and turnip rape',
    'B33': 'Soya',
    'B34': 'Cotton',
    'B35': 'Other fibre and oleaginous crops',
    'B36': 'Tobacco',
    'B37': 'Other non-permanent industrial crops',
    'B41': 'Dry pulses',
    'B42': 'Tomatoes',
    'B43': 'Other fresh vegetables',
    'B44': 'Floriculture and ornamental plants',
    'B45': 'Strawberries',
    'B51': 'Clovers',
    'B52': 'Lucerne',
    'B53': 'Other leguminous and mixtures for fodder',
    'B54': 'Mix of cereals',
    'B55': 'Temporary grassland',
    'B71': 'Apple fruit',
    'B72': 'Pear fruit',
    'B73': 'Cherry fruit',
    'B74': 'Nuts trees',
    'B75': 'Other fruit trees and berries',
    'B76': 'Oranges',
    'B77': 'Other citrus fruit',
    'B81': 'Olive groves',
    'B82': 'Vineyards',
    'B83': 'Nurseries',
    'B84': 'Permanent industrial crops',
    'BX1': 'Arable land (only PI)',
    'BX2': 'Permanent crops (only PI)',

    # WOODLAND
    'C10': 'Broadleaved woodland',
    'C21': 'Spruce dominated coniferous woodland',
    'C22': 'Pine dominated coniferous woodland',
    'C23': 'Other coniferous woodland',
    'C31': 'Spruce dominated mixed woodland',
    'C32': 'Pine dominated mixed woodland',
    'C33': 'Other mixed woodland',

    # SHRUBLAND
    'D10': 'Shrubland with sparse tree cover',
    'D20': 'Shrubland without tree cover',

    # GRASSLAND
    'E10': 'Grassland with sparse tree/shrub cover',
    'E20': 'Grassland without tree/shrub cover',
    'E30': 'Spontaneously re-vegetated surfaces',

    # BARELAND
    'F10': 'Rocks and stones',
    'F20': 'Sand',
    'F30': 'Lichens and moss',
    'F40': 'Other bare soil',

    # WATER
    'G11': 'Inland fresh water bodies',
    'G12': 'Inland salty water bodies',
    'G21': 'Inland fresh running water',
    'G22': 'Inland salty running water',
    'G30': 'Transitional water bodies',
    'G50': 'Glaciers, permanent snow',

    # WETLANDS
    'H11': 'Inland marshes',
    'H12': 'Peatbogs',
    'H21': 'Salt marshes',
    'H22': 'Salines and other chemical deposits',
    'H23': 'Intertidal flats'
}

# -------------------------
# Land Use (LU) Mapping
# -------------------------

lu0_class_mapping = {
    'U11': 'Agriculture',
    'U12': 'Forestry',
    'U13': 'Aquaculture and Fishing',
    'U14': 'Mining and quarrying',
    'U15': 'Other primary production',
    'U21': 'Energy production',
    'U22': 'Industry and Manufacturing',
    'U31': 'Transport, Communication, Storage, Protection',
    'U32': 'Water and Waste Treatment',
    'U33': 'Construction',
    'U34': 'Commerce, Financial, Professional and Information Services',
    'U35': 'Community services',
    'U36': 'Recreation, Leisure, Sport',
    'U37': 'Residential',
    'U41': 'Abandoned areas',
    'U42': 'Semi-natural and natural areas not in use'
}

lu1_class_mapping = {
    'U111': 'Agriculture (excluding fallow land and kitchen gardens)',
    'U112': 'Fallow Land',
    'U113': 'Kitchen Garden',
    'U120': 'Forestry',
    'U130': 'Aquaculture and Fishing',
    'U140': 'Mining and quarrying',
    'U150': 'Other primary production',
    'U210': 'Energy production',
    'U221': 'Manufacturing of food, beverages and tobacco products',
    'U222': 'Manufacturing of textile products',
    'U223': 'Coal, oil and metal processing',
    'U224': 'Production of non-metal mineral goods',
    'U225': 'Chemical and allied industries',
    'U226': 'Machinery and equipment',
    'U227': 'Wood based products',
    'U228': 'Printing and reproduction',
    'U311': 'Railway transport',
    'U312': 'Road transport',
    'U313': 'Water transport',
    'U314': 'Air transport',
    'U315': 'Transport via pipelines',
    'U316': 'Telecommunication',
    'U317': 'Logistics and storage',
    'U318': 'Protection infrastructures',
    'U319': 'Electricity, gas and thermal power distribution',
    'U321': 'Water supply and treatment',
    'U322': 'Waste treatment',
    'U330': 'Construction',
    'U341': 'Commerce',
    'U342': 'Financial, professional and information services',
    'U350': 'Community services',
    'U361': 'Amenities, museums, leisure',
    'U362': 'Sport',
    'U370': 'Residential',
    'U411': 'Abandoned industrial areas',
    'U412': 'Abandoned commercial areas',
    'U413': 'Abandoned transport areas',
    'U414': 'Abandoned residential areas',
    'U415': 'Other abandoned areas',
    'U420': 'Semi-natural and natural areas not in use'
}
# -------------------------
# Crops Mapping
# -------------------------

crop_labels ={
            'B11': 'Common wheat','B12': 'Durum wheat','B13': 'Barley','B14': 'Rye','B15': 'Oats', 'B16': 'Maize','B17': 'Rice','B18': 'Triticale','B19': 'Other cereals','B21': 'Potatoes',
            'B22': 'Sugar beet','B23': 'Other root crops',
            'B31': 'Sunflower','B32': 'Rape and turnip rape','B33': 'Soya', 'B34': 'Cotton','B35': 'Other fibre and oleaginous crops','B36': 'Tobacco', 'B37': 'Other non-permanent industrial crops', 
            'B41': 'Dry pulses', 'B42': 'Tomatoes' ,'B43': 'Other fresh vegetables' ,'B44': 'Floriculture and ornamental plants' ,'B45': 'Strawberries' ,
            'B51': 'Clovers' ,'B52': 'Lucerne' ,'B53': 'Other leguminous and mixtures for fodder' ,'B54': 'Mix of cereals' ,'B55': 'Temporary grassland' ,
            'B71': 'Apple fruit','B72': 'Pear fruit','B73': 'Cherry fruit' ,'B74': 'Nuts trees' ,'B75': 'Other fruit trees and berries' ,'B76': 'Oranges' ,'B77': 'Other citrus fruit' ,
            'B81': 'Olive groves' ,'B82': 'Vineyards' ,'B83': 'Nurseries' ,'B84': 'Permanent industrial crops'
}


bioregion_mapping = {
    0: 'Alpine', 1: 'Anatolian', 2: 'Arctic', 3: 'Atlantic', 4: 'BlackSea', 5: 'Boreal', 6: 'Continental',
    7: 'Macaronesia', 8: 'Mediterranean', 9: 'Outside', 10: 'Pannonian', 11: 'Steppic'
}


eunis_mapping = {0: 'Arable land and market gardens', 1: 'Low density buildings', 2: 'Mesic grasslands', 
                3: 'Broadleaved deciduous woodland', 
                4: 'Coniferous woodland', 5: 'Temperate and mediterranean-montane scrub', 
                6: 'Seasonally wet and wet grasslands', 7: 'Surface running waters', 
                8: 'Buildings of cities, towns and villages', 9: 'Mixed deciduous and coniferous woodland', 10: 'Dry grasslands', 
                11: 'Arctic, alpine and subalpine scrub', 12: 'Alpine and subalpine grasslands', 
                13: 'Broadleaved evergreen woodland', 
                14: 'Lines of trees, small anthropogenic woodlands, recently felled woodland, early-stage woodland and coppice', 
                15: 'Shrub plantations', 16: 'Maquis, arborescent matorral and thermo-Mediterranean brushes', 17: 'Temperate shrub heathland', 
                18: 'Raised and blanket bogs', 19: 'Transport networks and other constructed hard-surfaced areas', 20: 'Extractive industrial sites', 
                21: 'Spiny Mediterranean heaths (phrygana, hedgehog-heaths and related coastal cliff vegetation)', 22: 'Surface standing waters', 23: 'Tundra', 
                24: 'Inland cliffs, rock pavements and outcrops', 25: 'Garrigue', 26: 'Aapa, palsa and polygon mires', 27: 'Cultivated areas of gardens and parks', 
                28: 'Waste deposits', 29: 'Miscellaneous inland habitats with very sparse or no vegetation', 30: 'Inland salt steppes', 
                31: 'Sparsely wooded grasslands', 32: 'Sedge and reedbeds, normally without free-standing water', 33: 'Valley mires, poor fens and transition mires', 
                34: 'Riverine and fen scrubs', 35: 'Screes', 36: 'Inland saline and brackish marshes and reedbeds', 37: 'Coastal dunes and sandy shores', 
                38: 'Littoral zone of inland surface waterbodies', 39: 'Highly artificial man-made waters and associated structures', 
                40: 'Base-rich fens and calcareous spring mires', 41: 'Snow or ice-dominated habitats', 
                42: 'Rock cliffs, ledges and shores, including the supralittoral', 43: 'Coastal shingle'}

#### Country NUTS0 Labels ####
# NUTS0 labels for European countries
# These labels are used to identify the country of origin for each image in the dataset.
nuts_labels = {
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "CY": "Cyprus",
    "CZ": "Czech Republic",
    "DE": "Germany",
    "DK": "Denmark",
    "EE": "Estonia",
    "EL": "Greece",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "HR": "Croatia", 
    "HU": "Hungary",
    "IE": "Ireland",
    "IT": "Italy",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "LV": "Latvia",
    "MT": "Malta",
    "NL": "Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "SE": "Sweden",
    "SI": "Slovenia",
    "SK": "Slovakia",
    "UK": "United Kingdom",
   
}

LU0_class_list = [value for id, value in lu0_class_mapping.items()]
LC0_class_list = [value for id, value in lc0_class_mapping.items()]
LU1_class_list = [value for id, value in lu1_class_mapping.items()]
LC1_class_list = [value for id, value in lc1_class_mapping.items()]

LU0_class_map = {id: i for i, id in enumerate(lu0_class_mapping.keys())}
LU1_class_map = {id: i for i, id in enumerate(lu1_class_mapping.keys())}
LC0_class_map = {id: i for i, id in enumerate(lc0_class_mapping.keys())}
LC1_class_map = {id: i for i, id in enumerate(lc1_class_mapping.keys())}

crop_class_map = {id: i for i, id in enumerate(crop_labels.keys())}
crop_class_list = [value for id, value in crop_labels.items()]
bio_class_list = [value for id, value in bioregion_mapping.items()]
eunis_class_list = [value for id, value in eunis_mapping.items()]
class_list_nuts = list(dict.fromkeys([nuts_labels[value] for value in nuts_labels.keys()]))
# print(len(LC1_class_list))

