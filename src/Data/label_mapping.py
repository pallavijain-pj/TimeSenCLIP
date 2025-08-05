
labels = {
    'A10': 0, 'A11': 0, 'A12': 0, 'A13': 0, 
    'A20': 0, 'A21': 0, 'A30': 0, 
    'A22': 0, 
    'B50': 1, 'B51': 1, 'B52': 1,
    'B53': 1, 'B54': 1, 'B55': 1,
    'B10': 1, 'B11': 1, 'B12': 1, 'B13': 1, 'B14': 1, 'B15': 1,
    'B16': 1, 'B17': 1, 'B18': 1, 'B19': 1, 'B10': 1, 'B20': 1, 
    'B21': 1, 'B22': 1, 'B23': 1, 'B30': 1, 'B31': 1, 'B32': 1,
    'B33': 1, 'B34': 1, 'B35': 1, 'B30': 1, 'B36': 1, 'B37': 1,
    'B40': 1, 'B41': 1, 'B42': 1, 'B43': 1, 'B44': 1, 'B45': 1,
    'B70': 1, 'B71': 1, 'B72': 1, 'B73': 1, 'B74': 1, 'B75': 1,
    'B76': 1, 'B77': 1, 'B80': 1, 'B81': 1, 'B82': 1, 'B83': 1,
    'B84': 1, 
    'BX1': 1, 'BX2': 1,
    'C10': 2, 'C20': 2, 'C21': 2, 'C22': 2,
    'C23': 2, 'C30': 2, 'C31': 2, 'C32': 2,
    'C33': 2, 
    'CXX1': 2, 'CXX2': 2, 'CXX3': 2, 'CXX4': 2, 'CXX5': 2,
    'CXX5': 2, 'CXX6': 2, 'CXX7': 2, 'CXX8': 2, 'CXX9': 2,
    'CXXA': 2, 'CXXB': 2, 'CXXC': 2, 'CXXD': 2, 'CXXE': 2,
     'D10': 3, 'D20': 3, 'D10': 3,
     'E10': 4, 'E20': 4, 'E30': 4, 
    'F10': 5, 'F20': 5, 
    'F30': 5, 'F40': 5,  
    'G10': 6, 'G11': 6, 'G12': 6, 'G20': 6, 'G21': 6, 'G22': 6, 'G30': 6, 
    'G40': 6,
    'G50': 6,
    'H10': 7, 'H11': 7, 'H12': 7, 'H11': 7,'H20': 7, 'H21': 7,
    'H22': 7, 'H23': 7 #, '': 10
}
lu_labels = {'U111': 0,'U112':0, 'U113': 0,'U120':1,'U130':2, 'U140':3, 'U150':4,
        'U210':5, 'U221': 6, 'U222': 6, 
        'U223':6, 
        'U224':6, 'U225':6, 'U226':6, 
        'U227':6,
        'U228': 6, 'U311':7, 'U312': 7, 'U313':7, 'U314':7, 
        'U315':7, 'U316':7, 'U317':7,
        'U318': 7, 'U319':7, 'U321':8, 'U322':8,       
        'U330':9, 'U341':10, 'U342':10, 'U350':11,
        'U361':12, 'U362':12, 'U370':13, 'U411':14, 'U412':14, 
        'U413':14, 'U414':14, 'U415':14,
        'U420': 15
}        
crop_labels={'A10':0, 'A11':0, 'A12':0, 'A13':0,  'A20':0, 'A21':0, 'A30':0, 'A22':0, 'B50':0, 'B10':0,'B20':0, 'B30':0, 'B40':0,'BX1':0, 'BX2':0,
            'CXX1':0, 'CXX2':0, 'CXX3':0, 'CXX4':0, 'CXX5':0,'CXX5':0, 'CXX6':0, 'CXX7':0, 'CXX8':0, 'CXX9':0, 'CXXA':0, 'CXXB':0, 'CXXC':0, 'CXXD':0, 
            'CXXE':0,'G10':0, 'G11':0, 'G12':0, 'G20':0, 'G21':0, 'G22':0, 'G30':0,'G40':0, 'G50':0,'H10':0, 'H11':0, 'H12':0, 'H11':0,'H20':0, 'H21':0,
           'H22':0, 'H23':0,  'B11':1, 'B12':1, 'B13':1, 'B14':1, 'B15':1, 'B16':1, 'B17':1, 'B18':1, 'B19':1, 'B21':2,
            'B22':2, 'B23':2, 'B34':3, 'B35':3, 'B36':3, 'B37':3, 'B31':3, 'B32':3, 'B33':3, 'B41':4, 
            'B42':4, 'B43':4, 'B44':4, 'B45':4, 'B51':5, 'B52':5,'B53':5, 'B54':5, 'B70':7, 'B71':7, 'B72':7, 'B73':7, 'B74':7, 'B75':7,
           'B76':7, 'B77':7, 'B80':7, 'B81':7, 'B82':7, 'B83':7,'B84':7, 'C10':7, 'C20':7, 'C21':7, 'C22':7,'C23':7, 'C30':7, 'C31':7, 'C32':7,
           'C33':7, 'D10':7, 'D20':7, 'B55':8, 'E10':8, 'E20':8, 'E30':8, 'F10':6, 'F20':6, 'F30':6, 'F40':6, '': 9}


class_mapping_lc = {
    0: "Artificial Land", 
    1: "Cropland", 
    2: "Woodland", 
    3: "Shrubland", 
    4: "Grassland", 
    5: "Bare Land", 
    6: "Water", 
    7: "Wetlands", 
    # 10: "Others"
}
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
class_mapping_lu = {0:'Agriculture',1:'Forestry', 2:'Aquaculture and Fishing', 3:'Mining and Quarrying', 
                    4:'Other primary production',
                    5:'Energy production',
                    6:'Industry and Manufacturing', 7: 'Transport, Communication Networks, Storage, Protection Works',
                    8:'Water and Waste Treatment', 9:'Construction',
                    10:'Commerce, Financial, Professional and Information Services', 
                    11:'Community services', 12:'Recreation, Leisure, Sport',13:'Residential', 14:'Abandoned areas', 
                    15:'Semi-natural and natural areas not in use'}

# Convert the labels dictionary to a list of classes
class_dict_lc = [{key: class_mapping_lc[value]} for key, value in labels.items()]

# Create a unique list of classes in order
class_list_lc = list(dict.fromkeys([class_mapping_lc[value] for _, value in labels.items()]))

class_list_lu = list(dict.fromkeys([class_mapping_lu[value] for _, value in lu_labels.items()]))
print(class_list_lc, class_list_lu)
class_list_nuts = list(dict.fromkeys([nuts_labels[value] for value in nuts_labels.keys()]))

print(class_list_nuts)