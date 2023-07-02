import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0])

def get_location_names():
    if __locations is None:
        load_saved_artifacts()
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model
    
    with open("/home/pranav3229/User/MLStuff/bhp/server/artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    
    with open("/home/pranav3229/User/MLStuff/bhp/server/artifacts/price_predictor_model.pickle", "rb") as f:
        __model = pickle.load(f)
    
    print("loading saved artifacts...done")

# Call the load_saved_artifacts function when the module is imported
load_saved_artifacts()



if __name__=='__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('Hebbal',1000,3,3))