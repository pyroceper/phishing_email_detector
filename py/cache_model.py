import pickle
import os.path

def save_model(name, model):
    try:
        with open(name+'.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
    except Exception as e:
        print("Error saving "+name+" "+model+": ", str(e))

def check_cache(name):
    check_file = os.path.isfile(name+'.pkl')
    return check_file

def load_model(name):
    try:
        with open(name+'.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
            return model
    except Exception as e:
        print("Error loading "+name+" "+model+": ", str(e))
