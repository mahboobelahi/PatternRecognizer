# Importing modules
import tensorflow  as tf
import json
import numpy as np


# loading transformer/scaler Object to scale both features between 0 and 1
# Load_scaler = joblib.load('pallet-scaler.save')
# Power_scaler = joblib.load('Power-scaler.save')


def predict(**data_in):
    """

    :param data: {
  "powerConsumption": num,
  "load": num
}
    :return:
    """
    print(f"Data Type: {type(data_in)}")
    print(data_in)
    "model loading and data transfor"
    model = tf.keras.models.load_model('./model/M_iter3_1.h5', compile=True)
    features_1 = np.array(np.append([data_in.get("data")["powerConsumption"]],
                                     [data_in.get("data")["load"]]), ndmin=2)
    pred = np.argmax(model.predict(features_1), axis=1) # need to pus that vakue to message bus
    json_results = json.dumps({"BT_Class":str(pred)}, indent=4)
    return json_results
"""
    from predict import predict
    predict({"data_in": {"powerConsumption": 0.175, "load": 0.73}})
	Sample input
	power 	load
	0.26  	0.13	
	0.81	1	
	0.15	0.07	

"""