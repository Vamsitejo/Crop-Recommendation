import pandas as pd
import pickle

# Load the saved models from .pkl files
rfc = pickle.load(open('model.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))

# Sample input data for prediction
sample_input = {
    'N': [80],
    'P': [40],
    'K': [35],
    'temperature': [25],
    'humidity': [75],
    'ph': [6.5],
    'rainfall': [180]
}

# Creating a DataFrame from the sample input
sample_df = pd.DataFrame(sample_input)

# Transforming the sample input data using the loaded scalers
sample_df = ms.transform(sample_df)
sample_df = sc.transform(sample_df)

# Making predictions using the loaded RandomForestClassifier model
predicted_crop_num = rfc.predict(sample_df)

# Reverse mapping the predicted crop number to the crop label
reverse_crop_dict = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut', 6: 'papaya',
    7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon', 11: 'grapes',
    12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil', 16: 'blackgram',
    17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas', 20: 'kidneybeans',
    21: 'chickpea', 22: 'coffee'
}

predicted_crop = reverse_crop_dict[predicted_crop_num[0]]

print(f"The recommended crop for the given input is: {predicted_crop}")
