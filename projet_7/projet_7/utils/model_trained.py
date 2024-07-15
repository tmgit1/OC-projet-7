from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# load model
loaded_model = load_model('utils/efficient_net_v2_b0_120_classes.keras')

# Define the class labels dictionary
class_labels = {
    'Afghan_hound': 0, 'African_hunting_dog': 1, 'Airedale': 2, 'American_Staffordshire_terrier': 3,
    'Appenzeller': 4, 'Australian_terrier': 5, 'Bedlington_terrier': 6, 'Bernese_mountain_dog': 7,
    'Blenheim_spaniel': 8, 'Border_collie': 9, 'Border_terrier': 10, 'Boston_bull': 11,
    'Bouvier_des_Flandres': 12, 'Brabancon_griffon': 13, 'Brittany_spaniel': 14, 'Cardigan': 15,
    'Chesapeake_Bay_retriever': 16, 'Chihuahua': 17, 'Dandie_Dinmont': 18, 'Doberman': 19,
    'English_foxhound': 20, 'English_setter': 21, 'English_springer': 22, 'EntleBucher': 23,
    'Eskimo_dog': 24, 'French_bulldog': 25, 'German_shepherd': 26, 'German_short': 27,
    'Gordon_setter': 28, 'Great_Dane': 29, 'Great_Pyrenees': 30, 'Greater_Swiss_Mountain_dog': 31,
    'Ibizan_hound': 32, 'Irish_setter': 33, 'Irish_terrier': 34, 'Irish_water_spaniel': 35,
    'Irish_wolfhound': 36, 'Italian_greyhound': 37, 'Japanese_spaniel': 38, 'Kerry_blue_terrier': 39,
    'Labrador_retriever': 40, 'Lakeland_terrier': 41, 'Leonberg': 42, 'Lhasa': 43, 'Maltese_dog': 44,
    'Mexican_hairless': 45, 'Newfoundland': 46, 'Norfolk_terrier': 47, 'Norwegian_elkhound': 48,
    'Norwich_terrier': 49, 'Old_English_sheepdog': 50, 'Pekinese': 51, 'Pembroke': 52, 'Pomeranian': 53,
    'Rhodesian_ridgeback': 54, 'Rottweiler': 55, 'Saint_Bernard': 56, 'Saluki': 57, 'Samoyed': 58,
    'Scotch_terrier': 59, 'Scottish_deerhound': 60, 'Sealyham_terrier': 61, 'Shetland_sheepdog': 62,
    'Shih': 63, 'Siberian_husky': 64, 'Staffordshire_bullterrier': 65, 'Sussex_spaniel': 66,
    'Tibetan_mastiff': 67, 'Tibetan_terrier': 68, 'Walker_hound': 69, 'Weimaraner': 70,
    'Welsh_springer_spaniel': 71, 'West_Highland_white_terrier': 72, 'Yorkshire_terrier': 73,
    'affenpinscher': 74, 'basenji': 75, 'basset': 76, 'beagle': 77, 'black': 78, 'bloodhound': 79,
    'bluetick': 80, 'borzoi': 81, 'boxer': 82, 'briard': 83, 'bull_mastiff': 84, 'cairn': 85, 'chow': 86,
    'clumber': 87, 'cocker_spaniel': 88, 'collie': 89, 'curly': 90, 'dhole': 91, 'dingo': 92, 'flat': 93,
    'giant_schnauzer': 94, 'golden_retriever': 95, 'groenendael': 96, 'keeshond': 97, 'kelpie': 98,
    'komondor': 99, 'kuvasz': 100, 'malamute': 101, 'malinois': 102, 'miniature_pinscher': 103,
    'miniature_poodle': 104, 'miniature_schnauzer': 105, 'otterhound': 106, 'papillon': 107, 'pug': 108,
    'redbone': 109, 'schipperke': 110, 'silky_terrier': 111, 'soft': 112, 'standard_poodle': 113,
    'standard_schnauzer': 114, 'toy_poodle': 115, 'toy_terrier': 116, 'vizsla': 117, 'whippet': 118, 'wire': 119
}

def model_prediction(uploaded_image):
    # Load and preprocess the image
    image = uploaded_image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # Make a prediction
    predictions = loaded_model.predict(image_array)

    # Get the index of the maximum value in the predictions array
    predicted_class_index = np.argmax(predictions)

    # Get the corresponding class label
    predicted_class = [k for k, v in class_labels.items() if v == predicted_class_index][0]

    # Get the probability of the predicted class
    probability = predictions[0][predicted_class_index]

    return predicted_class, probability