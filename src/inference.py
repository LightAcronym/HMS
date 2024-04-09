import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import Model, optimizers, losses, Input
from keras.layers import Dense, GlobalAveragePooling2D, Concatenate
import warnings
import efficientnet.tfkeras as efn
from DataGenerator import DataGenerator
import json
from PIL import Image

# Disable TensorFlow and Keras warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

with open('config.json', 'r') as file:
    conf = json.load(file)


def load_images():
    # Example: Load images from file or any other source
    # Adjust this based on your actual data format
    image_path = f'images/display_img.png'
    image = Image.open(image_path)
    return image


def build_model():
    inp = Input(shape=(128, 256, 4))
    base_model = efn.EfficientNetB0(include_top=False, weights=None, input_shape=None)

    x = [inp[:, :, :, i:i + 1] for i in range(4)]
    x = Concatenate(axis=1)(x)
    x = Concatenate(axis=3)([x, x, x])

    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(6, activation='softmax', dtype='float32')(x)

    model = Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(learning_rate=1e-3)
    loss = losses.KLDivergence()

    model.compile(loss=loss, optimizer=opt)
    return model


def process_img(img):
    img = np.clip(img, np.exp(-4), np.exp(8))
    img = np.log(img)

    ep = 1e-6
    m = np.nanmean(img.flatten())
    s = np.nanstd(img.flatten())
    img = (img - m) / (s + ep)
    img = np.nan_to_num(img, nan=0.0)

    min_val = np.min(img)
    max_val = np.max(img)
    img = (img - min_val) / (max_val - min_val)
    return img


def show_images(df):
    st.write('''
    
    ## 10-min Spectrogram image
    
    ''')
    data = np.array(df)
    large_img = np.ones((400, 300), dtype='float32')

    length_sec = len(df) * 2
    st.write(f"""
    
    The total duration of this spectrogram is {length_sec} seconds (or {length_sec/60:.2f} minutes).
    The classifier utilizes a 10-minute window of a spectrogram. 
    
    Please enter the starting point (in seconds) to indicate the start point of the 10-minute window.
    
    """)
    # start_seconds = st.number_input("Enter here", min_value=0,
    #                               max_value=length_sec - 600, step=1)
    start_seconds = st.slider("Select start time (seconds)", min_value=0,
                              max_value=length_sec - 600, step=1)
    r = start_seconds // 2
    if start_seconds is not None:
        for k in range(4):
            img = data[r:r + 300, k * 100:(k + 1) * 100].T
            img = process_img(img)
            large_img[100 * k:100 * (k + 1), :] = img[:, :]
        large_img_color = plt.cm.viridis(large_img)
        st.image(large_img_color,
                 caption=f"Concatenated spectrogram images of the LL, RL, RP, LP brain regions",
                 use_column_width='auto',
                 width=20
                 )
    return start_seconds
##

def main(model):
    st.title('Harmful Brain Activity Classifier')

    # displaying images of clear-cut examples
    image = load_images()
    st.image(image, use_column_width='auto', width=10)

    uploaded_file = st.file_uploader('Upload a .parquet file', type='parquet')
    if uploaded_file is not None:
        file_name = uploaded_file.name.split('.')[0]
        df = pd.read_parquet(uploaded_file)
        df = df.drop('time', axis=1)
        start = show_images(df)

        test_gen = DataGenerator(np.array(df), mode='test',
                                 start_sec=start)
        prediction = model.predict(test_gen[0], verbose=1)
        columns = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
        res = pd.DataFrame(data=prediction, columns=columns)

        st.write(f"""
        
        ## Model predictions 
        
        Spectrogram ID {file_name}:
        
        """)
        st.write(res)
        max_vote_diagnosis = columns[np.argmax(res)]
        max_vote_probability = np.max(res)
        st.write(
            f"The final diagnosis with the highest vote probability is '{max_vote_diagnosis}', "
            f"with a probability of {max_vote_probability:.2f}")
        st.write("Explanation:")
        st.write(conf[max_vote_diagnosis])


if __name__ == '__main__':
    model = build_model()
    model.load_weights('model_weights.h5')
    main(model)
