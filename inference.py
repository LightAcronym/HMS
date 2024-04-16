import streamlit as st
import pandas as pd
import numpy as np
from keras import Model, optimizers, losses, Input
from keras.layers import Dense, GlobalAveragePooling2D, Concatenate
import warnings
import efficientnet.tfkeras as efn
from DataGenerator import DataGenerator
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

    length_sec = len(df) * 2
    st.write(f"""

    The total duration of this spectrogram is {length_sec} seconds (or {length_sec / 60:.2f} minutes).
    The classifier utilizes a 10-minute window of a spectrogram. 

    Please enter the starting point (in seconds) to indicate the start point of the 10-minute window.

    """)
    regions = ['LL', 'RL', 'RP', 'LP']
    start_seconds = st.slider("Select start time (seconds)", min_value=0,
                              max_value=length_sec - 600, step=1)
    r = start_seconds // 2
    if start_seconds is not None:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        for k, ax in enumerate(axes.flat):
            img = data[r:r + 300, k * 100:(k + 1) * 100].T
            img = process_img(img)
            img_color = ax.imshow(img, cmap='viridis', aspect='auto', origin='lower')
            ax.set_title(f"Region: {regions[k]}")
            ax.set_xlabel("Time (sec)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xticks(np.arange(0, 301, 50))
            ax.set_yticks(np.arange(0, 101, 20))
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_yticklabels([f'{i / 5}' for i in range(0, 101, 20)])

            # Optionally, you can add colorbar for each image
            cbar = fig.colorbar(img_color, ax=ax)
            cbar.set_label('Intensity')

        plt.tight_layout()
        st.pyplot(fig)

    return start_seconds

def main(model):
    st.title('Harmful Brain Activity Classifier')

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
        max_vote_diagnosis = columns[np.argmax(res)]

        st.write(f"""
        
        ## Model prediction: 
        
        ### Diagnosis: {max_vote_diagnosis}
        
        {conf[max_vote_diagnosis]}
        """)

        plt.figure(figsize=(10, 6))
        bars = plt.barh(res.columns, res.iloc[0], color='skyblue')  # Plot all bars with skyblue color
        max_index = res.iloc[0].argmax()
        bars[max_index].set_color('red')

        plt.xlabel('Probability')
        plt.title('Distribution of Votes')
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        plt.gca().invert_yaxis()
        for i, v in enumerate(res.values[0]):
            plt.text(v + 0.01, i, str(round(v, 2)), va='center', color='black')
        st.pyplot(plt)


if __name__ == '__main__':
    model = build_model()
    model.load_weights('model_weights.h5')
    main(model)
