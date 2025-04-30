import pandas as pd
import numpy as np
import re

captions_dict = {}

with open("flickr8k\\captions.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line == '':
            continue
        img_caption, caption = line.split(',', 1)
        img_name = img_caption.split('#')[0]
        caption = caption.strip()

        if img_name not in captions_dict:
            captions_dict[img_name] = []
        captions_dict[img_name].append(caption)
# print(captions_dict)

def preprocess_caption(caption):
    caption = caption.lower()
    caption = '<start> ' + caption + ' <end>'
    return caption

for img_name in captions_dict:
    new_captions = []
    for caption in captions_dict[img_name]:
        new_caption = preprocess_caption(caption)
        new_captions.append(new_caption)
    captions_dict[img_name] = new_captions

# print(captions_dict)

from tensorflow.keras.preprocessing.text import Tokenizer

all_captions = []
for caption in captions_dict.values():
    all_captions.extend(caption)

tokenizer = Tokenizer(num_words=10000, oov_token='<unk>')
tokenizer.fit_on_texts(all_captions)

captions_seqs_dict = {}
for img_name, captions in captions_dict.items():
    sequence = tokenizer.texts_to_sequences(captions)
    captions_seqs_dict[img_name] = sequence

# print(captions_seqs_dict)

import os
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tqdm import tqdm
import tensorflow

image_folder = "flickr8k/images/"
base_model = InceptionV3(weights='imagenet')
model =  Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

image_features = {}
image_list = list(captions_seqs_dict.keys())
for img_name in tqdm(image_list):
    img_path = os.path.join(image_folder, img_name)
    try:
        img_array = preprocess_image(img_path)
        feature = model.predict(img_array, verbose=0)
        image_features[img_name] = feature.flatten()    
    except Exception as e:
        print(f"Error processing image {img_name}: {e}" )

from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(seq) for captions in captions_seqs_dict.values() for seq in captions)
embedding_dim = 256

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(embedding_dim, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])  
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def data_generator(captions_seqs_dict, image_features, vocab_size, max_length, batch_size):
    while True:
        x1, x2, y = [], [], []
        for img_name, caption_list in captions_seqs_dict.items():
            for seq in caption_list:
                for i in range(1, len(seq)):
                    in_seq, out_word = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    out_word = to_categorical(out_word, num_classes=vocab_size)[0]

                    x1.append(image_features[img_name])
                    x2.append(in_seq)
                    y.append(out_word)

                    if len(x1) == batch_size:
                        yield [np.array(x1), np.array(x2)], np.array(y)
                        x1, x2, y = [], [], []

batch_size = 64
steps = sum(len(v) for v in captions_seqs_dict.values())

dataset = tensorflow.data.Dataset.from_generator(
    lambda: data_generator(captions_seqs_dict, image_features, vocab_size, max_length, batch_size),
    output_signature=(
        (
            tensorflow.TensorSpec(shape=(None, 2048), dtype=tensorflow.float32),
            tensorflow.TensorSpec(shape=(None, max_length), dtype=tensorflow.int32)
        ),
        tensorflow.TensorSpec(shape=(None, vocab_size), dtype=tensorflow.float32)
    )
)

model.fit(dataset, epochs=20, steps_per_epoch=steps // batch_size, verbose=1)

# model.save('image_captioning_model.h5')

def generate_caption_beam_search(model, tokenizer, photo, max_length, beam_index=3):
    start_token = tokenizer.word_index['start']
    end_token = tokenizer.word_index['end']

    start_seq = [start_token]
    sequences = [[start_seq, 0.0]]

    while len(sequences[0][0]) < max_length:
        all_candidates = []
        for seq, score in sequences:
            padded_seq = pad_sequences([seq], maxlen=max_length, padding='post')
            yhat = model.predict([photo, padded_seq], verbose=0)[0]
            top_preds = np.argsort(yhat)[-beam_index:]

            for word in top_preds:
                new_seq = seq + [word]
                new_score = score + np.log(yhat[word] + 1e-10)
                all_candidates.append([new_seq, new_score])

        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_index]

    final_seq = sequences[0][0]

    # Convert to words and stop at 'end'
    final_caption = []
    for idx in final_seq:
        word = tokenizer.index_word.get(idx, '')
        if word == 'end':
            break
        if word not in ['start', '<unk>', '']:
            final_caption.append(word)

    return ' '.join(final_caption)

from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

# def preprocess_image_for_captioning(image_path):
#     img = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
#     img = tensorflow.keras.preprocessing.image.img_to_array(img)
#     img = tensorflow.keras.applications.inception_v3.preprocess_input(img)
#     return np.expand_dims(img, axis=0)

def preprocess_image_for_captioning(image_array):
    img = keras_image.array_to_img(image_array)  # Convert NumPy array to PIL Image
    img = img.resize((299, 299))  # Resize image
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize if required by your model
    return img_array

def extract_features(image_path, model):
    img = preprocess_image_for_captioning(image_path)
    features = model.predict(img)
    return features

def show_image_with_caption(image_path, caption):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.title(caption)
    plt.show()

# image_path = r"C:\Users\Tanuj Rajput\OneDrive\Desktop\tanujkaggle\10\flickr8k\images\54501196_a9ac9d66f2.jpg"
# image_model = InceptionV3(weights='imagenet')
# image_model = Model(image_model.input, image_model.layers[-2].output)
# features = extract_features(image_path, image_model)

# caption = generate_caption_beam_search(model, tokenizer, features, max_length)
# show_image_with_caption(image_path, caption)

import gradio as gr

def caption_image(image):
    # Preprocess and extract features
    img_array = preprocess_image_for_captioning(image)
    feature = image_model.predict(img_array)
    
    # Generate caption
    caption = generate_caption_beam_search(model, tokenizer, feature, max_length)
    return caption

# Create the Gradio interface
interface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="numpy", label="Upload an image"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Captioning with Deep Learning",
    description="Upload an image and get a descriptive caption using a trained deep learning model.",
)

interface.launch()