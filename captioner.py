import numpy as np
from PIL import Image
import os
import string
from pickle import dump, load
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tqdm import tqdm


def load_doc(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text


def img_capt(filename):
    file = load_doc(filename)
    captions = file.strip().split('\n')
    descriptions = {}
    for caption in captions:
        img, caption = caption.split('\t')
        if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [caption]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions

def txt_clean(captions):
    table = str.maketrans('', '', string.punctuation)
    for img, caps in captions.items():
        for i, img_caption in enumerate(caps):
            img_caption = img_caption.replace("-", " ")
            desc = img_caption.split()
            desc = [word.lower() for word in desc]
            desc = [word.translate(table) for word in desc]
            desc = [word for word in desc if len(word) > 1 and word.isalpha()]
            img_caption = ' '.join(desc)
            captions[img][i] = img_caption
    return captions


def txt_vocab(descriptions):
    vocab = set()
    for key in descriptions.keys():
        for d in descriptions[key]:
            vocab.update(d.split())
    return vocab


def save_descriptions(descriptions, filename):
    lines = []
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc)
    data = "\n".join(lines)
    with open(filename, "w") as file:
        file.write(data)


dataset_text = "D:/shikha/Project - Image Caption Generator/Flickr_8k_text"
dataset_images = "D:/shikha/Project - Image Caption Generator/Flicker8k_Dataset"


filename = os.path.join(dataset_text, "Flickr8k.token.txt")
descriptions = img_capt(filename)
print("Length of descriptions =", len(descriptions))


clean_descriptions = txt_clean(descriptions)


vocabulary = txt_vocab(clean_descriptions)
print("Length of vocabulary =", len(vocabulary))

save_descriptions(clean_descriptions, "descriptions.txt")

# Model
def extract_features(directory):
    model = Xception(include_top=False, pooling='avg')
    features = {}
    for pic in tqdm(os.listdir(directory)):
        file = os.path.join(directory, pic)
        image = Image.open(file)
        image = image.resize((299, 299))
        image = np.array(image)
        if image.shape[2] == 4:
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image / 127.5 - 1.0
        feature = model.predict(image)
        img_id = pic.split('.')[0]
        features[img_id] = feature
    return features


features = extract_features(dataset_images)
dump(features, open("features.p", "wb"))


features = load(open("features.p", "rb"))


def load_photos(filename):
    file = load_doc(filename)
    photos = file.strip().split('\n')
    return photos


def load_clean_descriptions(filename, photos):
    file = load_doc(filename)
    descriptions = {}
    for line in file.strip().split('\n'):
        words = line.split()
        image_id, image_caption = words[0], words[1:]
        if image_id in photos:
            if image_id not in descriptions:
                descriptions[image_id] = []
            desc = ' '.join(image_caption)
            descriptions[image_id].append(desc)
    return descriptions


def load_features(photos):
    all_features = load(open("features.p", "rb"))
    features = {k: all_features[k] for k in photos}
    return features


filename = os.path.join(dataset_text, "Flickr_8k.trainImages.txt")
train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
train_features = load_features(train_imgs)


def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer


tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size:", vocab_size)


def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)

max_length = max_length(descriptions)
print("Description Length:", max_length)


def data_generator(descriptions, features, tokenizer, max_length):
    while True:
        for key, description_list in descriptions.items():
            feature = features[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, description_list, feature)
            yield [[in_img, in_seq], out_word]

def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = [], [], []
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

[a, b], c = next(data_generator(train_descriptions, train_features, tokenizer, max_length))
print(a.shape, b.shape, c.shape)

#  model
from keras.utils import plot_model

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


print('Dataset: ', len(train_imgs))
print('Descriptions: train=', len(train_descriptions))
print('Photos: train=', len(train_features))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_length)

model = define_model(vocab_size, max_length)
epochs = 10
steps = len(train_descriptions)
os.mkdir("models")
for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save(f"models/model_{i}.h5")


import matplotlib.pyplot as plt
import argparse

def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Can't open image! Ensure that image path and extension is correct")
        return None
    image = image.resize((299, 299))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5 - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]

sequence = pad_sequences([sequence], maxlen=max_length)
       pred = model.predict([photo,sequence], verbose=0)
       pred = np.argmax(pred)
       word = word_for_id(pred, tokenizer)
if word is None:
           break
       in_text += ' ' + word
if word == 'end':
           break
return in_text
max_length = 32

tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

img_path=r'C:Users\pashu\Desktop\Image1'
photo = extract_features(img_path, xception_model)
img = Image.open(img_path)
description = generate_desc(model, tokenizer, photo, max_length)
print("nn")
print(description)
plt.imshow(img)
