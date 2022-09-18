import re
import numpy as np
import pandas as pd
import string
from string import digits

from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Input, Dense,Embedding
from keras.models import Model

from tensorflow.keras.preprocessing.sequence import pad_sequences


import pickle as pkl

from sklearn.model_selection import train_test_split
import tensorflow as tf

# For spliting the data for training and testing

# Reading CSV File
df = pd.read_csv('language_data.csv')

# Data Cleaning
english_text = df['English'].values
marathi_text = df['Marathi'].values

# print(english_text[0], marathi_text[0])

# print(len(english_text), len(marathi_text))

english_text_ = [x.lower() for x in english_text]
marathi_text_ = [x.lower() for x in marathi_text]

# print(type(english_text_), type(marathi_text_))

# Text Processing

english_text_ = [re.sub("'", '', x) for x in english_text_]
marathi_text_ = [re.sub("'", '', x) for x in marathi_text_]


def remove_punc(text_list):
    table = str.maketrans('', '', string.punctuation)
    removed_punc_text = []
    for sent in text_list:
        sentance = [w.translate(table) for w in sent.split(' ')]
        removed_punc_text.append(' '.join(sentance))
    return removed_punc_text


english_text_ = remove_punc(english_text_)
marathi_text_ = remove_punc(marathi_text_)

# removing the digits from english sentances
remove_digits = str.maketrans('', '', digits)
removed_digits_text = []
for sent in english_text_:
    sentance = [w.translate(remove_digits) for w in sent.split(' ')]
    removed_digits_text.append(' '.join(sentance))

english_text_ = removed_digits_text

# removing the digits from the marathi sentances
marathi_text_ = [re.sub("[२३०८१५७९४६]", "", x) for x in marathi_text_]
marathi_text_ = [re.sub("[\u200d]", "", x) for x in marathi_text_]

# removing the stating and ending whitespaces

english_text_ = [x.strip() for x in english_text_]
marathi_text_ = [x.strip() for x in marathi_text_]

# Putting the start and end words in the marathi sentances
marathi_text_ = ["start " + x + " end" for x in marathi_text_]

# print(english_text_[0],marathi_text_[0])

x = marathi_text_
y = english_text_

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# print(len(X_train), len(y_train), len(X_test), len(y_test))

# print(x[0], y[0])

# print(X_train[0], y_train[0])


# preparing data for the word embedding
def Max_length(data):
    max_length_ = max([len(x.split(' ')) for x in data])
    return max_length_


# Training data
max_length_english = Max_length(X_train)
max_lenght_marathi = Max_length(y_train)

# Test data
max_length_english_test = Max_length(X_test)
max_lenght_marathi_test = Max_length(y_test)

print(max_lenght_marathi, max_length_english,
      max_lenght_marathi_test, max_length_english_test)

def tokenizer_(text_data):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(text_data)
  return tokenizer

tokenizer_input = tokenizer_(X_train)
vocab_size_input = len(tokenizer_input.word_index) + 1
tokenizer_target = tokenizer_(y_train)
vocab_size_target = len(tokenizer_target.word_index) + 1

with open('tokenizer_input.pkl','wb') as f:
  pkl.dump(tokenizer_input,f)

with open('tokenizer_target.pkl','wb') as f:
  pkl.dump(tokenizer_target,f)
  
# pkl.dump(tokenizer_input, open('tokenizer_input2.pkl', 'wb'))
# pkl.dump(tokenizer_target, open('tokenizer_target2.pkl', 'wb'))

print(vocab_size_input,vocab_size_target)

def generator_batch(X= X_train,Y=y_train, batch_size=128):
  while True:
    for j in range(0, len(X), batch_size):
      encoder_data_input = np.zeros((batch_size,max_length_english),dtype='float32') #metrix of batch_size*max_length_english
      decoder_data_input = np.zeros((batch_size,max_lenght_marathi),dtype='float32') #metrix of batch_size*max_length_marathi
      decoder_target_input = np.zeros((batch_size,max_lenght_marathi,vocab_size_target),dtype='float32') # 3d array one hot encoder decoder target data
      for i, (input_text,target_text) in enumerate(zip(X[j:j+batch_size],Y[j:j+batch_size])):
        for t, word in enumerate(input_text.split()):
          encoder_data_input[i,t] = tokenizer_input.word_index[word] # Here we are storing the encoder 
                                                                     #seq in row here padding is done automaticaly as 
                                                                     #we have defined col as max_lenght
        for t, word in enumerate(target_text.split()):
          decoder_data_input[i,t] = tokenizer_target.word_index[word] # same for the decoder sequence
          if t>0:
            decoder_target_input[i,t-1,tokenizer_target.word_index[word]] = 1 #target is one timestep ahead of decoder input because it does not have 'start tag'
      yield ([encoder_data_input,decoder_data_input],decoder_target_input)


latent_dim = 50
encoder_inputs = Input(shape=(None,),name="encoder_inputs")
emb_layer_encoder = Embedding(vocab_size_input,latent_dim, mask_zero=True)(encoder_inputs)
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(emb_layer_encoder)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,),name="decoder_inputs")


emb_layer_decoder = Embedding(vocab_size_target,latent_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(emb_layer_decoder, initial_state=encoder_states)
decoder_dense = Dense(vocab_size_target, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_samples = len(X_train)
val_samples = len(X_test)
batch_size = 16
epochs = 60

model.fit_generator(generator = generator_batch(X_train, y_train, batch_size = batch_size),steps_per_epoch = train_samples//batch_size, epochs=epochs)

"""## Saving the trained model into H5 type"""

model.save('trainedModel.h5')

model_loaded = tf.keras.models.load_model('trainedModel.h5')



latent_dim = 50
#inference encoder
encoder_inputs_inf = model_loaded.input[0] #Trained encoder input layer
encoder_outputs_inf, inf_state_h, inf_state_c = model_loaded.layers[4].output # retoring the encoder lstm output and states
encoder_inf_states = [inf_state_h,inf_state_c]
encoder_model = Model(encoder_inputs_inf,encoder_inf_states)

#inference decoder
# The following tensor will store the state of the previous timestep in the "starting the encoder final time step"
decoder_state_h_input = Input(shape=(latent_dim,)) #becase during training we have set the lstm unit to be of 50
decoder_state_c_input = Input(shape=(latent_dim,))
decoder_state_input = [decoder_state_h_input,decoder_state_c_input]

# # inference decoder input
decoder_input_inf = model_loaded.input[1] #Trained decoder input layer
# decoder_input_inf._name='decoder_input'
decoder_emb_inf = model_loaded.layers[3](decoder_input_inf)
decoder_lstm_inf = model_loaded.layers[5]
decoder_output_inf, decoder_state_h_inf, decoder_state_c_inf = decoder_lstm_inf(decoder_emb_inf, initial_state =decoder_state_input)
decoder_state_inf = [decoder_state_h_inf,decoder_state_c_inf]
#inference dense layer
dense_inf = model_loaded.layers[6]
decoder_output_final = dense_inf(decoder_output_inf)# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_model = Model([decoder_input_inf]+decoder_state_input,[decoder_output_final]+decoder_state_inf)

with open('tokenizer_input.pkl','rb') as f:
  tokenizer_input = pkl.load(f)
with open('tokenizer_target.pkl','rb') as f:
  tokenizer_target = pkl.load(f)
# Creating the reverse mapping to get the word from the index in the sequence
reverse_word_map_input = dict(map(reversed, tokenizer_input.word_index.items()))
reverse_word_map_target = dict(map(reversed, tokenizer_target.word_index.items()))


# Code to predct the input sentences translation
def decode_seq(input_seq):
  state_values_encoder = encoder_model.predict(input_seq)
  target_seq = np.zeros((1,1))
  target_seq[0, 0] = tokenizer_target.word_index['start']
  stop_condition = False
  decoder_sentance = ''
  # print("Beforee the while loop")
  while not stop_condition:
    sample_word , decoder_h,decoder_c= decoder_model.predict([target_seq] + state_values_encoder)
    # print("sample_word: =>",sample_word)
    sample_word_index = np.argmax(sample_word[0,-1,:])
    # print("sample_word_index: ",sample_word_index)
    decoder_word = reverse_word_map_target[sample_word_index]
    decoder_sentance += ' '+ decoder_word
    if (decoder_word == 'end' or 
        len(decoder_sentance) > 80):
        stop_condition = True
    target_seq[0, 0] = sample_word_index
    state_values_encoder = [decoder_h,decoder_c]
  return decoder_sentance


sentence = 'नाव'

input_seq = tokenizer_input.texts_to_sequences([sentance])
pad_sequence = pad_sequences(input_seq, maxlen= 30, padding='post')

prediction = decode_seq(input_seq)
print("sentance: ",sentence)
print("predicted Translate:",prediction)
