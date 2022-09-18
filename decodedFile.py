import os
from tensorflow.keras.models import load_model,Model
from keras.layers import  Input
import pickle as pkl
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

print(os.getcwd())
model_loaded = load_model('Model.h5')


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


# sentence = 'नाव'
# input_seq = tokenizer_input.texts_to_sequences([sentence])
# pad_sequence = pad_sequences(input_seq, maxlen= 30, padding='post')

# prediction = decode_seq(input_seq)
# print("sentance: ",sentence)
# print("predicted Translate:",prediction)

def EncodeAndDecode(text):
    x = tokenizer_input.texts_to_sequences([text])
    pad_sequence = pad_sequences(x, maxlen= 30, padding='post')
    y = decode_seq(pad_sequence)
    y = y.rstrip('end')
    return y

# list1 = [    'पळा',    'कोण',    'वाह']

# for text in list1:
#     print(EncodeAndDecode(text))