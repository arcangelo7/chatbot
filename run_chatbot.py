import pickle

import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Carica il modello e il tokenizer
model = load_model('chatbot_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('config.pkl', 'rb') as f:
    config = pickle.load(f)

# Parametri del modello
units = 512
max_length_questions = config['max_length_questions']
max_length_answers = config['max_length_answers']

# Step 5: Inference e Generazione delle Risposte
# L'inference, o inferenza, è il processo mediante il quale il modello, 
# una volta addestrato, viene utilizzato per fare previsioni. 
# Nel caso di un chatbot, questo significa generare risposte basate sugli input dell'utente. 

# Modello Encoder per l'Inference
# Prima, dobbiamo creare il modello encoder per l'inference. 
# Questo modello prenderà come input una sequenza di parole (una frase) 
# e produrrà gli stati interni del LSTM, che saranno utilizzati come input iniziali per il decoder.
encoder_inputs = model.input[0]  # Input del encoder
encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output  # Strato LSTM encoder
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)  # Crea il modello encoder con gli stati

# Modello decoder per l'inference
# Ora, creiamo il modello decoder per l'inference. 
# Questo modello genererà la risposta una parola alla volta, utilizzando i seguenti passaggi:
# Inputs per gli Stati del Decoder: Iniziamo con gli stati generati dall'encoder.
decoder_inputs = model.input[1]  # Input del decoder
decoder_state_input_h = Input(shape=(units,), name='input_3')  # Input per lo stato h del decoder
decoder_state_input_c = Input(shape=(units,), name='input_4')  # Input per lo stato c del decoder
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Strato LSTM del Decoder: Genera la sequenza di output e aggiorna gli stati.
decoder_embedding = model.layers[3]
decoder_lstm = model.layers[5]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding(decoder_inputs), initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]  # Stati aggiornati del decoder

# Strato Dense del Decoder: Calcola la probabilità di ciascuna parola nel vocabolario per l'output corrente.
decoder_dense = model.layers[6]
decoder_outputs = decoder_dense(decoder_outputs)

# Crea il modello decoder con uscite e stati
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)  # Crea il modello decoder con uscite e stati

# Funzione per generare risposte
def decode_sequence(input_sentence):
    # 1. Preparazione degli stati iniziali

    # Converte la frase di input in una sequenza numerica
    input_sequence = tokenizer.texts_to_sequences([input_sentence])
    # Padding della sequenza per avere la stessa lunghezza
    input_sequence = pad_sequences(input_sequence, maxlen=max_length_questions, padding='post')
    # Genera la risposta utilizzando la funzione decode_sequence

    # Encode the input as state vectors.
    # Utilizza il modello encoder per predire gli stati interni (h e c) dalla sequenza di input
    states_value = encoder_model.predict(input_sequence)

    # 2. Inizializzazione della sequenza target
    # Generate empty target sequence of length 1.
    # Inizializza la sequenza target con una dimensione di (1, 1)
    target_seq = np.zeros((1, 1))
    # Imposta il primo valore della sequenza target come il token di inizio
    target_seq[0, 0] = tokenizer.word_index['<start>']

    # 3. Ciclo di generazione della risposta
    stop_condition = False
    decoded_sentence = ''  # Inizializza la frase decodificata come una stringa vuota

    while not stop_condition:
        # 4. Generazione delle parole una per una

        # Predice i token successivi utilizzando il modello decoder
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # 5. Estrazione della parola campionata
        # Campiona un token (parola) dalla distribuzione delle probabilità
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = None
        for word, index in tokenizer.word_index.items():
            if sampled_token_index == index:
                sampled_word = word
                break
        
        # 6. Verifica della condizione di stop

        # Se la parola campionata è il token di fine o la lunghezza massima della sequenza è stata raggiunta, ferma la generazione
        if sampled_word == '<end>' or len(decoded_sentence.split()) >= max_length_answers:
            stop_condition = True
        else:
            # 7. Aggiunta della parola alla risposta e aggiornamento della sequenza target

            # Altrimenti, aggiungi la parola alla frase decodificata
            decoded_sentence += ' ' + sampled_word

            # Aggiorna la sequenza target (di lunghezza 1)
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Aggiorna gli stati
            states_value = [h, c]

    return decoded_sentence.strip()

# Esempio di utilizzo

# Ciclo per interazione continua
while True:
    input_sentence = input("Tu: ")
    if input_sentence.lower() in ['exit', 'quit']:
        print("Chatbot: Arrivederci!")
        break
    response = decode_sequence(input_sentence)
    print(f"Chatbot: {response}")