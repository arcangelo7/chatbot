import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Step 1: Preprocessing del Dataset

# Carica il dataset
with open('data/dataset.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()  # Leggi tutte le linee del file e memorizzale in una lista

# Prepara le coppie domanda-risposta
pairs = []
for i in range(0, len(lines), 2):  # Itera ogni due linee
    question = lines[i].strip().lower()  # Rimuove spazi bianchi e converte in minuscolo la domanda
    answer = lines[i + 1].strip().lower()  # Rimuove spazi bianchi e converte in minuscolo la risposta
    pairs.append((question, answer))  # Aggiungi la coppia (domanda, risposta) alla lista

# pairs = [("hello", "hi"), ("how are you?", "I am fine, thank you")]

# Tokenizza il testo
questions = []
for pair in pairs:
    questions.append(pair[0])  # Estrae tutte le domande dalle coppie
answers = []
for pair in pairs:
    answers.append(pair[1])  # Estrae tutte le risposte dalle coppie

# Converte il testo in sequenze di numeri
tokenizer = Tokenizer()

# La funzione fit_on_texts è un metodo della classe Tokenizer di Keras, 
# che viene utilizzata per convertire un testo in una sequenza di numeri. 
# Questo metodo analizza il testo fornito, costruisce un vocabolario di parole 
# e assegna un indice a ciascuna parola basato sulla frequenza di apparizione. 
tokenizer.fit_on_texts(questions + answers)  # Costruisce il vocabolario basato sul testo delle domande e risposte
# texts = ["hello how are you", "I am fine thank you", "how are you doing today"]
# Output: {'you': 1, 'how': 2, 'are': 3, 'hello': 4, 'i': 5, 'am': 6, 'fine': 7, 'thank': 8, 'doing': 9, 'today': 10}


# La funzione texts_to_sequences della classe Tokenizer di Keras converte il testo 
# in una sequenza di numeri. Ogni numero rappresenta l'indice di una parola nel 
# vocabolario costruito dal tokenizer.
# questions = ["hello", "how are you", "what is your name"]
# Supponendo che fit_on_texts sia già stato eseguito su questions + answers
# print(tokenizer.word_index)
# Output: {'i': 1, 'am': 2, 'hello': 3, 'how': 4, 'are': 5, 'you': 6, 'what': 7, 'is': 8, 'your': 9, 'name': 10, 'hi': 11, 'fine': 12, 'thank': 13, 'a': 14, 'chatbot': 15}
# question_sequences = tokenizer.texts_to_sequences(questions)
# Output: [[3], [4, 5, 6], [7, 8, 9, 10]]
question_sequences = tokenizer.texts_to_sequences(questions) # Converte il testo in sequenze di numeri
answer_sequences = tokenizer.texts_to_sequences(answers)

# Aggiungi un token di inizio e di fine alle risposte
start_token = len(tokenizer.word_index) + 1  # Definisce il token di inizio come il prossimo indice disponibile
end_token = len(tokenizer.word_index) + 2  # Definisce il token di fine come il successivo indice disponibile
tokenizer.word_index['<start>'] = start_token  # Aggiunge '<start>' al vocabolario
tokenizer.word_index['<end>'] = end_token  # Aggiunge '<end>' al vocabolario

for i in range(len(answer_sequences)):
    answer_sequences[i] = [start_token] + answer_sequences[i] + [end_token]  # Aggiunge i token di inizio e fine a ogni sequenza di risposta

# answer_sequences = [[10, 5, 11], [10, 6, 7, 8, 9, 1, 11]]

# Padding delle sequenze per avere tutte la stessa lunghezza
max_length_questions = max(len(seq) for seq in question_sequences)
max_length_answers = max(len(seq) for seq in answer_sequences)
# Indica che il padding (l'aggiunta di valori) deve essere fatto alla fine (post) di ogni sequenza.
question_sequences = pad_sequences(question_sequences, maxlen=max_length_questions, padding='post')
answer_sequences = pad_sequences(answer_sequences, maxlen=max_length_answers, padding='post')
# Esempio: Se max_length_questions è 5 e abbiamo le seguenti sequenze:
# [1, 2, 3]
# [4, 5, 6, 7]
# [8, 9]
# Dopo il padding, diventeranno:
# [1, 2, 3, 0, 0]
# [4, 5, 6, 7, 0]
# [8, 9, 0, 0, 0]

# Step 2: Creazione del Modello Seq2Seq

# Un modello Seq2Seq (Sequence to Sequence) è una specifica architettura 
# di reti neurali progettata per trasformare una sequenza di input in una 
# sequenza di output, con lunghezze potenzialmente diverse.
# Questa architettura è particolarmente utile per compiti come la traduzione automatica, 
# il riassunto di testi, e i chatbot. 
# La struttura di base di un modello Seq2Seq comprende un encoder e un decoder, 
# entrambi solitamente implementati utilizzando LSTM (Long Short-Term Memory). 

# LSTM (Long Short-Term Memory) è un tipo di rete neurale ricorrente (RNN) 
# progettata per apprendere dipendenze a lungo termine in sequenze di dati.
# Ogni LSTM ha celle di memoria che possono mantenere informazioni per lunghi 
# periodi di tempo. Queste celle sono progettate per preservare il gradiente durante 
# l'addestramento, evitando il problema della scomparsa del gradiente.

# Dimensioni del vocabolario e embedding

# Aggiunge uno per includere il token di padding (zero) utilizzato durante 
# il padding delle sequenze per assicurarsi che tutte le sequenze abbiano la stessa lunghezza.
vocab_size = len(tokenizer.word_index) + 1  

# Un embedding è una rappresentazione delle parole in uno spazio vettoriale 
# di dimensione embedding_dim.

# Gli embedding sono utili perché permettono al modello di catturare 
# le relazioni semantiche tra le parole. 
# Ad esempio, le parole con significati simili avranno vettori 
# di embedding vicini nello spazio vettoriale.
# La dimensione 256 è un valore comunemente utilizzato che 
# bilancia la capacità di rappresentazione con l'efficienza computazionale.
embedding_dim = 256  # Dimensione degli embedding

# Le unità LSTM sono responsabili di mantenere le informazioni 
# lungo la sequenza di input, permettendo al modello di ricordare 
# informazioni a lungo termine e di gestire dipendenze a lungo termine nel testo.
units = 512  # Numero di unità LSTM

# Encoder

# L'encoder è una parte cruciale del modello Seq2Seq (Sequence to Sequence) 
# utilizzato per convertire una sequenza di input (domande) in una 
# rappresentazione interna (stati), che il decoder utilizzerà 
# per generare una sequenza di output (risposte). 

# Definire la forma degli input è necessario per creare il modello di rete neurale. 
# Indica al modello cosa aspettarsi come input.
encoder_inputs = Input(shape=(max_length_questions,))  # Input del encoder con lunghezza massima delle sequenze

# Strato di embedding per convertire gli indici in vettori
# Word2Vec
# Come funziona Word2Vec?
    # Word2Vec può essere addestrato in due modi principali:
        # CBOW (Continuous Bag of Words): Questo approccio predice una parola target basandosi sul contesto circostante (cioè le parole vicine).
        # Skip-gram: Questo modello, al contrario, cerca di predire le parole di contesto a partire dalla parola target.
# embedding("re")−embedding("uomo")+embedding("donna")≈embedding("regina")
encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)  

# L'LSTM è una versione avanzata delle reti neurali ricorrenti (RNN) 
# che può memorizzare informazioni a lungo termine, 
# essenziale per comprendere sequenze lunghe come frasi.
encoder_lstm = LSTM(units, return_state=True)  # Strato LSTM per l'encoder che ritorna anche gli stati

# Questo applica il layer LSTM agli embedding di input.
# encoder_outputs sono le uscite del layer LSTM per ogni timestep.
# state_h e state_c sono rispettivamente lo stato nascosto e lo stato della cella alla fine della sequenza.
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)  # Calcola le uscite e gli stati
encoder_states = [state_h, state_c]  # Memorizza gli stati dell'encoder

# Decoder
# Il decoder nel modello Seq2Seq prende gli stati finali dell'encoder e gli input del decoder, 
# li passa attraverso un layer di embedding e un layer LSTM, 
# e produce una sequenza di probabilità per ogni parola del vocabolario tramite 
# un layer Dense con softmax. Gli stati iniziali del decoder sono impostati 
# agli stati finali dell'encoder per mantenere il contesto, permettendo 
# al modello di generare una risposta coerente basata sulla sequenza di input.

# Input del decoder con lunghezza massima delle sequenze. Non permette una lunghezza variabile, che si adatta
decoder_inputs = Input(shape=(None,))   
decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)  # Strato di embedding per il decoder
decoder_lstm = LSTM(units, return_sequences=True, return_state=True)  # Strato LSTM per il decoder che ritorna le sequenze e gli stati. return_sequences=True fa sì che l'LSTM restituisca l'intera sequenza di output, necessaria per generare parole multiple.
# Inizializza con gli stati dell'encoder. 
# Gli underscore _ rappresentano gli stati state_h e state_c che vengono calcolati 
# ma non utilizzati qui direttamente.
# initial_state=encoder_states imposta gli stati iniziali del decoder agli stati 
# finali dell'encoder, trasferendo così il contesto appreso.
# Impostare gli stati iniziali del decoder agli stati finali dell'encoder 
# permette al decoder di iniziare la generazione della sequenza di output 
# con il contesto della sequenza di input.
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
# Questo crea un layer Dense (completamente connesso) con un'unità per ogni parola nel vocabolario.
# Il layer Dense con softmax è utilizzato per predire la parola successiva nella sequenza di output. 
# Softmax normalizza le uscite in probabilità, permettendo di scegliere la parola 
# con la probabilità più alta come output del decoder.
decoder_dense = Dense(vocab_size, activation='softmax')  # Strato denso finale con softmax per la predizione delle parole

# Questo applica il layer Dense alle uscite del layer LSTM del decoder.
# decoder_outputs ora contiene le probabilità di ogni parola del vocabolario 
# per ogni timestep della sequenza di output.
decoder_outputs = decoder_dense(decoder_outputs)  # Calcola le uscite finali del decoder

# Definisci il modello
# questa linea di codice combina l'encoder e il decoder in un singolo modello di rete neurale
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compila il modello
# Questa funzione configura il processo di apprendimento del modello. 
# Definisce come il modello verrà addestrato.

# L'ottimizzatore è l'algoritmo utilizzato per aggiornare i pesi del modello durante l'addestramento. 
# Adam (Adaptive Moment Estimation) è un algoritmo di ottimizzazione
# La funzione di perdita è usata per valutare quanto bene il modello sta performando durante l'addestramento.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()  # Mostra il sommario del modello

# Step 3: Addestramento del Modello

# Obiettivo: Creare le sequenze di input per il decoder.

# answer_sequences è un array NumPy 2D contenente le sequenze di risposta tokenizzate e padded.
# [:, :-1] è una operazione di slicing che significa:
# : seleziona tutte le righe
# :-1 seleziona tutte le colonne tranne l'ultima
# In pratica, questa operazione rimuove l'ultimo token da ogni sequenza di risposta. Ecco perché:
# Durante l'addestramento, il decoder deve imparare a prevedere il prossimo token 
# basandosi sui token precedenti.
# L'input del decoder in ogni timestep dovrebbe essere il token che precede 
# quello che stiamo cercando di prevedere.
# Rimuovendo l'ultimo token, creiamo la sequenza di input corretta per il decoder.

decoder_input_data = answer_sequences[:, :-1] 

# Obiettivo: Creare le sequenze di output target per il decoder.
# seleziona tutte le colonne a partire dalla seconda (indice 1)
# In pratica, questa operazione rimuove il primo token da ogni sequenza di risposta. Ecco perché:
# Il target del decoder in ogni timestep dovrebbe essere il token successivo rispetto all'input.
# Rimuovendo il primo token (che è sempre <start>), creiamo la sequenza target corretta per il decoder.
decoder_target_data = answer_sequences[:, 1:]

# Addestramento del modello
# [question_sequences, decoder_input_data] sono le sequenze di input del modello. 
# question_sequences sono le sequenze di input dell'encoder 
# e decoder_input_data sono le sequenze di input del decoder.

# decoder_target_data sono le etichette di output target che il modello deve predire.

# batch_size=64 indica che il modello verrà addestrato con 64 campioni alla volta.

# epochs=300 indica che l'addestramento sarà eseguito per 300 epoche, 
# ovvero il modello vedrà l'intero set di dati 300 volte.

# validation_split=0.2 indica che il 20% dei dati sarà utilizzato per 
# la validazione durante l'addestramento, per monitorare le prestazioni del modello su dati non visti.
model.fit([question_sequences, decoder_input_data], decoder_target_data,
          batch_size=128, epochs=300, validation_split=0.2)

# Step 4: Salvare il Modello
model.save('chatbot_model.h5')

# Salva il tokenizer e i parametri del modello per la successiva elaborazione
import pickle

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('config.pkl', 'wb') as f:
    pickle.dump({'max_length_questions': max_length_questions, 'max_length_answers': max_length_answers}, f)