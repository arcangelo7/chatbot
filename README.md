# Chatbot Project

This project implements a simple chatbot using a Sequence-to-Sequence (Seq2Seq) model with LSTM layers. The chatbot is trained on a dataset of question-answer pairs and can generate responses to user input.

## Project Structure

The project consists of two main Python scripts:

1. `train_chatbot.py`: This script preprocesses the data, creates and trains the Seq2Seq model, and saves the trained model and necessary configurations.
2. `run_chatbot.py`: This script loads the trained model and provides an interface for interacting with the chatbot.

## Requirements

- Python 3.10
- TensorFlow 2.x
- NumPy
- Keras

You can install the required packages using pip:

```
pip install -r requirements.txt
```

## Setup and Training

1. Prepare your dataset:
   - Create a file named `dataset.txt` in a `data` directory.
   - Format the file with alternating lines of questions and answers.

2. Run the training script:
   ```
   python train_chatbot.py
   ```
   This will preprocess the data, create and train the model, and save the following files:
   - `chatbot_model.h5`: The trained model
   - `tokenizer.pkl`: The tokenizer used for text processing
   - `config.pkl`: Configuration parameters for the model

## Running the Chatbot

After training the model, you can interact with the chatbot by running:

```
python run_chatbot.py
```

This will start an interactive session where you can type messages and receive responses from the chatbot. Type 'exit' or 'quit' to end the session.

## How It Works

1. The `train_chatbot.py` script:
   - Preprocesses the text data, tokenizing and padding sequences.
   - Creates a Seq2Seq model with an encoder-decoder architecture using LSTM layers.
   - Trains the model on the preprocessed data.
   - Saves the trained model and necessary configurations.

2. The `run_chatbot.py` script:
   - Loads the trained model, tokenizer, and configuration.
   - Implements the inference process, using the encoder to process input and the decoder to generate responses.
   - Provides an interactive loop for chatting with the bot.

## Customization

You can customize the chatbot by modifying parameters in `train_chatbot.py`, such as:
- `embedding_dim`: Size of word embeddings
- `units`: Number of LSTM units
- `batch_size` and `epochs`: Training parameters

Remember to retrain the model after making changes.

## Limitations

This is a basic implementation and may not produce highly coherent or contextually appropriate responses. The quality of responses depends largely on the training data and the complexity of the model.

## Future Improvements

- Use more advanced architectures like Transformer.
- Incorporate larger and more diverse datasets for training.