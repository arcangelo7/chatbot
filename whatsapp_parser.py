import re


def extract_message(line):
    # Rimuove il timestamp e il nome del mittente
    message = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} - [^:]*: ', '', line)
    return message.strip()

def process_chat(input_file, output_file):
    with open(input_file, 'r', encoding='utf8') as file:
        lines = file.readlines()

    # Rimuovi tutte le occorrenze di "<Media omessi>" e le linee vuote
    cleaned_lines = [line.strip() for line in lines if line.strip() and '<Media omessi>' not in line]

    # Unire i messaggi consecutivi e rimuovere i duplicati
    merged_messages = []
    current_message = ''

    for line in cleaned_lines:
        if re.match(r'\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} - ', line):
            if current_message:
                merged_messages.append(current_message.strip())
            current_message = extract_message(line)
        else:
            current_message += ' ' + line.strip()

    if current_message:
        merged_messages.append(current_message.strip())

    # Rimuovi i duplicati mantenendo l'ordine
    unique_messages = []
    for message in merged_messages:
        if message not in unique_messages:
            unique_messages.append(message)

    # Estrazione delle coppie domanda-risposta
    pairs = []
    for i in range(1, len(unique_messages)):
        question = unique_messages[i-1].lower()
        answer = unique_messages[i].lower()
        if question and answer:
            pairs.append((question, answer))

    # Salva le coppie in un file di testo
    with open(output_file, 'w', encoding='utf8') as f:
        for question, answer in pairs:
            f.write(f"{question}\n")
            f.write(f"{answer}\n")

    print(f"Coppie domanda-risposta salvate in '{output_file}'.")

# Uso della funzione
process_chat('data/Chat WhatsApp con Corrado Camponeschi.txt', 'data/output_chat_pairs.txt')