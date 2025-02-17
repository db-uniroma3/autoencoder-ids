import hashlib
import json
import re

import pandas as pd


def label_simulated_attacks_http(path, output_csv_path):  # etichetta i record e filtra quelli richiesta/risposta
    new_dataset = []

    with open(path, 'r') as file:
        for line in file:
            record = json.loads(line.strip())
            if ((record['response_body_len'] > 0 and record['request_body_len'] > 0) or
                    ('response_content_type' in record and 'request_content_type' in record) or
                    ('method' in record and 'status_code' in record)):
                if (record['source_ip'] == '1ccdb898890cce841210e3fb0bcc3e7974f069ca89da96625e7b7699bf277165' or record[
                    'source_ip'] == '398da219038116cf525c6902f7b3b3c4256cbadd8d6eb4f9bf3ff9dbc76f6132') and record[
                    'dest_ip'] == 'f4c36b35451f863e37f34989cca218a6e7c40d22f699aafeef3a6d7ae76a75a2':
                    record['label'] = 1
                else:
                    record['label'] = 0
                new_dataset.append(record)

    # Converte la lista filtrata in un DataFrame
    filtered_df = pd.DataFrame(new_dataset)

    # Salva il DataFrame filtrato in un nuovo file CSV
    filtered_df.to_csv(output_csv_path, index=False)


def label_simulated_attacks_ssh(path, output_csv_path):  # etichetta i record e filtra quelli richiesta/risposta
    new_dataset = []
    with open(path, 'r') as file:
        for line in file:
            record = json.loads(line.strip())
            if record['source_ip'] == '1ccdb898890cce841210e3fb0bcc3e7974f069ca89da96625e7b7699bf277165' and record[
                'dest_ip'] == 'f4c36b35451f863e37f34989cca218a6e7c40d22f699aafeef3a6d7ae76a75a2':
                record['label'] = 1
            else:
                record['label'] = 0
            new_dataset.append(record)

    # Converte la lista filtrata in un DataFrame
    filtered_df = pd.DataFrame(new_dataset)

    # Salva il DataFrame filtrato in un nuovo file CSV
    filtered_df.to_csv(output_csv_path, index=False)


def find_all_ips(data,
                 exclude_ips):  # Funzione per trovare gli indirizzi IP nel JSON senza includere combinazioni IP:Porta e quelli in record["related"]["ip"]
    ip_set = set()
    ip_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')

    def extract_ips(value):
        if isinstance(value, dict):
            for k, v in value.items():
                extract_ips(v)
        elif isinstance(value, list):
            for item in value:
                extract_ips(item)
        elif isinstance(value, str):
            if ip_pattern.fullmatch(
                    value):  # Verifica se il valore è esattamente un indirizzo IP e non parte di un'altra stringa
                ip_set.add(value)

    extract_ips(data)
    ip_set -= exclude_ips  # Rimuove gli IP da escludere
    return ip_set


def read_json_file(file_path):  # Funzione per leggere un file JSON riga per riga
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                try:
                    record = json.loads(line.strip())
                    data.append(record)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()} - {e}")
    return data


def add_ip_to_record(darktrace_ai_analyst):
    for record in darktrace_ai_analyst:  # Aggiornare i record con i nuovi IP
        exclude_ips = set(record.get("related", {}).get("ip", []))
        ips = find_all_ips(record, exclude_ips)
        for ip in ips:
            record["related"]["ip"].append(
                ip)  # aggiungo gli ip trovati nel messaggio testuale per fare il filtro successivamente

    return darktrace_ai_analyst


def calculate_sha256(data):
    if isinstance(data, str):
        data = data.encode()
    sha256_hash = hashlib.sha256(data).hexdigest()
    return sha256_hash


def find_with_dest_ip(timestamp_start, timestamp_end, source_ip, dest_ip, darktrace_row):
    for _, row_record in darktrace_row.iterrows():
        if ((row_record['@timestamp'] >= timestamp_start and
             row_record['@timestamp'] <= timestamp_end) and
                row_record['source_ip'] == calculate_sha256(source_ip) and
                row_record['dest_ip'] == calculate_sha256(dest_ip)):
            row_record['label'] = 1
        else:
            if 'label' not in row_record or pd.isna(row_record['label']):
                row_record['label'] = 0


def find_without_dest_ip(timestamp_start, timestamp_end, source_ip, darktrace_row):
    for _, row_record in darktrace_row.iterrows():
        if ((row_record['@timestamp'] >= timestamp_start and
             row_record['@timestamp'] <= timestamp_end) and
                row_record['source_ip'] == calculate_sha256(source_ip)):
            row_record['label'] = 1
        else:
            if 'label' not in row_record or pd.isna(row_record['label']):
                row_record['label'] = 0


def label_detected_attacks(darktrace_ai_analyst, csv_path):
    darktrace_ai_analyst = add_ip_to_record(darktrace_ai_analyst)
    # Carica il dataset CSV come DataFrame
    darktrace_row = pd.read_csv(csv_path)

    for record in darktrace_ai_analyst:
        source_ip = record["related"]["ip"][0]
        timestamp_start = record["event"]["start"][0][:-5]
        timestamp_end = record["event"]["end"][0][:-5]

        if len(record["related"]["ip"]) > 1:
            for dest_ip in record["related"]["ip"][1:]:
                find_with_dest_ip(timestamp_start, timestamp_end, source_ip, dest_ip, darktrace_row)
        else:
            find_without_dest_ip(timestamp_start, timestamp_end, source_ip, darktrace_row)

    # Salva il DataFrame aggiornato come CSV
    darktrace_row.to_csv(csv_path, index=False)
