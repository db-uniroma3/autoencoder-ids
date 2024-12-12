import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Funzione per leggere il dataset
def read_dataset(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

# Funzione per definire i tipi di colonne
def define_columns_types():
    numeric_columns = ['auth_success', 'auth_attempts']
    categorical_columns = ['dest_port', 'status_guess', 'version', 'kex_alg',
                           'mac_alg', 'host_key_alg', 'cipher_alg', 'client', 'direction']
    return numeric_columns, categorical_columns

# Funzione per rimuovere colonne non necessarie
def remove_columns(dataset, numeric_columns, categorical_columns):
    for column in dataset.columns:
        if column not in numeric_columns and column not in categorical_columns and column != 'label':
            dataset.drop(column, axis=1, inplace=True)
    return dataset

# Funzione per sostituire valori nulli
def replace_null_values(dataset, numeric_columns, categorical_columns):
    numeric_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    dataset[numeric_columns] = numeric_imputer.fit_transform(dataset[numeric_columns])
    dataset[categorical_columns] = categorical_imputer.fit_transform(dataset[categorical_columns])
    return dataset

# Funzione per normalizzare colonne numeriche
def normalize_numeric_columns(dataset, numeric_columns):
    scaler = MinMaxScaler()
    dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])
    return dataset

# Funzione per sostituire valori categorici con i top N valori
def top_n(dataset, categorical_columns):
    top_val = 10
    for col in categorical_columns:
        value_counts = dataset[col].value_counts()
        top_n_categories = value_counts.index[:top_val].tolist()
        dataset[col] = dataset[col].where(dataset[col].isin(top_n_categories), other='Other')
    return dataset

# Funzione per codificare colonne categoriche
def categorical_data_encoding(dataset, categorical_columns):
    encoded_dataset = pd.get_dummies(dataset[categorical_columns])
    dataset = dataset.drop(columns=categorical_columns, axis=1)
    dataset = pd.concat([dataset, encoded_dataset], axis=1)
    return dataset

# Funzione per allineare le colonne tra train e test
def align_columns(train, test):
    missing_in_test = set(train.columns) - set(test.columns)
    missing_in_train = set(test.columns) - set(train.columns) 

    for col in missing_in_test:  # Aggiungi le colonne mancanti in test e riempi con False
        test[col] = False
    
    for col in missing_in_train:  # Aggiungi le colonne mancanti in train e riempi con False
        train[col] = False

    # Riordina le colonne in entrambi i dataset per avere la stessa struttura
    train = train.reindex(sorted(train.columns), axis=1)
    test = test.reindex(sorted(test.columns), axis=1)

    return train, test

# Funzione per salvare i dati di test
def save_test(df, output_prefix):
    # Separazione delle feature (X) e delle label (y)
    y = df['label']
    x = df.drop('label', axis=1)

    # Conversione dei dati in float32
    x = x.astype('float32')
    y = y.astype('float32')

    # Salvataggio dei dati in file CSV
    x.to_csv(f'X_{output_prefix}.csv', index=False)
    y.to_csv(f'y_{output_prefix}.csv', index=False)

    return x, y

# Funzione per salvare i dati di train
def save_train(df, output_prefix):
    data_label_0 = df[df['label'] == 0]

    x = data_label_0.drop('label', axis=1)

    # Conversione dei dati in float32
    x = x.astype('float32')

    # Salvataggio dei dati in file CSV
    x.to_csv(f'X_{output_prefix}.csv', index=False)

    return x

# Main per eseguire il preprocessing
def main():
    # Percorsi dei file di input
    train_file_path = 'train_ssh_raw.csv'
    test_file_path = 'test_ssh_raw.csv'

    # Lettura dei dataset
    print("Caricamento dei dataset...")
    df_train = read_dataset(train_file_path)
    df_test = read_dataset(test_file_path)

    # Definizione dei tipi di colonne
    print("Definizione dei tipi di colonne...")
    numeric_columns, categorical_columns = define_columns_types()

    # Preprocessing per il dataset di train
    print("Preprocessing del dataset di train...")
    df_train = remove_columns(df_train, numeric_columns, categorical_columns)
    df_train = replace_null_values(df_train, numeric_columns, categorical_columns)
    df_train = normalize_numeric_columns(df_train, numeric_columns)
    df_train = top_n(df_train, categorical_columns)
    df_train = categorical_data_encoding(df_train, categorical_columns)

    # Preprocessing per il dataset di test
    print("Preprocessing del dataset di test...")
    df_test = remove_columns(df_test, numeric_columns, categorical_columns)
    df_test = replace_null_values(df_test, numeric_columns, categorical_columns)
    df_test = normalize_numeric_columns(df_test, numeric_columns)
    df_test = top_n(df_test, categorical_columns)
    df_test = categorical_data_encoding(df_test, categorical_columns)

    # Allineamento delle colonne tra train e test
    print("Allineamento delle colonne tra train e test...")
    df_train, df_test = align_columns(df_train, df_test)

    # Salvataggio dei dataset preprocessati
    print("Salvataggio dei dataset preprocessati...")
    save_train(df_train, 'train')
    save_test(df_test, 'test')

    print("Preprocessing completato e file salvati!")

# Esegui il main
if __name__ == "__main__":
    main()
