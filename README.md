# Istruzioni d'uso

Un programma da CLI per la costruzione e l'addestramento di un autoencoder per IDS.

## Creazione dell'immagine Docker

Apri un terminale nella root del progetto e esegui:

```bash
docker build -t autoencoder-ids .
```

## Esempio HTTP

L'immmagine contiene un piccolo dataset per eseguire un esempio. Il dataset iniziale è in formato `.json`, come appena
scaricato da Darktrace.

I dataset necessari sono 3:

1. `train_http_raw.json`: record raw ottenuti da darktrace, utilizzati per il training
2. `train_http_ai.json`: contiene i report di darktrace per lo stesso giorno dei dati raw di training (opzionale,
   richiesto solo se ci sono report in quel giorno).
3. `test_http_raw.json`: record raw ottenuti da darktrace, utilizzati per il test
4. `test_http_ai.json`: contiene i report di darktrace per lo stesso giorno dei dati raw di test (opzionale, richiesto
   solo se ci sono report in quel giorno)

Nel container è presente un alias `autoencoder-ids`, per eseguire il tool da CLI.

### Eseguire il container

```bash
docker run -ti -v $(pwd)/resources:/autoencoder-ids/resources autoencoder-ids bash 
```

### Preparazione dei dati

I dati prima di essere preprocessati, devono essere trasformati in formato csv.

**Trasformazione del training set:**

```bash
autoencoder-ids prepare --http --raw-data resources/raw/http/train_http_raw.json --output resources/prepared/http/train_http_raw.csv
```

**Trasformazione del test set:**

```bash
autoencoder-ids prepare --http --raw-data resources/raw/http/test_http_raw.json --output resources/prepared/http/test_http_raw.csv --darktrace-report resources/raw/http/test_http_ai.json
```

### Preprocessamento dei dati

I dataset di train e test vanno preprocessati insieme:

```bash
autoencoder-ids preprocess --http --train resources/prepared/http/train_http_raw.csv --test resources/prepared/http/test_http_raw.csv --output resources/preprocessed/http/
```

Questo produrrà in uscita (nella cartella `resources/preprocessed/http`) i dataset pronti per l'addestramento e il test . 

### Addestramento del Modello

Ora è possibile eseguire l'addastramento, salvando il relativo modello, con il seguente comando: 
```bash
autoencoder-ids train --dataset resources/preprocessed/http/feature_train.csv --save-model resources/models/http.keras
```

### Testare il Modello

Una volta completato l'addestramento, è possibile verificare le prestazioni del modello sul testset, con il seguente comando:

```bash
autoencoder-ids test --feature-dataset resources/preprocessed/http/feature_test.csv --label-dataset resources/preprocessed/http/label_test.csv --load-model resources/models/http.keras --figure-path resources/figures/
```

### In breve

```bash
autoencoder-ids prepare --http --raw-data resources/raw/http/train_http_raw.json --output resources/prepared/http/train_http_raw.csv
autoencoder-ids prepare --http --raw-data resources/raw/http/test_http_raw.json --output resources/prepared/http/test_http_raw.csv --darktrace-report resources/raw/http/test_http_ai.json
autoencoder-ids preprocess --http --train resources/prepared/http/train_http_raw.csv --test resources/prepared/http/test_http_raw.csv --output resources/preprocessed/http/
autoencoder-ids train --dataset resources/preprocessed/http/feature_train.csv --save-model resources/models/http.keras
autoencoder-ids test --feature-dataset resources/preprocessed/http/feature_test.csv --label-dataset resources/preprocessed/http/label_test.csv --load-model resources/models/http.keras --figure-path resources/figures/
```

## Esempio SSH

Si puo ripetere la stessa procedura per il modello ssh:

```bash
autoencoder-ids prepare --ssh --raw-data resources/raw/ssh/train_ssh_raw.json --output resources/prepared/ssh/train_ssh_raw.csv
autoencoder-ids prepare --ssh --raw-data resources/raw/ssh/test_ssh_raw.json --output resources/prepared/ssh/test_ssh_raw.csv --darktrace-report resources/raw/ssh/test_ssh_ai.json
autoencoder-ids preprocess --ssh --train resources/prepared/ssh/train_ssh_raw.csv --test resources/prepared/ssh/test_ssh_raw.csv --output resources/preprocessed/ssh/
autoencoder-ids train --dataset resources/preprocessed/ssh/feature_train.csv --save-model resources/models/ssh.keras
autoencoder-ids test --feature-dataset resources/preprocessed/ssh/feature_test.csv --label-dataset resources/preprocessed/ssh/label_test.csv --load-model resources/models/ssh.keras --figure-path resources/figures/

```

