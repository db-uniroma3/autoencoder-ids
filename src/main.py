import argparse
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from autoencoder import load_dataset, create_autoencoder, training, calculate_loss, curva_roc, \
    detect_anomaly, real_labels, metrics, feature_importance
from preparation import label_simulated_attacks_http, read_json_file, label_detected_attacks, \
    label_simulated_attacks_ssh
from preprocessing import read_dataset, define_columns_types_http, remove_columns, modify_host_column, \
    replace_null_values, normalize_numeric_columns, top_n, categorical_data_encoding, align_columns, save_test, \
    save_train, define_columns_types_ssh


def parse_arguments(arguments):
    parser = argparse.ArgumentParser(
        description="IDS-Autoencoder: train and test your model for intrusion detection!",
        prog="IDS-Autoencoder",
        add_help=True,
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    preparation_parser = subparsers.add_parser('prepare')
    group = preparation_parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        '--http',
        action='store_true',
        help="Prepare the data for the http model."
    )

    group.add_argument(
        '--ssh',
        action='store_true',
        help="Prepare the data for the ssh model."
    )

    preparation_parser.add_argument(
        '--raw-data',
        required=True,
        help="The path to the data to prepare."
    )

    preparation_parser.add_argument(
        '--output',
        required=True,
        help="The output path for the prepared data."
    )

    preparation_parser.add_argument(
        '--darktrace-report',
        required=False,
        help="The path to the darktrace report."
    )

    preprocess_parser = subparsers.add_parser('preprocess')

    group = preprocess_parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        '--http',
        help="Preprocess the data for the http model.",
        action='store_true'
    )

    group.add_argument(
        '--ssh',
        action='store_true',
        help="Preprocess the data for the ssh model."
    )

    preprocess_parser.add_argument(
        '--train',
        required=True,
        help="The path to the train set."
    )

    preprocess_parser.add_argument(
        '--test',
        required=True,
        help="The path to the test set."
    )

    preprocess_parser.add_argument(
        '--output',
        required=True,
        help="The output path for the prepared data."
    )

    train_parser = subparsers.add_parser('train')

    train_parser.add_argument(
        '--dataset',
        '-d',
        required=True,
        help="The dataset to train on."
    )

    train_parser.add_argument(
        '--load-model',
        required=False,
        help="The model to load before starting the training."
    )

    train_parser.add_argument(
        '--save-model',
        required=True,
        help="The path in which to save the model after training."
    )

    test_parser = subparsers.add_parser('test')

    test_parser.add_argument(
        '--feature-dataset',
        required=True,
        help="The features' dataset for the test."
    )

    test_parser.add_argument(
        '--label-dataset',
        required=True,
        help="The labels' dataset for the test."
    )

    test_parser.add_argument(
        '--load-model',
        required=True,
        help="The model to load before starting the test."
    )

    test_parser.add_argument(
        '--figure-path',
        required=True,
        help="The path to save figures."
    )

    return parser.parse_args(arguments if arguments else ['--help'])


def prepare_data(raw_data_path: str, output_path: str, darktrace_report_path: str = None, http: bool = False, ssh: bool = False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Labeling data...")
    if http:
        label_simulated_attacks_http(raw_data_path, output_path)
    elif ssh:
        label_simulated_attacks_ssh(raw_data_path, output_path)
    if darktrace_report_path:
        print(f"Darktrace report path: {darktrace_report_path}")
        darktrace_ai_analyst = read_json_file(darktrace_report_path)
        label_detected_attacks(darktrace_ai_analyst, output_path)
    print("Data labeled successfully.")


def preprocess_data(train_path: str, test_path: str, output_path: str, http: bool = False, ssh: bool = False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Loading Dataset...")
    df_train = read_dataset(train_path)
    train_name = os.path.basename(train_path)
    df_test = read_dataset(test_path)
    test_name = os.path.basename(test_path)

    # Definizione dei tipi di colonne
    print("Defining columns types...")
    if http:
        numeric_columns, categorical_columns = define_columns_types_http()
    elif ssh:
        numeric_columns, categorical_columns = define_columns_types_ssh()

    # Preprocessing per il dataset di train
    print("Preprocessing training dataset...")
    df_train = remove_columns(df_train, numeric_columns, categorical_columns)
    if http:
        df_train = modify_host_column(df_train)
    df_train = replace_null_values(df_train, numeric_columns, categorical_columns)
    df_train = normalize_numeric_columns(df_train, numeric_columns)
    df_train = top_n(df_train, categorical_columns)
    df_train = categorical_data_encoding(df_train, categorical_columns)

    # Preprocessing per il dataset di test
    print("Preprocessing test dataset...")
    df_test = remove_columns(df_test, numeric_columns, categorical_columns)
    if http:
        df_test = modify_host_column(df_test)
    df_test = replace_null_values(df_test, numeric_columns, categorical_columns)
    df_test = normalize_numeric_columns(df_test, numeric_columns)
    df_test = top_n(df_test, categorical_columns)
    df_test = categorical_data_encoding(df_test, categorical_columns)

    # Allineamento delle colonne tra train e test
    print("Aligning columns between datasets...")
    df_train, df_test = align_columns(df_train, df_test)

    # Salvataggio dei dataset preprocessati
    print("Saving preprocessed dataset...")
    save_train(df_train, output_path)
    save_test(df_test, output_path)

    print("Preprocessing completed!")


def train_model(dataset_path: str, output_path: str, model_path: str = None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)

    if model_path:
        print("Loading model...")
        import keras
        autoencoder = keras.models.load_model(model_path)
    else:
        print("Creating model...")
        autoencoder = create_autoencoder(dataset)

    training(autoencoder, dataset)
    autoencoder.save(output_path)

    print("Process completed!")


def test_model(feature_dataset_path: str, label_dataset_path: str, model_path: str, figures_path: str):
    # Caricamento dei dataset
    os.makedirs(figures_path, exist_ok=True)
    print("Loading features and labels dataset...")
    feature_dataset = load_dataset(feature_dataset_path)
    label_dataset = load_dataset(label_dataset_path)

    print(f"Loading model from {model_path}...")
    import keras
    autoencoder = keras.models.load_model(model_path)

    # Calcolo della loss e della curva ROC
    print("Computing loss and ROC curve...")
    loss = calculate_loss(autoencoder, feature_dataset)
    threshold = curva_roc(label_dataset, loss, figures_path)

    # Rilevazione delle anomalie
    print("Detecting anomalies...")
    predicted_labels = detect_anomaly(feature_dataset, autoencoder, threshold)
    real_labels_array = real_labels(label_dataset)

    # Calcolo delle metriche
    print("Computing metrics...")
    print(predicted_labels)
    print(real_labels_array)
    metrics(real_labels_array, predicted_labels, figures_path)

    # Importanza delle feature
    print("Computing mean reconstruction error for each feature...")
    feature_importance(autoencoder, feature_dataset, figures_path)

    print("Process completed!")


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])

    if args.subcommand == 'prepare':
        prepare_data(args.raw_data, args.output, args.darktrace_report, args.http, args.ssh)
    elif args.subcommand == 'preprocess':
        preprocess_data(args.train, args.test, args.output, args.http, args.ssh)
    elif args.subcommand == 'train':
        train_model(args.dataset, args.save_model, args.load_model)
    elif args.subcommand == 'test':
        test_model(args.feature_dataset, args.label_dataset, args.load_model, args.figure_path)
