from pprint import pprint

import pandas as pd

from albertee import AlbertForSequenceClassificationEarlyExit
from typing import Union, Optional
from transformers import AlbertForSequenceClassification, AlbertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import json


def run(batch_size: int, minize_dataset: bool = False, dataset_to_run: Optional[str] = None) -> dict:
    def get_number_of_hidden_layers(model: Union[str, AlbertForSequenceClassificationEarlyExit]) -> int:
        if isinstance(model, str):
            model = AlbertForSequenceClassification.from_pretrained(model)
        return model.config.num_hidden_layers

    def get_dataset(dataset_name):
        if dataset_name.lower() not in ['snli', 'multi_nli', 'sst2', 'imdb']:
            raise ValueError(
                f"Invalid dataset name. Expected 'snli', 'multi_nli', 'sst2', or 'imdb', but got {dataset_name}")

        dataset = load_dataset(dataset_name)

        # Handle different validation set naming across datasets
        if dataset_name.lower() in ['snli', 'sst2', 'imdb']:
            validation_set_name = 'validation'
        elif dataset_name.lower() == 'multi_nli':
            validation_set_name = 'validation_matched'  # or 'validation_mismatched'

        if dataset_name.lower() in ['snli', 'multi_nli']:
            dataset['train'] = dataset['train'].select(range(len(dataset['train']) // 3))

            dataset['train'] = dataset['train'].filter(lambda example: example['label'] >= 0)
            dataset[validation_set_name] = dataset[validation_set_name].filter(lambda example: example['label'] >= 0)

            def encode(examples):
                return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length',
                                 max_length=128)

            num_labels = 3
        elif dataset_name.lower() == 'sst2':
            def encode(examples):
                return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)

            num_labels = 2
        elif dataset_name.lower() == 'imdb':
            def encode(examples):
                return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

            num_labels = 2
        if minize_dataset:
            dataset['train'] = dataset['train'].select(range(100))
            dataset[validation_set_name] = dataset[validation_set_name].select(range(100))
        encoded_dataset = dataset.map(encode, batched=True)

        return encoded_dataset, num_labels, validation_set_name

    def model_init():
        model = AlbertForSequenceClassificationEarlyExit.from_pretrained(model_name,
                                                                         num_labels=num_labels,
                                                                         **early_exit_config)
        return model

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = np.mean(predictions == labels)
        return {'accuracy': accuracy}

    def save_model(model, training_args):
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)


    def save_eval_results(results, model_dir):
        with open(f"{model_dir}/eval_results.json", 'w') as f:
            json.dump(results, f)

    # ALBERT model name (can be "albert-base-v2", "albert-large-v2", "albert-xlarge-v2", "albert-xxlarge-v2")

    # Dataset name. Can be 'snli'/3, 'multi_nli'/3, 'sst2'.
    model_names_to_hidden_layers_num = {
        "sst2": {
            # batch size 128/256
            "albert-base-v2": [int(_ * get_number_of_hidden_layers("albert-base-v2")) for _ in [1, 1.5, 2]],
        },
        "snli": {
            "albert-base-v2": [int(_ * get_number_of_hidden_layers("albert-base-v2")) for _ in [1, 1.5, 2]],
        },
        "multi_nli": {
            "albert-base-v2": [int(_ * get_number_of_hidden_layers("albert-base-v2")) for _ in [1, 1.5, 2]],
        },
    }
    exit_thesholds = [0.6, 0.4, 0.2, 0.0]
    buffer = {}

    datasets_to_run = [dataset_to_run] if dataset_to_run is not None else model_names_to_hidden_layers_num.keys()
    # run on combinations
    for weight_name in [
        # 'equal',
        'dyn'
    ]:
        for dataset_name in datasets_to_run:
            for model_name in model_names_to_hidden_layers_num[dataset_name]:
                for hidden_layers in model_names_to_hidden_layers_num[dataset_name][model_name]:
                    for exit_th in exit_thesholds:

                        tokenizer = AlbertTokenizer.from_pretrained(model_name)
                        dataset, num_labels, validation_set_name = get_dataset(dataset_name)

                        save_directory = f"outputs/ALBERT_{hidden_layers}_{model_name}_{weight_name}_{dataset_name}_exit_th_{exit_th}"

                        thres_name = "entropy"  # "bias_1" if exit_th > 0.0 else "entropy"
                        # define early exit config
                        early_exit_config = {
                            "exit_layers_depth": 1,
                            "exit_thres": exit_th,
                            "use_out_pooler": True,
                            "fc_size1": 768,
                            "pooler_input": "cls",
                            "w_init": 4.0,
                            "weight_name": weight_name,
                            "thres_name": thres_name,
                            "cnt_thres": 8,
                            "margin": 0.0,
                            "exits": hidden_layers - 1,
                        }
                        if hidden_layers is not None:
                            early_exit_config['num_hidden_layers'] = hidden_layers

                        training_args = TrainingArguments(
                            output_dir=save_directory,
                            num_train_epochs=2,
                            per_device_train_batch_size=batch_size,
                            per_device_eval_batch_size=64,
                            warmup_steps=500,
                            weight_decay=0.01,
                            logging_dir=f'{save_directory}/logs',
                            save_strategy='steps',  # Save checkpoint every 10000 steps
                            save_steps=100000,
                            learning_rate=2e-5,  # previously 2e-5
                        )

                        trainer = Trainer(
                            model_init=model_init,
                            args=training_args,
                            train_dataset=dataset['train'],
                            eval_dataset=dataset[validation_set_name],
                            compute_metrics=compute_metrics,
                        )

                        # Train the model
                        trainer.train(resume_from_checkpoint=False)

                        # Save the model
                        save_model(trainer.model, training_args)

                        # Evaluate the model
                        trainer.model.set_eval_mode(True)
                        eval_results = trainer.evaluate()

                        # Save evaluation results
                        save_eval_results(eval_results, save_directory)

                        # To load a saved model

                        eval_results['exit_th'] = exit_th
                        buffer[f'{hidden_layers}_{model_name}_{dataset_name}_W_{weight_name}_thn_{thres_name}_exit_th_{exit_th}'] = eval_results
                        pprint(buffer)
    return buffer


def fill_in_dataframes(data: dict[str, dict]) -> dict[str, pd.DataFrame]:
    dataframes = {}

    for key, values in data.items():
        # Extract dataset name from the key
        dataset_name = key.split('_')[2]

        # Create an entry in the dictionary if it doesn't exist
        if dataset_name not in dataframes:
            dataframes[dataset_name] = []

        # Append configuration and values to the respective dataframe list
        dataframes[dataset_name].append({'Configuration': key, **values})

    # Convert lists to pandas dataframes
    for dataset_name, records in dataframes.items():
        dataframes[dataset_name] = pd.DataFrame(records)
    return dataframes


if __name__ == '__main__':
    # minize dataset for quick end to end check
    results = run(batch_size=32, minize_dataset=True)
    print(fill_in_dataframes(results)['sst2'])
