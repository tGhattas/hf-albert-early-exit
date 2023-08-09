from albertee import AlbertForSequenceClassificationEarlyExit
import torch
from transformers import AlbertForSequenceClassification, AlbertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import json


def run():

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

        encoded_dataset = dataset.map(encode, batched=True)

        return encoded_dataset, num_labels, validation_set_name

    def model_init():
        model = AlbertForSequenceClassificationEarlyExit.from_pretrained(model_name, num_labels=num_labels)
        if hidden_layers is not None:
            model.config.num_hidden_layers = hidden_layers
        return model

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = np.mean(predictions == labels)
        return {'accuracy': accuracy}

    def plot_loss(model_dir):
        try:
            logs = []
            with open(f"{model_dir}/trainer_state.json", 'r') as f:
                for line in f:
                    logs.append(json.loads(line))

            losses = [log['loss'] for log in logs]
            plt.plot(losses)
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.show()
        except FileNotFoundError:
            print(f"No trainer_state.json found in {model_dir}. Cannot plot the loss.")

    def save_model(model, training_args):
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    def load_model_and_evaluate(model_dir, dataset, validation_set_name):
        model = AlbertForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AlbertTokenizer.from_pretrained(model_dir)

        # Print the model's configuration
        print(model.config)

        # Plot the training loss if possible
        if os.path.isfile(f"{model_dir}/trainer_state.json"):
            plot_loss(model_dir)

        # Create a new trainer with the loaded model and tokenizer
        dataset['train'] = dataset['train'].filter(lambda example: example['label'] >= 0)
        dataset[validation_set_name] = dataset[validation_set_name].filter(lambda example: example['label'] >= 0)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset[validation_set_name],
            compute_metrics=compute_metrics,
            learning_rate=2e-5,
        )

        # Evaluate the model
        eval_results = trainer.evaluate()

        return model, tokenizer, eval_results

    def save_eval_results(results, model_dir):
        with open(f"{model_dir}/eval_results.json", 'w') as f:
            json.dump(results, f)

    # ALBERT model name (can be "albert-base-v2", "albert-large-v2", "albert-xlarge-v2", "albert-xxlarge-v2")

    # Dataset name. Can be 'snli'/3, 'multi_nli'/3, 'sst2'.
    model_names_to_hidden_layers_num = {
        "sst2": {
            "albert-base-v2": [16],
        },
    }

    buffer = []

    # run on combinations
    for dataset_name in model_names_to_hidden_layers_num:
        for model_name in model_names_to_hidden_layers_num[dataset_name]:
            for hidden_layers in model_names_to_hidden_layers_num[dataset_name][model_name]:

                tokenizer = AlbertTokenizer.from_pretrained(model_name)
                dataset, num_labels, validation_set_name = get_dataset(dataset_name)

                save_directory = f"outputs/ALBERT_{hidden_layers}_{model_name}_{dataset_name}"

                training_args = TrainingArguments(
                    output_dir=save_directory,
                    num_train_epochs=2,
                    per_device_train_batch_size=16 if dataset_name != "multi_nli" else 8,
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

                # Plot the training loss if possible
                if os.path.isfile(f"{save_directory}/trainer_state.json"):
                    plot_loss(save_directory)

                # Evaluate the model
                eval_results = trainer.evaluate()

                # Save evaluation results
                save_eval_results(eval_results, save_directory)

                # To load a saved model
                # model, tokenizer, eval_results = load_model_and_evaluate(save_directory, dataset, validation_set_name)

                buffer += ['-' * 30 + f'{hidden_layers}_{model_name}_{dataset_name}\\n' + str(eval_results) + '\\n']
                print(buffer)


if __name__ == '__main__':
    run()
