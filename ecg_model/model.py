from data import load_data
import datasets
from sklearn.model_selection import train_test_split
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from ecg_model.params import *
# from preprocessor import preprocess
import torch
from transformers import  AutoModelForImageClassification, TrainingArguments, Trainer


def split_dataset():
    """
    Load data from the files, split the dataset into training, evaluation and test,
    preprocess the data
    """
    labels_csv="scp_codes.csv"
    MODEL_NAME = os.environ.get("MODEL_NAME")

    images, labels = load_data(labels_csv)
    train_X, hold_X, train_y, hold_y = train_test_split(images, labels, test_size=0.2)
    eval_X, test_X, eval_y, test_y = train_test_split(hold_X, hold_y, test_size=0.5)

    train_dataset = datasets.Dataset.from_dict({"image": train_X, "label": train_y})
    eval_dataset = datasets.Dataset.from_dict({"image": eval_X, "label": eval_y})
    test_dataset = datasets.Dataset.from_dict({"image": test_X, "label": test_y})

    # train_dataset = preprocess(train_dataset)
    # eval_dataset = preprocess(eval_dataset)
    # test_dataset = preprocess(test_dataset)

    # train_dataset.set_transform(preprocess)
    # eval_dataset.set_transform(preprocess)
    # test_dataset.set_transform(preprocess)

    print(train_dataset)
    return train_dataset, eval_dataset, test_dataset

def initiate_model():
    MODEL_NAME = os.environ.get("MODEL_NAME")

    model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    label2id = {'Normal': 1, 'Abnormal': 0},
    id2label = {'1': 'Normal', '0': 'Abnormal'},
    ignore_mismatched_sizes = True)

    print("Model initiated")

    return model

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    metric = datasets.load_metric("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def evaluate_model(test_dataset):
    return trainer.evaluate(test_dataset)

def train_model(model):
    train_dataset, eval_dataset, test_dataset = split_dataset()
    MODEL_NAME = os.environ.get("MODEL_NAME")

    training_args = TrainingArguments(
        output_dir="./output",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
        )

    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=extractor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    trainer.train()
    evaluation = evaluate_model(test_dataset)

    return trainer, evaluation


if __name__ == '__main__':
    model = initiate_model()
    train_model(model)
    # split_dataset()
