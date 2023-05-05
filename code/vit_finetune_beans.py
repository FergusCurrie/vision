from transformers import (
    BertForSequenceClassification,
    ViTImageProcessor,
    ViTFeatureExtractor,
    ViTForImageClassification,
    TFViTModel,
    ViTConfig,
    Trainer,
)
from PIL import Image
import requests
import numpy as np
import tensorflow as tf
from datasets import load_dataset
import numpy as np
from datasets import load_metric
import torch
from transformers import ViTForImageClassification
from transformers import TrainingArguments
from torchsummary import summary

"""
All models seem to have:
Config
FeatureExtractor # gettting fazed out 
ImageProcessor # preprocessing, postprocessing ... resize, norm etc  . deal with raw iamges
Model # does this have preprocesing 
ForMaskedImageModeling
FOrImageClassification
TFModel

In summary, an image processor is responsible for preprocessing raw images, while a feature extractor is responsible for extracting high-level features from preprocessed images. 

following : https://huggingface.co/blog/fine-tune-vit

"""


def process_example(example):
    inputs = feature_extractor(example["image"], return_tensors="pt")
    inputs["labels"] = example["labels"]
    return inputs


def transform(example_batch):
    """
    While you could call ds.map and apply this to every example at once, this can be very slow,
    especially if you use a larger dataset. Instead, you can apply a transform to the dataset.
    Transforms are only applied to examples as you index them.
    """
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch["image"]], return_tensors="pt")

    # Don't forget to include the labels!
    inputs["labels"] = example_batch["labels"]
    return inputs


def collate_fn(batch):
    return {"pixel_values": torch.stack([x["pixel_values"] for x in batch]), "labels": torch.tensor([x["labels"] for x in batch])}


def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


ds = load_dataset("beans")
example = ds["train"][400]
image = example["image"]
labels = ds["train"].features["labels"]


model_name_or_path = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
res = feature_extractor(image, return_tensors="pt")

# VitModel = TFViTModel.from_pretrained(model_name_or_path)
# VitModel = TFViTModel(ViTConfig()).build()

from transformers import ViTModel

model_name = "google/vit-base-patch16-224"
model = ViTModel.from_pretrained(model_name)
# print(model.summary())

res = process_example(ds["train"][0])


prepared_ds = ds.with_transform(transform)
example_batch = prepared_ds["train"][0:2]


metric = load_metric("accuracy")


labels = ds["train"].features["labels"].names
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),  # model creates classification head with correct number of classes
    id2label={str(i): c for i, c in enumerate(labels)},  # mapping between human readable labels and model's internal labels
    label2id={c: str(i) for i, c in enumerate(labels)},
)

training_args = TrainingArguments(
    output_dir="./vit-base-beans",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="tensorboard",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=feature_extractor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_ds["validation"])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
