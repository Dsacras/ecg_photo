from transformers import AutoFeatureExtractor
from ecg_model.params import *
from torchvision.transforms import Compose, Normalize, ToTensor, Resize


def preprocess(dataset):
    MODEL_NAME = os.environ.get("MODEL_NAME")

    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    normalize = Normalize(mean=extractor.image_mean, std=extractor.image_std)
    resize = Resize((extractor.size['shortest_edge'], extractor.size['shortest_edge']))

    transform = Compose([resize, ToTensor(), normalize])

    def transform(example):
        example["pixel_values"] = [transform(image.convert('RGB')) for image in example["image"]]
        return example

    dataset.set_transform(transform)
    return transform(dataset)
