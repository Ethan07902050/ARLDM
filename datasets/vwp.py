import random
import json
import cv2
import h5py
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

# import hydra
# import sys
# from tqdm import tqdm
# from omegaconf import DictConfig
# from torch.utils.data import DataLoader
# sys.path.append('../')

from models.blip_override.blip import init_tokenizer

class StoryDataset(Dataset):
    """
    A custom subset class for the LRW (includes train, val, test) subset
    """

    def __init__(self, subset, args):
        super(StoryDataset, self).__init__()
        self.args = args
        self.story_len = 5

        self.data_root = Path(args.get(args.dataset).data_root)
        self.subset = subset
        
        metadata_path = self.data_root / 'metadata_arldm.json'
        self.samples = json.load(open(metadata_path))[subset]

        self.augment = transforms.Compose([
            transforms.CenterCrop(240),
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.sample_transform = transforms.Compose([
            transforms.CenterCrop(240),
            transforms.ToTensor()
        ])

        self.max_length = args.get(args.dataset).max_length
        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
        self.blip_tokenizer = init_tokenizer()

        self.blip_image_processor = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])

    def __getitem__(self, index):
        sample = self.samples[index]

        images = list()
        for image in sample['image']:
            image_path = self.data_root / image
            im = Image.open(image_path)
            images.append(im)

        source_images = torch.stack([self.blip_image_processor(im) for im in images])
        images = images[1:] if self.args.task == 'continuation' else images

        if self.subset in ['train', 'val']:
            images = torch.stack([self.augment(im) for im in images])
        else:
            images = torch.stack([self.sample_transform(im) for im in images])

        texts = sample['text']

        # tokenize caption using default tokenizer
        tokenized = self.clip_tokenizer(
            texts[1:] if self.args.task == 'continuation' else texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        captions, attention_mask = tokenized['input_ids'], tokenized['attention_mask']

        tokenized = self.blip_tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        source_caption, source_attention_mask = tokenized['input_ids'], tokenized['attention_mask']
        return images, captions, attention_mask, source_images, source_caption, source_attention_mask

    def __len__(self):
        return len(self.samples)

# @hydra.main(config_path="../", config_name="config")
# def main(args: DictConfig) -> None:
#     train_data = StoryDataset('train', args)
#     trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
#     max_length = 0
#     for batch in tqdm(trainloader, total=len(trainloader)):
#         if batch[1].shape[-1] > max_length:
#             max_length = batch[1].shape[-1]
#     print(max_length)

# if __name__ == '__main__':
#     main()