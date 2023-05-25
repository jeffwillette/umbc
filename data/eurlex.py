import glob
import json
import logging
import os

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

logging.getLogger(
    "transformers.tokenization_utils_base").setLevel(logging.ERROR)


def vectorize_labels(all_labels):
    """
    Combine labels across all data and reformat the labels e.g. [[1, 2], ..., [123, 343, 4] ] --> [[0, 1, 1, ... 0], ...]
    Only used for multi-label classification
    :param all_labels: dict with labels with keys 'train', 'dev', 'test'
    :return: dict of vectorized labels per split and total number of labels
    """
    all_set = []
    for split in all_labels:
        for labels in all_labels[split]:
            all_set.extend(labels)
    all_set = list(set(all_set))

    mlb = MultiLabelBinarizer()
    mlb.fit([all_set])
    num_labels = len(mlb.classes_)

    print(f'Total number of labels: {num_labels}')

    result = {}
    for split in all_labels:
        result[split] = mlb.transform(all_labels[split])

    return result, num_labels


def prepare_eurlex_data(inverted=True, eur_path='./data/EURLEX57K'):
    """
    Load EURLEX-57K dataset and prepare the datasets
    :param inverted: whether to invert the section order or not
    :param eur_path: path to the EURLEX files
    :return: dicts of lists of documents and labels and number of labels
    """
    if not os.path.exists(eur_path):
        raise Exception("Data path not found: {}".format(eur_path))

    text_set = {'train': [], 'dev': [], 'test': []}
    label_set = {'train': [], 'dev': [], 'test': []}

    for split in ['train', 'dev', 'test']:
        file_paths = glob.glob(os.path.join(eur_path, split, '*.json'))
        for file_path in tqdm(sorted(file_paths), leave=False):
            text, tags = read_eurlex_file(file_path, inverted)
            text_set[split].append(text)
            label_set[split].append(tags)

    vectorized_labels, num_labels = vectorize_labels(label_set)

    return text_set, vectorized_labels, num_labels


def read_eurlex_file(eur_file_path, inverted):
    """
    Read each json file and return lists of documents and labels
    :param eur_file_path: path to a json file
    :param inverted: whether to invert the section order or not
    :return: list of documents and labels
    """
    tags = []
    with open(eur_file_path) as file:
        data = json.load(file)
    sections = []
    text = ''
    if inverted:
        sections.extend(data['main_body'])
        sections.append(data['recitals'])
        sections.append(data['header'])

    else:
        sections.append(data['header'])
        sections.append(data['recitals'])
        sections.extend(data['main_body'])

    text = '\n'.join(sections)

    for concept in data['concepts']:
        tags.append(concept)

    return text, tags


class EurLexDataset(Dataset):
    def __init__(self, root: str, split: str = "train", num_points: int = -1, shuffle: bool = False):
        super().__init__()
        self.name = "eurlex_inverted"
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        eur_path = os.path.join(root, "EURLEX57K")
        text_set, label_set, num_labels = prepare_eurlex_data(
            inverted=True, eur_path=eur_path)
        self.text, self.labels = text_set[split], label_set[split]
        self.num_labels = num_labels
        self.shuffle = shuffle
        self.num_points = num_points

        del text_set
        del label_set
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text=text,
            text_pair=None,
            add_special_tokens=False,
            truncation=False,
            return_token_type_ids=False,
        )
        ids = torch.tensor(inputs['input_ids'])
        mask = torch.tensor(inputs['attention_mask'])
        if self.shuffle:
            # permute the order of tokens
            perm_idx = torch.randperm(ids.size(0))
            ids = ids[perm_idx]
            mask = mask[perm_idx]
        # truncate the sentence
        if self.num_points > 0:
            ids = ids[:self.num_points]
            mask = mask[:self.num_points]

        labels = torch.tensor(self.labels[index]).float()

        return ids, mask, labels


if __name__ == "__main__":
    train_dataset = EurLexDataset("../dataset", split="train")
    # val_dataset = EurLexDataset("../dataset", split="dev")
    # test_dataset = EurLexDataset("../dataset", split="test")
    def collate_fn(data):
        def merge(sequences):
            lengths = [seq.size(0) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs

        input_ids, input_mask, labels = zip(*data)
        
        input_ids = merge(input_ids)
        input_mask = merge(input_mask)
        labels = torch.stack(labels, dim=0)

        return input_ids, input_mask, labels
    dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True, pin_memory=True)

    for batch in dataloader:
        print(len(batch))