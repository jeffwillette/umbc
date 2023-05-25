# https://github.com/amazon-science/efficient-longdoc-classification/blob/main/src/datasets.py
import json
import logging
import os

import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from transformers import BertTokenizer

logging.getLogger(
    "transformers.tokenization_utils_base").setLevel(logging.ERROR)



def parse_json_column(genre_data):
    """
    Read genre information as a json string and convert it to a dict
    :param genre_data: genre data to be converted
    :return: dict of genre names
    """
    try:
        return json.loads(genre_data)
    except Exception as e:
        return None  # when genre information is missing


def load_booksummaries_data(book_path):
    """
    Load the Book Summary data and split it into train/dev/test sets
    :param book_path: path to the booksummaries.txt file
    :return: train, dev, test as pandas data frames
    """
    book_df = pd.read_csv(book_path, sep='\t', names=["Wikipedia article ID",
                                                      "Freebase ID",
                                                      "Book title",
                                                      "Author",
                                                      "Publication date",
                                                      "genres",
                                                      "summary"],
                          converters={'genres': parse_json_column})
    # remove rows missing any genres or summaries
    book_df = book_df.dropna(subset=['genres', 'summary'])
    book_df['word_count'] = book_df['summary'].str.split().str.len()
    book_df = book_df[book_df['word_count'] >= 10]
    train = book_df.sample(frac=0.8, random_state=22)
    rest = book_df.drop(train.index)
    dev = rest.sample(frac=0.5, random_state=22)
    test = rest.drop(dev.index)
    return train, dev, test


def prepare_book_summaries(book_path='data/booksummaries/booksummaries.txt'):
    """
    Load the Book Summary data and prepare the datasets
    :param pairs: whether to combine pairs of documents or not
    :param book_path: path to the booksummaries.txt file
    :return: dicts of lists of documents and labels and number of labels
    """
    if not os.path.exists(book_path):
        raise Exception("Data not found: {}".format(book_path))

    text_set = {'train': [], 'dev': [], 'test': []}
    label_set = {'train': [], 'dev': [], 'test': []}
    train, dev, test = load_booksummaries_data(book_path)

    train_temp = train['summary'].tolist()
    dev_temp = dev['summary'].tolist()
    test_temp = test['summary'].tolist()

    train_genres = train['genres'].tolist()
    train_genres_temp = [list(genre.values()) for genre in train_genres]
    dev_genres = dev['genres'].tolist()
    dev_genres_temp = [list(genre.values()) for genre in dev_genres]
    test_genres = test['genres'].tolist()
    test_genres_temp = [list(genre.values()) for genre in test_genres]

    for i in range(0, len(train_temp) - 1, 2):
        text_set['train'].append(train_temp[i] + train_temp[i+1])
        label_set['train'].append(
            list(set(train_genres_temp[i] + train_genres_temp[i+1])))

    for i in range(0, len(dev_temp) - 1, 2):
        text_set['dev'].append(dev_temp[i] + dev_temp[i+1])
        label_set['dev'].append(
            list(set(dev_genres_temp[i] + dev_genres_temp[i+1])))

    for i in range(0, len(test_temp) - 1, 2):
        text_set['test'].append(test_temp[i] + test_temp[i+1])
        label_set['test'].append(
            list(set(test_genres_temp[i] + test_genres_temp[i+1])))

    vectorized_labels, num_labels = vectorize_labels(label_set)
    return text_set, vectorized_labels, num_labels


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


class CMUBookSummaryDataset(Dataset):
    def __init__(self, root: str, split: str = "train", num_points: int = -1, shuffle: bool = False):
        super().__init__()
        book_path = os.path.join(root, "booksummaries", "booksummaries.txt")
        self.name = "cmu_book_summaries"
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        text_set, label_set, num_labels = prepare_book_summaries(book_path)
        self.text, self.labels = text_set[split], label_set[split]
        self.num_labels = num_labels
        self.shuffle = shuffle
        self.num_points = num_points

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
    train = CMUBookSummaryDataset("../data/booksummaries", split="train")
    val = CMUBookSummaryDataset("../data/booksummaries", split="dev")
    test = CMUBookSummaryDataset("../data/booksummaries", split="test")
    print(len(train), len(val), len(test))
