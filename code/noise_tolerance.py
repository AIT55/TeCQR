import json
import random
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
sbert = SentenceTransformer("/home/xxxxxxxx/TeCQR/prelm/all-MiniLM-L6-v2")
item_dict_path = "/home/xxxxxxxx/TeCQR/data/lastfm/Graph_generate_data/item_dict(num)_new.json"
question_new_num_path = "/home/xxxxxxxx/TeCQR/data/lastfm/Graph_generate_data/question_new_num.txt"
text_tokenized_path = "/home/xxxxxxxx/TeCQR/data/lastfm/Graph_generate_data/text_tokenized.txt"
tag_num_path = "/home/xxxxxxxx/TeCQR/tag_num.txt"
with open(item_dict_path, 'r') as f:
    item_dict = json.load(f)
query_to_old = {}
with open(question_new_num_path, 'r') as f:
    for line in f:
        new_id, old_id = line.strip().split('\t')
        query_to_old[int(new_id)] = int(old_id)
query_texts = {}
with open(text_tokenized_path, 'r') as f:
    for line in f:
        old_id, text, _ = line.strip().split('\t')  
        query_texts[int(old_id)] = text
tag_to_id = {}
id_to_tag = {}
with open(tag_num_path, 'r') as f:
    for line in f:
        tag, tag_id = line.strip().split('\t')
        tag_to_id[tag] = int(tag_id)
        id_to_tag[int(tag_id)] = tag
positive_examples = []
negative_examples = []
for query_id, data in item_dict.items():
    for tag_id in data['feature_index']:
        query_text = query_texts[query_to_old[int(query_id)]]
        tag_text = id_to_tag[tag_id]
        positive_examples.append((query_text, tag_text, 1))
        all_tags = list(tag_to_id.values())
        negative_tag_id = random.choice([t for t in all_tags if t != tag_id])
        negative_tag_text = id_to_tag[negative_tag_id]
        negative_examples.append((query_text, negative_tag_text, 0))
all_examples = positive_examples + negative_examples
def encode_texts(texts):
    return sbert.encode(texts, show_progress_bar=True)
queries, tags, labels = zip(*all_examples)
encoded_queries = encode_texts(queries)
encoded_tags = encode_texts(tags)
train_data = pd.DataFrame({
    'query': list(encoded_queries),
    'tag': list(encoded_tags),
    'label': labels
})
X_train, X_test, y_train, y_test = train_test_split(train_data[['query', 'tag']], train_data['label'], test_size=0.2)
class QueryTagDataset(Dataset):
    def __init__(self, queries, tags, labels):
        self.queries = queries
        self.tags = tags
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        query = torch.tensor(self.queries.iloc[idx].values).float()
        tag = torch.tensor(self.tags.iloc[idx].values).float()
        label = torch.tensor(self.labels.iloc[idx]).float()
        return query, tag, label
train_dataset = QueryTagDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
class noise_tolerance(nn.Module):
    def __init__(self):
        super(noise_tolerance, self).__init__()
        self.fc1 = nn.Linear(384 * 2, 128)  
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, query_embed, tag_embed):
        query_embed = query_embed.unsqueeze(0) if query_embed.dim() == 1 else query_embed
        tag_embed = tag_embed.unsqueeze(0) if tag_embed.dim() == 1 else tag_embed
        x = torch.cat((query_embed, tag_embed), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)
model = noise_tolerance(input_dim=384)  
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for query, tag, label in train_loader:
        optimizer.zero_grad()
        output = model(query, tag)
        loss = criterion(output, label.unsqueeze(1))  
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
model.eval()
