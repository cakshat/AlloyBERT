import torch


def random_mask(tensor, mask_token):
    rand = torch.rand(tensor.shape)
    mask_indices = rand < 0.15
    tensor[mask_indices] = mask_token

    return tensor


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length): 
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        ids = torch.tensor(encodings['input_ids'], dtype=torch.long)

        return {
            'ids': random_mask(ids.clone(), self.tokenizer.mask_token_id),
            'mask': torch.tensor(encodings['attention_mask'], dtype=torch.long),
            'labels': ids
        }


class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, texts, targets, tokenizer, max_length): 
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        return {
            'ids': torch.tensor(encodings['input_ids'], dtype=torch.long),
            'mask': torch.tensor(encodings['attention_mask'], dtype=torch.long),
            'target': torch.tensor(self.targets[idx], dtype=torch.float)
        }
