from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast
import pandas as pd
from data.dataset import PretrainDataset, FinetuneDataset


def load_data(config):
    print(f'{"="*30}{"DATA":^20}{"="*30}')

    df_train = pd.read_pickle(config['paths']['train_data'])
    df_val = pd.read_pickle(config['paths']['val_data'])

    tokenizer = RobertaTokenizerFast.from_pretrained(
        config['paths']['tokenizer'],
        max_len=config['network']['max_len']
    )

    if config['stage'] == 'pretrain':
        train_dataset = PretrainDataset(
            texts=df_train['text'].values,
            tokenizer=tokenizer,
            max_length=tokenizer.model_max_length
        )
        val_dataset = PretrainDataset(
            texts=df_val['text'].values,
            tokenizer=tokenizer,
            max_length=tokenizer.model_max_length
        )
    elif config['stage'] == 'finetune':
        train_dataset = FinetuneDataset(
            texts=df_train['text'].values,
            targets=df_train['target'].values,
            tokenizer=tokenizer,
            max_length=tokenizer.model_max_length
        )
        val_dataset = FinetuneDataset(
            texts=df_val['text'].values,
            targets=df_val['target'].values,
            tokenizer=tokenizer,
            max_length=tokenizer.model_max_length
        )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    config['train_len'] = len(train_data_loader)

    print('Batch size: ', config['batch_size'])

    print('Train dataset samples: ', len(train_dataset))
    print('Validation dataset samples: ', len(val_dataset))

    print('Train dataset batches: ', len(train_data_loader))
    print('Validataion dataset batches: ', len(val_data_loader))

    print()

    return train_data_loader, val_data_loader
