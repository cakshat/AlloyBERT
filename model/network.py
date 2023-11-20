import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaModel, get_scheduler, logging

logging.set_verbosity_error()


class AlloyBERT(torch.nn.Module):
    def __init__(self, config, model):
        super(AlloyBERT, self).__init__()

        self.roberta = model
        self.head = torch.nn.Sequential(
            torch.nn.Linear(config['network']['hidden_size'], 1)
        )

    def forward(self, inputs, attention_mask):
        output = self.roberta(inputs, attention_mask=attention_mask)

        return self.head(output.pooler_output)


def create_model(config):
    roberta_config = RobertaConfig(
        max_position_embeddings=config['network']['max_position_embeddings'],
        hidden_size=config['network']['hidden_size'],
        num_attention_heads=config['network']['attn_heads'],
        num_hidden_layers=config['network']['hidden_layers'],
        type_vocab_size=1,
        hidden_dropout_prob=config['network']['drp'],
        attention_probs_dropout_prob=config['network']['attn_drp']
    )

    if config['stage'] == 'pretrain':
        model = RobertaForMaskedLM(roberta_config).to(config['device'])
        config['network']['max_len'] = model.embeddings.position_embeddings.num_embeddings-2
    elif config['stage'] == 'finetune':
        model = RobertaModel.from_pretrained(
            'roberta-base', 
            config=roberta_config, 
            ignore_mismatched_sizes=True
        )
        config['network']['max_len'] = model.embeddings.position_embeddings.num_embeddings-2
        model = AlloyBERT(config, model).to(config['device'])

    return model


def cri_opt_sch(config, model):
    if config['stage'] == 'pretrain':
        criterion = None
    elif config['stage'] == 'finetune':
        criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optim']['lr'])

    scheduler = get_scheduler(config['sch']['name'],
        optimizer=optimizer,
        num_warmup_steps=config['sch']['warmup_steps'],
        num_training_steps=int(config['train_len'] / config['batch_size'] * config['epochs'])    
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     factor=config['sch']['factor'],
    #     patience=config['sch']['patience']
    # )

    return criterion, optimizer, scheduler
