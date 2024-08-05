import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaModel, get_scheduler, logging, BertForMaskedLM, BertModel

logging.set_verbosity_error()


class AlloyBERT(torch.nn.Module):
    def __init__(self, config, model):
        super(AlloyBERT, self).__init__()

        self.roberta = model
        self.head = torch.nn.Sequential(
            torch.nn.Linear(model.embeddings.word_embeddings.embedding_dim, 1)
        )

    def forward(self, inputs, attention_mask):
        output = self.roberta(inputs, attention_mask=attention_mask)

        ### BERT ###
        # output = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # output = output.hidden_states[-1].mean(dim=1)
        # return self.head(output)
        ### BERT ###

        return self.head(output.pooler_output)


def create_model(config):
    if config['stage'] == 'pretrain':
        # model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(config['device'])
        model = RobertaForMaskedLM.from_pretrained('roberta-base').to(config['device'])
    elif config['stage'] == 'finetune':
        model = RobertaModel.from_pretrained('roberta-base')
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
        num_training_steps=int(config['train_len'] * config['epochs'])
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     factor=config['sch']['factor'],
    #     patience=config['sch']['patience']
    # )

    return criterion, optimizer, scheduler
