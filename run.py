# %%
from recbole.quick_start import run_recbole
from pathlib import Path
import wandb
from random import random


# %%
import logging
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color


# %%
config_path = Path('Config')
data_path = Path('Dataset_dkt')


# %%
# result_dict = run_recbole(
#     model='RecVAE', dataset='train', 
#     config_file_list=[config_path / 'config.yaml', config_path / 'base.yaml'], 
#     saved=True
#     )

# %%
# model은 다음 링크를 참고해서 적용하기
# https://recbole.io/docs/recbole/recbole.model.html
model='RecVAE'
dataset='train'
config_file_list=[config_path / 'config.yaml', config_path / 'RecVAE.yaml']
config_dict=None
saved=True

# %%
# configurations initialization
config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
init_seed(config['seed'], config['reproducibility'])
# logger initialization
init_logger(config)
logger = getLogger()

logger.info(config)

# %%
if not config.wandb_name:
    config.wandb_name = f'{model}_{dataset}_{int(random()*1000)}'

wandb.init(
    entity=config.wandb_team,
    project=config.wandb_project,
    config=config,
    name=config.wandb_name
)

# dataset filtering
dataset = create_dataset(config)
logger.info(dataset)

# dataset splitting
train_data, valid_data, test_data = data_preparation(config, dataset)

# model loading and initialization
init_seed(config['seed'], config['reproducibility'])
model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
logger.info(model)

# trainer loading and initialization
trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

# model training
best_valid_score, best_valid_result = trainer.fit(
    train_data, valid_data, saved=saved, show_progress=config['show_progress']
)

# model evaluation
test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
logger.info(set_color('test result', 'yellow') + f': {test_result}')
