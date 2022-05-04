# %%
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.quick_start import load_data_and_model
from recbole.utils import get_trainer

import pandas as pd

# %%
data_dir = '/opt/ml/workspace/iksad/RecBole/dataset/dkt'
model_file='/opt/ml/workspace/iksad/RecBole/saved/RecVAE-Apr-29-2022_14-11-10.pth'
_, model, _, _, _, _ = load_data_and_model(model_file)

# %%
model_name = model.__class__.__name__
config = Config(model=model_name, dataset="submission", config_file_list=['Config/submission.yaml'])
config['epochs'] = 1

test_dataset = create_dataset(config)
test_dataloader, _, _ = data_preparation(config, test_dataset)
trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

answer_loc = pd.read_csv(f'{data_dir}/answer_loc.txt', header=None)
answer_loc = answer_loc[0].to_list()

# %%
trainer.evaluate(test_dataloader, model_file=model_file)


# %%
name = f'submission_{model_name}'

a_prob = model.predict(test_dataset).tolist()
submission = pd.DataFrame(a_prob).reset_index()
submission.columns=['id', 'prediction']
submission.to_csv(f'submission/{name}.csv', index=False)

# %%
