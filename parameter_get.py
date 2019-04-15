
from DataProcess.DataProcess import batches_train, batches_test
from DataProcess.DataProcess import para as para_dict
from Constants import TPS_DIR, CATEGORY_NUM

# review process
print('review preprocess done')


para_dict['category_num'] = CATEGORY_NUM
para_dict['embedding_title_size'] = 300
para_dict['embedding_tip_size'] = 300
para_dict['embedding_id_size'] = 300
para_dict['review_len'] = 1608
para_dict['n_latent'] = 400  # paper
para_dict['hidden_dimension'] = 128

para_dict['learning_rate'] = 0.001
para_dict['learning_rate_decay'] = 0.5
para_dict['lambda_l2'] = 1e-4

para_dict['batch_size'] = 400
para_dict['num_epochs'] = 50
para_dict['save_path'] = TPS_DIR + '/save_path/'

