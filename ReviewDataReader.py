import os
import gzip
import demjson
import pickle
import numpy as np
import pandas as pd
import sys
import re
sys.path.append('./')
from Constants import TPS_DIR, REVIEW_DIR, CATEGORY
from DataProcess import clean_str


def review_file_reader(TP_file, item_id):
    f= gzip.open(TP_file, 'r')
    users_id, items_id, ratings, summary, reviews=[], [], [], [], []
    for line in f:

        js = demjson.decode(line)
        if str(js['asin']) == 'unknown' or str(js['asin']) not in item_id:
            continue

        if str(js['reviewerID']) == 'unknown':
            continue

        tip = clean_str(js['summary'].strip())
        if not re.search(r'[a-z]+', tip):
            print(js['summary'].strip())
            continue


        users_id.append(str(js['reviewerID']))
        items_id.append(str(js['asin']))
        ratings.append(str(js['overall']))
        summary.append(tip)
        reviews.append(clean_str(js['reviewText'].strip()))


    review_data = pd.DataFrame({ 'user_id': pd.Series(users_id),
                                 'item_id': pd.Series(items_id),
                                 'ratings': pd.Series(ratings),
                                 'summary': pd.Series(summary),
                                 'reviews': pd.Series(reviews),})[['user_id','item_id','ratings','summary', 'reviews']]
    return review_data


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index = False)
    count = playcount_groupbyid.size()
    return count


def numerize(tp):
    uid = map(lambda x: user2id[x], tp['user_id'])
    sid = map(lambda x: item2id[x], tp['item_id'])
    tp['user_id'] = list(uid)
    tp['item_id'] = list(sid)
    return tp

def numerize_meta(tp, item2id):
    sid = []
    for idx, term in enumerate(tp['item_id']):
        if term in item2id.keys():
            sid.append(item2id[term])
        else:
            sid.append(-1)
    tp['item_id'] = sid
    return tp



meta_data = pd.read_csv(os.path.join(TPS_DIR, CATEGORY + '_infor.csv'))
data = review_file_reader(os.path.join(TPS_DIR, REVIEW_DIR), list(meta_data['item_id']))
usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')
unique_uid, unique_sid = usercount.index, itemcount.index
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
data = numerize(data)
meta_data = numerize_meta(meta_data, item2id)
meta_data= meta_data[~meta_data['item_id'].isin([-1])]


n_ratings = data.shape[0]
test = np.random.choice(n_ratings, size = int(0.20 * n_ratings), replace = False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True
tp_test_valid = data[test_idx]
tp_train = data[~test_idx]

n_ratings = tp_test_valid.shape[0]
test = np.random.choice(n_ratings, size = int(0.50 * n_ratings), replace = False)
test_idx = np.zeros(n_ratings, dtype = bool)
test_idx[test] = True
tp_valid = tp_test_valid[~test_idx]
tp_test = tp_test_valid[test_idx]

tp_train.to_csv(os.path.join(TPS_DIR, CATEGORY + '_train.csv'), index = False, header = None)
tp_valid.to_csv(os.path.join(TPS_DIR, CATEGORY + '_valid.csv'), index = False, header = None)
tp_test.to_csv(os.path.join(TPS_DIR, CATEGORY + '_test.csv'), index = False, header = None)
meta_data.to_csv(os.path.join(TPS_DIR, CATEGORY + '_infor_meta.csv'), index=False, header=None)
pickle.dump(user2id, open(os.path.join(TPS_DIR, 'user2id'), 'wb'))
pickle.dump(item2id, open(os.path.join(TPS_DIR, 'item2id'), 'wb'))

