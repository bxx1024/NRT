import pandas as pd
import os
import gzip
import demjson
import sys
sys.path.append('./')
from Constants import TPS_DIR, META_DIR, CATEGORY


def meta_file_reader(TP_file):
    f= gzip.open(TP_file, 'r')
    items_id, categories, description, title =[], [], [], []

    for line in f:
        js = demjson.decode(line)

        if str(js['asin']) == 'unknown':
            continue

        try:
            title.append(str(js['title']))
        except:
            continue

        try:
            categories.append(js['categories'])
        except:
            categories.append('')

        try:
            description.append(js['description'])
        except:
            description.append('')


        items_id.append(str(js['asin']))

    meta_data = pd.DataFrame({ 'item_id': pd.Series(items_id),
                               'categories': pd.Series(categories),
                               'description': pd.Series(description),
                               'title': pd.Series(title)})[['item_id', 'categories', 'description', 'title']]
    print('reading finished...')
    return meta_data


if __name__ == "__main__":
    TP_file = os.path.join(TPS_DIR, META_DIR)
    meta_data = meta_file_reader(TP_file)
    meta_data.to_csv(os.path.join(TPS_DIR, CATEGORY + '_infor.csv'), index=False, header = ['item_id', 'categories', 'description', 'title'])

