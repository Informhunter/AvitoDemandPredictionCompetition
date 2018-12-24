import pandas as pd
import pickle
from scipy.sparse import load_npz

categorical = ['category_name', 'city', 'image_top_1', 'param_1', 'param_2', 'param_3',
               'parent_category_name', 'region', 'user_type', 'image_is_null', 'user_id']

numerical = ['item_seq_number', 'price', 'title_length_chars', 'description_length_chars',
             'categorical_one_hot', 'ridge_predictions', 'mean_encoded_categorical']

text = ['title_tfidf', 'description_tfidf', 'title_pymorphy_tfidf',
        'description_pymorphy_tfidf', 'title_bof', 'description_bof',
        'title_tfidf_no_2grams', 'description_tfidf_no_2grams',
        'title_tfidf_7000', 'description_tfidf_50000',
        'title_tfidf_7000_pymorphy', 'description_tfidf_50000_pymorphy']

image = ['vgg16_512']

all_features = categorical + numerical + text

def load_feature(category, feature_name, dataset='train'):
    path = '../feature_exploration/{}/{}/{}.npz'.format(category, dataset, feature_name)
    with open(path, 'rb') as f:
        feature = load_npz(f)
    return feature_name, feature


def extract_baseline_features(dfs):
    df = dfs[0] #main df
    feature_names = ['region', 'city', 'parent_category_name', 'category_name',
                     'param_1', 'param_2', 'param_3',
                     'price', 'item_seq_number', 'user_type', 'image_top_1']
    iscategorical = [True, True, True, True,
                     True, True, True,
                     False, False, True, True]
    
    features = []
    for feature_name, iscat in zip(feature_names, iscategorical):
        if feature_name.startswith('param_'):
            feature = df[feature_name].fillna("NA")
        elif feature_name == 'image_top_1':
            feature = df[feature_name].fillna(-1)
        else:
            feature = df[feature_name]
        features.append((feature_name, feature, iscat))
    return features


def extract_date_features(dfs):
    df = dfs[0] #main df
    dates = pd.to_datetime(df['activation_date'])
    return [('activation_dayofweek', dates.dt.dayofweek, True)]

def extract_isnull_features(dfs):
    df = dfs[0] #main df
    
    return [
        ('image_isnotnull', ~pd.isnull(df['image']), True),
        ('price_isnotnull', ~pd.isnull(df['price']), True),
    ]

def get_train_vgg16():
    with open('../feature_exploration/train_img_features_vgg16/all.pkl', 'rb') as f:
        return pickle.load(f)

def get_test_vgg16():
    with open('../feature_exploration/test_img_features_vgg16/all.pkl', 'rb') as f:
        return pickle.load(f)
    

def get_train_vgg16pca100():
    with open('../feature_exploration/train_img_features_vgg16/pca100.pkl', 'rb') as f:
        return pickle.load(f)

def get_test_vgg16pca100():
    with open('../feature_exploration/test_img_features_vgg16/pca100.pkl', 'rb') as f:
        return pickle.load(f)


def extract_train_vgg16(dfs):
    df = dfs[0]
    vggtrain = get_train_vgg16()
    vggtrain = pd.merge(df, vggtrain, 'left', on='image', copy=False, sort=False)
    vggtrain = vggtrain[list(range(512))]
    vggtrain.fillna(0, inplace=True)
    return [('train_vgg16', vggtrain.as_matrix(), False)]

def extract_test_vgg16(dfs):
    df = dfs[0]
    vggtest = get_test_vgg16()
    vggtest = pd.merge(df, vggtrain, 'left', on='image', copy=False, sort=False)
    vggtest = vggtest[list(range(512))]
    vggtest.fillna(0, inplace=True)
    return [('test_vgg16', vggtest.as_matrix(), False)]

def extract_train_vgg16pca100(dfs):
    df = dfs[0]
    vggtrain = get_train_vgg16pca100()
    vggtrain = pd.merge(df, vggtrain, 'left', on='image', copy=False, sort=False)
    vggtrain = vggtrain[list(range(100))]
    vggtrain.fillna(0, inplace=True)
    return [('train_vgg16pca100', vggtest.as_matrix(), False)]

def extract_test_vgg16pca100(dfs):
    df = dfs[0]
    vggtest = get_test_vgg16pca100()
    vggtest = pd.merge(df, vggtest, 'left', on='image', copy=False, sort=False)
    vggtest = vggtest[list(range(100))]
    vggtest.fillna(0, inplace=True)
    return [('train_vgg16pca100', vggtest.as_matrix(), False)]

extractors = {
    'baseline': extract_baseline_features,
    'date_features': extract_date_features, 
    'isnull_features': extract_isnull_features,
    'train_vgg16': extract_train_vgg16,
    'test_vgg16': extract_test_vgg16,
    'train_vgg16pca100': extract_train_vgg16pca100,
    'test_vgg16pca100': extract_test_vgg16pca100,
}

def extract_features(dfs, names=['all']):
    features = [] # list of tuples (Series or np.array, boolean)
    if 'all' in names:
        for _, extractor in extractors.items():
            features += extractor(dfs)
    else:
        for name in names:
            features += extractors[name](dfs)
    return list(zip(*features))


def load_features(dataset='train', names=['all']):
    if 'all' in names:
        names = all_features
    
    features = []
    categorical_indices = []
    
    current_index = 0
    
    for feature_name in names:
        if feature_name in categorical:
            _, feature = load_feature('cat_features', feature_name, dataset=dataset)
            categorical_indices += list(range(current_index, current_index + feature.shape[1]))
        elif feature_name in numerical:
            _, feature = load_feature('num_features', feature_name, dataset=dataset)
        elif feature_name in text:
            _, feature = load_feature('text_features', feature_name, dataset=dataset)
        elif feature_name in image:
            _, feature = load_feature('img_features', feature_name, dataset=dataset)
        else:
            raise Exception('Wrong feature name: ' + feature_name)
        current_index += feature.shape[1]
        features.append(feature)
    return names, features, categorical_indices