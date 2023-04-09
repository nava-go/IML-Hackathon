import itertools
import sys

from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, SGDRegressor, RidgeClassifier, ElasticNet
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression



from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

label_to_ind = {}
ind_to_label = {}
num_of_label = None


def flatten(ls):
    """
    flatten a nested list
    """
    flat_ls = list(itertools.chain.from_iterable(ls))
    return flat_ls


def initilaize_labels(Y):
    global label_to_ind
    global ind_to_label
    global num_of_label
    col_name = Y.columns[0]
    ls_origin_labels = [eval(val) for val in Y[col_name]]
    labs = list(set(flatten(ls_origin_labels)))
    inds = list(range(len(labs)))
    label_to_ind = dict(zip(labs, inds))
    ind_to_label = dict(zip(inds, labs))
    num_of_label = len(labs)

# y is a single response
def convert_y_to_binary_vector(y):
    multi_hot = np.zeros(num_of_label)
    for lab in y:
        cur_ind = label_to_ind[lab]
        multi_hot[cur_ind] = 1
    return multi_hot

def convert_binary_vector_to_y(binary_vector):
    assert (num_of_label == len(binary_vector))
    y = []
    for ind in range(num_of_label):
        if(binary_vector[ind]==1):
            y.append(ind_to_label[ind])
    return y


def load_data_and_spilt(filename_train, filename_labels_0, filename_labels_1):
    train = pd.read_csv(filename_train)
    ind = train.duplicated(subset=['id-hushed_internalpatientid', 'אבחנה-Diagnosis date'])
    train = train[~ind]

    train_labels_0 = pd.read_csv(filename_labels_0)
    initilaize_labels(train_labels_0)
    train_labels_0 = train_labels_0[~ind]

    train_labels_1 = pd.read_csv(filename_labels_1)
    train_labels_1 = train_labels_1[~ind]

    x_train, x_dev, labels_0_train, labels_0_dev,\
    labels_1_train, labels_1_dev = train_test_split(train, train_labels_0,
                                                    train_labels_1,
                                                    test_size=0.25)


    return x_train, x_dev, labels_0_train, labels_0_dev, labels_1_train, labels_1_dev

def clean_reorder_data(x_feature):
    #delete cols
    x_feature.drop(['אבחנה-Histological diagnosis',' Form Name',' Hospital', 'User Name', 'אבחנה-Diagnosis date',
                    'אבחנה-Her2', 'אבחנה-Nodes exam', 'אבחנה-Positive nodes',
                    'אבחנה-Surgery date1', 'אבחנה-Surgery date2',
                    'אבחנה-Surgery date3',
                    'אבחנה-Surgery name1', 'אבחנה-Surgery name2',
                    'אבחנה-Surgery name3',
                    'אבחנה-Tumor depth',
                    'אבחנה-Tumor width', 'אבחנה-er', 'אבחנה-pr',
                    'surgery before or after-Activity date',
                    'surgery before or after-Actual activity',
                    'id-hushed_internalpatientid'
                    ], inplace=True, axis=1)


    #'אבחנה-Age'
    x_feature['אבחנה-Age'] = x_feature['אבחנה-Age'].astype(float)

    #'אבחנה-Basic stage'
    x_feature['אבחנה-Basic stage'].replace({'p - Pathological': 2,
                                            'c - Clinical': 1, 'Null': 0,
                                            'r - Reccurent': 3},
                                  inplace=True)


    # 'אבחנה-Histopatological degree'
    x_feature['אבחנה-Histopatological degree'] = x_feature['אבחנה-Histopatological degree'].replace({'Null':0,
                                                        'GX - Grade cannot be assessed':0,
                                                        'G1 - Well Differentiated':1,
                                                        'G2 - Modereately well differentiated':2,
                                                        'G3 - Poorly differentiated':3, 'G4 - Undifferentiated':4})

    #'אבחנה-Ivi -Lymphovascular invasion'
    x_feature['אבחנה-Ivi -Lymphovascular invasion'].replace({'0': 0, '+':1,
                                                           'extensive':1,
                                                           'yes':1,
                                                           '(+)':1,
                                                           'no': 0,
                                                           '(-)':0,
                                                           'none': 0,
                                                           'No': 0,
                                                           'not': 0,
                                                           '-':0,
                                                           'NO':0,
                                                           'neg':0,
                                                           'MICROPAPILLARY VARIANT':1},
                                                            inplace=True)

    x_feature['אבחנה-Ivi -Lymphovascular invasion'].where(((x_feature['אבחנה-Ivi -Lymphovascular invasion'] ==0 )|
                                                           (x_feature['אבחנה-Ivi -Lymphovascular invasion'] ==1)) ,0,
                                                          inplace=True)

    #'אבחנה-KI67 protein'
    x_feature['אבחנה-KI67 protein'] = x_feature['אבחנה-KI67 protein'].str.extract(r'(\d+)')
    x_feature['אבחנה-KI67 protein'].fillna(0, inplace=True)
    x_feature['אבחנה-KI67 protein'] = x_feature['אבחנה-KI67 protein'].astype('int')
    x_feature['אבחנה-KI67 protein'].where(x_feature['אבחנה-KI67 protein'] <= 100, 100, inplace=True)

    #'אבחנה-Lymphatic penetration'
    x_feature['אבחנה-Lymphatic penetration'].replace({'Null':0,
                                             'L0 - No Evidence of invasion':0,
                                             'LI - Evidence of invasion':1,
                                             'L1 - Evidence of invasion of superficial Lym.':2,
                                             'L2 - Evidence of invasion of depp Lym.':3
                                             }, inplace=True)
    x_feature['אבחנה-Ivi -Lymphovascular invasion'].where(((x_feature['אבחנה-Ivi -Lymphovascular invasion'] == 0) |
                                                           (x_feature['אבחנה-Ivi -Lymphovascular invasion'] == 1) |
                                                           (x_feature['אבחנה-Ivi -Lymphovascular invasion'] == 2) |
                                                           (x_feature['אבחנה-Ivi -Lymphovascular invasion'] == 3) ), 0,
                                                          inplace=True)
    #'אבחנה-M -metastases mark (TNM)'
    x_feature['אבחנה-M -metastases mark (TNM)'] = x_feature['אבחנה-M -metastases mark (TNM)'].str.extract(r'(\d+)')
    x_feature['אבחנה-M -metastases mark (TNM)'].replace({np.nan: 0}, inplace=True)
    x_feature['אבחנה-M -metastases mark (TNM)'] = x_feature['אבחנה-M -metastases mark (TNM)'].astype(int)

    #'אבחנה-Margin Type'
    x_feature["אבחנה-Margin Type"].replace({"נקיים": 0, "ללא": 0, "נגועים": 1}, inplace=True)


    #'אבחנה-N -lymph nodes mark (TNM)'
    x_feature["אבחנה-N -lymph nodes mark (TNM)"] = x_feature["אבחנה-N -lymph nodes mark (TNM)"].str.extract(r'(\d+)')
    x_feature["אבחנה-N -lymph nodes mark (TNM)"].replace({np.nan: 0}, inplace=True)
    x_feature["אבחנה-N -lymph nodes mark (TNM)"] = x_feature["אבחנה-N -lymph nodes mark (TNM)"].astype(int)

    #'אבחנה-Side'
    x_feature["אבחנה-Side"].replace({"שמאל": 1, "ימין": 1, "דו צדדי": 2, np.nan: 0}, inplace=True)
    x_feature["אבחנה-Side"] = x_feature["אבחנה-Side"].astype(int)

    #'אבחנה-Stage'
    x_feature["אבחנה-Stage"] = x_feature["אבחנה-Stage"].str.extract(r'(\d+)')
    x_feature["אבחנה-Stage"].replace({np.nan: 0}, inplace=True)
    x_feature["אבחנה-Stage"] = x_feature["אבחנה-Stage"].astype(int)

    #'אבחנה-Surgery sum'
    x_feature['אבחנה-Surgery sum'] = x_feature['אבחנה-Surgery sum'].astype(float)
    x_feature['אבחנה-Surgery sum'].replace({np.nan: 0}, inplace=True)

    #'אבחנה-T -Tumor mark (TNM)'
    x_feature['אבחנה-T -Tumor mark (TNM)'] = x_feature['אבחנה-T -Tumor mark (TNM)'].str.extract(r'(\d)')
    x_feature['אבחנה-T -Tumor mark (TNM)'].replace({np.nan: 0}, inplace=True)
    x_feature['אבחנה-T -Tumor mark (TNM)'] = x_feature['אבחנה-T -Tumor mark (TNM)'].astype('int')

    return x_feature

def part1_compare_models (x_train, y_train, x_dev, col_name):
    # basic estimator - LogisticRegression
    clf = MultiOutputClassifier(LogisticRegression()).fit(x_train, y_train)
    pred_labels_dev = clf.predict(x_dev)
    pred_labels_dev_str = [[convert_binary_vector_to_y(resp)] for resp in pred_labels_dev]
    df = pd.DataFrame(pred_labels_dev_str)
    df.rename(columns={0: col_name}, inplace=True)
    df.to_csv('pred_labels_dev_str_logistic.csv', index=False)

    # k-NN
    for k in range(1, 10):
        clf = MultiOutputClassifier(KNeighborsClassifier(k)).fit(x_train, y_train)
        pred_labels_dev = clf.predict(x_dev)
        pred_labels_dev_str = [[convert_binary_vector_to_y(resp)] for resp in pred_labels_dev]
        df = pd.DataFrame(pred_labels_dev_str)
        df.rename(columns={0: col_name}, inplace=True)
        df.to_csv(f'pred_labels_dev_str_knn_{k}.csv', index=False)

        # decsion tree
    for d in range(15, 20):
        clf = MultiOutputClassifier(DecisionTreeClassifier(max_depth=d)).fit(x_train, y_train)
        pred_labels_dev = clf.predict(x_dev)
        pred_labels_dev_str = [[convert_binary_vector_to_y(resp)] for resp in pred_labels_dev]
        df = pd.DataFrame(pred_labels_dev_str)
        df.rename(columns={0: col_name}, inplace=True)
        df.to_csv(f'pred_labels_dev_str_tree_{d}.csv', index=False)

        # randomForest
    for num_classifiers in [100, 150, 200, 250, 300]:
        clf_forest = MultiOutputClassifier(RandomForestClassifier(num_classifiers, random_state=1)).fit(x_train,
                                                                                                        y_train)
        pred_labels_dev = clf_forest.predict(x_dev)
        pred_labels_dev_str = [[convert_binary_vector_to_y(resp)] for resp in pred_labels_dev]
        df = pd.DataFrame(pred_labels_dev_str)
        df.rename(columns={0: col_name}, inplace=True)
        df.to_csv(f'pred_labels_dev_str_forest_{num_classifiers}.csv', index=False)

        # ridge
    for lam in [0.1, 0.2, 0.3, 0.4, 0.5]:
        clf_ridge = MultiOutputClassifier(RidgeClassifier(lam)).fit(x_train, y_train)
        pred_labels_dev = clf_ridge.predict(x_dev)
        pred_labels_dev_str = [[convert_binary_vector_to_y(resp)] for resp in pred_labels_dev]
        df = pd.DataFrame(pred_labels_dev_str)
        df.rename(columns={0: col_name}, inplace=True)
        df.to_csv(f'pred_labels_dev_str_ridge_{lam}.csv', index=False)

def part1_predicting_metastases(x_train, x_dev, labels_train, x_test, invistigate=False):
    # re-write the labels to binary vector - train data
    col_name = labels_train.columns[0]
    ls_origin_labels = [eval(val) for val in labels_train[col_name]]
    labels_train_binary = [convert_y_to_binary_vector(resp) for resp in ls_origin_labels]

    if invistigate:
        part1_compare_models(x_train, labels_train_binary, x_dev, col_name)

    #best classifier,get thw best micro (weighted)
    clf_forest = MultiOutputClassifier(RandomForestClassifier(200, random_state=1)).fit(x_train,
                                                                                                    labels_train_binary)
    pred_labels_dev = clf_forest.predict(x_test)
    pred_labels_dev_str = [[convert_binary_vector_to_y(resp)] for resp in pred_labels_dev]
    df = pd.DataFrame(pred_labels_dev_str)
    df.rename(columns={0: col_name}, inplace=True)
    df.to_csv('part1/predictions.csv', index=False)


def part2_compare_models (x_train, y_train, x_dev):
    #linear regression
    cls_reg = LinearRegression().fit(x_train, y_train)
    pred_dev = cls_reg.predict(x_dev)
    df = pd.DataFrame(pred_dev)
    df.rename(columns={0: labels_1_train.columns[0]}, inplace=True)
    df.to_csv('predictions_linear_reg.csv', index=False)


    #using boosting, ensemble
    cls_boost = GradientBoostingRegressor().fit(x_train, y_train)
    pred_dev = cls_boost.predict(x_dev)
    df = pd.DataFrame(pred_dev)
    df.rename(columns={0: labels_1_train.columns[0]}, inplace=True)
    df.to_csv('predictions_boost.csv', index=False)

    #SGDRegressor
    cls_SGDR = SGDRegressor().fit(x_train, y_train)
    pred_dev = cls_SGDR.predict(x_dev)
    df = pd.DataFrame(pred_dev)
    df.rename(columns={0: labels_1_train.columns[0]}, inplace=True)
    df.to_csv('predictions_SGDR.csv', index=False)

    #Elastic Net Regression
    cls_elasticnet = ElasticNet(0.2).fit(x_train, y_train)
    pred_dev = cls_elasticnet.predict(x_dev)
    df = pd.DataFrame(pred_dev)
    df.rename(columns={0: labels_1_train.columns[0]}, inplace=True)
    df.to_csv('predictions_elasticnet.csv', index=False)

def part2_predicting_tumor_size(x_train, x_dev, labels_train, x_test, invistigate=False):
    if invistigate:
        part2_compare_models(x_train, labels_train, x_dev)
    cls_reg = LinearRegression().fit(x_train, labels_train)
    pred_labels_dev = cls_reg.predict(x_test)
    df = pd.DataFrame(pred_labels_dev)
    df.rename(columns={0: labels_1_train.columns[0]}, inplace=True)
    df.to_csv('part2/predictions.csv', index=False)

def feats_heatmap(data):
    df_corr = data.corr()
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x = df_corr.columns,
         y = df_corr.index,
         z = np.array(df_corr)))
    fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    if len(sys.argv)-1 != 4:
        print("please run with the arguments: 1-path to train data,"
              " 2-path to labels (part1), 3-path to labels (part2),"
              " 4-path to test data")
    filename_train = sys.argv[1]
    filename_labels_0 = sys.argv[2]
    filename_labels_1 = sys.argv[3]
    filename_test = sys.argv[4]

    x_train, x_dev, labels_0_train, labels_0_dev, labels_1_train, labels_1_dev \
        = load_data_and_spilt(filename_train, filename_labels_0, filename_labels_1)

    x_train_c = clean_reorder_data(x_train)
    x_dev_c = clean_reorder_data(x_dev)

    x_test =pd.read_csv(filename_test)
    x_test_c = clean_reorder_data(x_test)

    # extract data to csv for the evaluate script
    x_train_c.to_csv('x_train.csv', index=False)
    x_dev_c.to_csv('x_dev.csv', index=False)
    labels_0_train.to_csv('labels_0_train.csv', index=False)
    labels_0_dev.to_csv('labels_0_dev.csv', index=False)
    labels_1_train.to_csv('labels_1_train.csv', index=False)
    labels_1_dev.to_csv('labels_1_dev.csv', index=False)

    part1_predicting_metastases(x_train_c, x_dev_c, labels_0_train, x_test_c)
    part2_predicting_tumor_size(x_train_c, x_dev_c, labels_1_train, x_test)

    feats_heatmap(x_train_c)
















