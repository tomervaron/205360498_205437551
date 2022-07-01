from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel, SelectFdr, f_classif, SelectKBest, RFE
from sklearn.svm import SVC
from mrmr import mrmr_classif
import sklearn_relief as relief
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import make_scorer, matthews_corrcoef, accuracy_score, precision_recall_curve, auc, average_precision_score, roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_iris
import time
import scipy.io
from scipy.io import arff
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
#####################################
# $ from internet https://datascience.stackexchange.com/questions/58565/conditional-entropy-calculation-in-python-hyx $
##Entropy
def entropy(Y):
    """
    H(Y)
    Also known as Shanon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    unique, count = np.unique(Y.astype(int), return_counts=True, axis=0)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en

#Joint Entropy
def jEntropy(Y, X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    YX = np.c_[Y, X]
    return entropy(YX)

#Conditional Entropy
def cEntropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return jEntropy(Y, X) - entropy(X)


#Mutual information
def MI(X, Y):
    """
    Mutual information,
    I(X;Y) = H(Y) - H(Y|X)
    Reference: https://en.wikipedia.org/wiki/Information_gain_in_decision_trees#Formal_definition
    """
    return entropy(Y) - cEntropy(Y, X)

#####################################################
#####################################################


# Conditional Mutual Information
def CMI(X, Y, Z):
    """
    Conditional Mutual Information,
    I(X; Y|Z) = H(X|Z) + H(Y|Z) - H(X, Y|Z)
    """
    return cEntropy(X, Z) + cEntropy(Y, Z) - cEntropy(np.c_[Y, X], Z)

#####################################################
def MRMD(X, y, k, print_steps=False):
    """
    @@ X :
    @@ y :
    @@ K : number of wanted features
    """
    ## stage 1
    # init the sets S and F
    num_col = X.shape[1]
    if num_col <= k:
        return np.ones((num_col,), dtype=int)

    cols_indxs = np.arange(num_col)
    S = set()
    F = set(cols_indxs)

    if print_steps:
        print(f'List of Features\n{cols_indxs}\n')
        print("################################ Iteration Number 1: ################################\n")

    ## stage 2
    # find the MI for y with every feature
    dict_of_mi = {}
    for fi in F:
        xi = X[:, fi]
        dict_of_mi[fi] = MI(xi, y)

        if print_steps:
            print(f'{fi} Mutual Information score:  {dict_of_mi[fi]}')

    ## stage 3
    # find the feature with the max MI
    max_fi = max(dict_of_mi, key=dict_of_mi.get)

    if print_steps:
        print(f'\nFeature Selected:  {max_fi}\n')

    # remove max_fi from F
    # add max_fi to S
    F.remove(max_fi)
    S.add(max_fi)

    if print_steps:
        print(f'New Set S: {S}\nNew Set F: {F}\n')

    ## stage 4
    # greedy selection
    ### a. Calculate the new feature redundancy term
    iter_num = 1
    while len(S) != k:
        J_xk = {}

        if print_steps:
            iter_num += 1
            print(f'################################################################ Iteration Number {iter_num}: ################################################################')

        for Xk_str in F:
            Xk = X[:, Xk_str]
            J_xk[Xk_str] = {"I(X;Y)": MI(Xk, y), "Formula19": 0}
            for Xj_str in S:
                Xj = X[:, Xj_str]
                mi_of_Xj_Xk = MI(Xj, Xk)
                CMI_of_Xk_y_Xj = CMI(Xk, y, Xj)
                # calculate formula 19 from the the artical
                J_xk[Xk_str]["Formula19"] += (mi_of_Xj_Xk - CMI_of_Xk_y_Xj)
            J_xk[Xk_str]["J(Xk)"] = J_xk[Xk_str]["I(X;Y)"] - ((1/len(S)) * J_xk[Xk_str]["Formula19"])

            if print_steps:
                print(f'J( {Xk_str} ) = {J_xk[Xk_str]["J(Xk)"]}\t\t\tFeature Dependency: {J_xk[Xk_str]["I(X;Y)"]}\t\t\tFeature Redundancy: {((1/len(S)) * J_xk[Xk_str]["Formula19"])}')

        ### b. Select the next feature
        max_J_xk = ""
        for Xk_str in F:
            if max_J_xk == "":
                max_J_xk = Xk_str
            else:
                if J_xk[Xk_str]["J(Xk)"] > J_xk[max_J_xk]["J(Xk)"]:
                    max_J_xk = Xk_str

        if print_steps:
            print(f'\nFeature Selected:  {max_J_xk}\n')

        #### remove and add the selected feature
        F.remove(max_J_xk)
        S.add(max_J_xk)

        if print_steps:
            print(f'New Set S: {S}\nNew Set F: {F}\n\n\n')

    list_to_return = []
    for f_name in cols_indxs:
        if f_name in S:
            list_to_return.append(1)
        else:
            list_to_return.append(0)
    return np.array(list_to_return)

######################################################################
class selectKBestSwitcher(BaseEstimator, TransformerMixin):

    def __init__(self, score_func=None, k=2):
        """
        A Custom TransformerMixin that can switch between score_func.
        :param score_func: sklearn object - The score_func
        :param k: number of feature to select
        """
        self.score_func = score_func
        self.k = k
        self.selectKBest_model = SelectKBest(score_func, k)
        self.MRMD_model = None

    def fit(self, X, y=None, **fit_params):

        if hex(id(self.score_func)) != hex(id(MRMD)):
            self.selectKBest_model = SelectKBest(self.score_func, self.k)
            self.selectKBest_model.fit(X, y)
        else:
            self.MRMD_model = self.score_func(X, y, self.k)
        return self

    def transform(self, X, **transform_params):
        if hex(id(self.score_func)) != hex(id(MRMD)):
            return self.selectKBest_model.transform(X)
        else:
            x_to_return = X
            indx = 0
            list_of_indx_to_remove = []
            for i in self.MRMD_model:
                if i == 0:
                    list_of_indx_to_remove.append(indx)
                indx += 1
            x_to_return = np.delete(x_to_return, list_of_indx_to_remove, 1)
            return x_to_return


class ClfSwitcher(BaseEstimator):

    def __init__(self, estimator=SGDClassifier(),):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """

        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)


# ######################## pipline ################################
#
# score_dict = {
#     "MCC": make_scorer(matthews_corrcoef),
#     "ACC": make_scorer(accuracy_score),
#     # "PR-AUC": make_scorer(precision_recall_curve),
#     "PR-AUC": make_scorer(average_precision_score),
#     "AUC": make_scorer(auc),
# }
#
# parameters_MRMD = {
#     'clf__estimator': [KNeighborsClassifier(), RandomForestClassifier(), LogisticRegression(), SVC(), GaussianNB()],
#     # "feature_selection__score_func": [MRMD(), mrmr_classif(), f_classif(), SelectFdr(sr()), SelectFromModel(estimator=SVC())],
#     "feature_selection__score_func": [MRMD],
#     "feature_selection__k": [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100],
# }
#
# pipe_MRMD = Pipeline(steps=[ #("completing_missing_values", SimpleImputer()),
#                        # ("categorical_handle", ),
#                        #("remove_std_zero", VarianceThreshold()),
#                        #("normalization", PowerTransformer()),
#                        # ("numeric_handle", ),
#                        # ("empty_value_in_y", TransformedTargetRegressor(func=handle_y)),
#                        # ("feature_selection", SelectKBest()),
#                        ("feature_selection", selectKBestSwitcher()),
#                        ('clf', ClfSwitcher()),
#                        ],
#                 verbose=True)
#
#
# search_MRMD = GridSearchCV(pipe_MRMD, parameters_MRMD, cv=2, scoring=score_dict, verbose=3, refit="AUC")
# x = df.iloc[:, df.columns != '1'].to_numpy()
# y = df["1"].to_numpy()
# # search_MRMD.fit(x, y)
#
# parameters = {
#     'clf__estimator': [KNeighborsClassifier(), RandomForestClassifier(), LogisticRegression(), SVC(), GaussianNB()],
#     "feature_selection__k": [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100],
# }
# kbst= SelectKBest(mrmr_classif)
# pipe_mrmr_classif = Pipeline(steps=[ #("completing_missing_values", SimpleImputer()),
#                        # ("categorical_handle", ),
#                        #("remove_std_zero", VarianceThreshold()),
#                        #("normalization", PowerTransformer()),
#                        # ("numeric_handle", ),
#                        # ("empty_value_in_y", TransformedTargetRegressor(func=handle_y)),
#                        ("feature_selection", kbst),
#                        ('clf', ClfSwitcher()),
#                        ],
#                 verbose=True)
#
# search_mrmr_classif = GridSearchCV(pipe_mrmr_classif, parameters, cv=2, scoring=score_dict, verbose=3, refit="AUC")
# search_mrmr_classif.fit(x, y)
# # pipe.fit(df.iloc[:, df.columns != '1'], df["1"])
# # Pipeline([('clf', KNeighborsClassifier())]).fit(df.iloc[:, df.columns != '1'], df["1"])

####################################################################################################################

def selector_transform(s_vec, x_train, x_test):
    """
    transform the df according to selector vector
    :param s_vec: selector vector
    :param x_train:
    :param x_test:
    :return: x_train, x_test with only the selected features
    """
    indx = 0
    list_of_indx_to_append = []
    for i in s_vec:
        if i == 1:
            list_of_indx_to_append.append(str(indx))
        indx += 1
    x_train = x_train[list_of_indx_to_append]
    x_test = x_test[list_of_indx_to_append]
    return x_train, x_test

def get_feature_names_and_score(x_train, s_vec):
    """
    get the features names and scores
    :param x_train:
    :param s_vec:
    :return: dict of name: score
    """
    dict_of_name_and_score = {}
    i = 0
    for name in x_train.columns:
        if s_vec[i] == 1:
            dict_of_name_and_score[name] = 1
        i += 1
    return dict_of_name_and_score

# read data
data1 = arff.loadarff('ARFF/Breast.arff')
df1 = pd.DataFrame(data1[0])
df1_y = df1["Class"].to_numpy()
df1_x = df1.iloc[:, df1.columns != 'Class']
df1_y[df1_y == b'relapse'] = 1
df1_y[df1_y == b'non-relapse'] = 0

## test that there or no Categorical features
# columns = df1_x.shape[1]
# for i in range(columns):
#     g = df1_x[:, i].tolist()
#     if type(g[0]) != float:
#         print(type(g[0]))

# read data
data2 = arff.loadarff('ARFF/CNS.arff')
df2 = pd.DataFrame(data2[0])
df2_y = df2["CLASS"].to_numpy()
df2_x = df2.iloc[:, df2.columns != 'CLASS']
df2_y[df2_y == b'1'] = 1
df2_y[df2_y == b'0'] = 0

## test that there or no Categorical features
# columns = df2_x.shape[1]
# for i in range(columns):
#     g = df2_x[:, i].tolist()
#     if type(g[0]) != float:
#         print(type(g[0]))

# read data
mat1 = scipy.io.loadmat('scikit_feature_datasets/leukemia.mat')
mat1_x = pd.DataFrame(mat1['X'])
mat1_x.columns = mat1_x.columns.astype(str)
mat2 = scipy.io.loadmat('scikit_feature_datasets/colon.mat')
mat2_x = pd.DataFrame(mat2['X'])
mat2_x.columns = mat2_x.columns.astype(str)
r1 = mat1['Y'].reshape(-1,)
r1[r1 == -1] = 0
r2 = mat2['Y'].reshape(-1,)
r2[r2 == -1] = 0

## test that there or no Categorical features
# columns = mat1['X'].shape[1]
# for i in range(columns):
#     g = mat1['X'][:, i].tolist()
#     if type(g[0]) != float and type(g[0]) != int:
#         print(type(g[0]))
## test that there or no Categorical features
# columns = mat2['X'].shape[1]
# for i in range(columns):
#     g = mat2['X'][:, i].tolist()
#     if type(g[0]) != float and type(g[0]) != int:
#         print(type(g[0]))

k_list = [1, 2, 5, 10, 20]
# k_list = [1, 2]
clf_list = {'KNeighborsClassifier': KNeighborsClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'LogisticRegression': LogisticRegression(),
            'SVC': SVC(probability=True),
            'GaussianNB': GaussianNB()
            }

feature_selection_list = ['MRMD', 'SelectFdr', 'mrmr_classif', 'relief', 'RFE']

data_dict = {
            'scikit_feature_datasets/leukemia': {'X': mat1_x, 'Y': r1},
            'scikit_feature_datasets/colon': {'X': mat2_x, 'Y': r2},
            'ARFF/Breast': {'X': df1_x, 'Y': df1_y},
            'ARFF/CNS': {'X': df2_x, 'Y': df2_y}
             }

# init the preprocess
simpleI = SimpleImputer(missing_values=np.nan, strategy='mean')
vt = VarianceThreshold()
pt = PowerTransformer()
five_bd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

# init the dict for the run
dict_for_runs = {}
for data_set_ in data_dict.keys():
    for clf_name_ in clf_list.keys():
        for fs_ in feature_selection_list:
            for k_ in k_list:
                st = f'{data_set_},{clf_name_},{fs_},{k_}'
                dict_for_runs[st] = {
                    "matthews_corrcoef": 0,
                    "accuracy": 0,
                    "PR-AUC": 0,
                    "AUC": 0,
                    "fs_time": 0,
                    "fit_clf_time": 0,
                }

for data_set in data_dict.keys():
    data_dict[data_set]['X_deal'] = simpleI.fit_transform(data_dict[data_set]['X'])
    # data_dict[data_set]['X'] = vt.fit_transform(data_dict[data_set]['X'])
    data_dict[data_set]['X_deal'] = pt.fit_transform(data_dict[data_set]['X_deal'])
    data_dict[data_set]['X_deal'] = five_bd.fit_transform(data_dict[data_set]['X_deal'])
    # handle the y array
    label_encoder = LabelEncoder()
    data_dict[data_set]['Y'] = label_encoder.fit_transform(data_dict[data_set]['Y'])
    loo = StratifiedKFold(n_splits=2)
    loo.get_n_splits(data_dict[data_set]['X_deal'], data_dict[data_set]['Y'])
    for train_index, test_index in loo.split(data_dict[data_set]['X_deal'], data_dict[data_set]['Y']):
        X_train, X_test = pd.DataFrame(data_dict[data_set]['X_deal'][train_index]), pd.DataFrame(data_dict[data_set]['X_deal'][test_index])
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)
        y_train, y_test = data_dict[data_set]['Y'][train_index], data_dict[data_set]['Y'][test_index]
        for clf_name in clf_list.keys():
            for fs in feature_selection_list:
                for k in k_list:

                    start_time_fs_method = time.time()

                    if fs == 'MRMD':
                        selector = SelectKBest(f_classif, k=1000).fit(X_train, y_train)
                        X_train_fs = pd.DataFrame(selector.transform(X_train))
                        X_test_fs = pd.DataFrame(selector.transform(X_test))
                        X_train_fs.columns = X_train_fs.columns.astype(str)
                        X_test_fs.columns = X_test_fs.columns.astype(str)
                        selected_features = MRMD(X_train_fs.to_numpy(), y_train, k=k)
                        # features_dict = get_feature_names_and_score(data_dict[data_set]['X'], selected_features)
                        X_train_fs, X_test_fs = selector_transform(selected_features, X_train_fs, X_test_fs)

                    elif fs == 'SelectFdr':
                        selector = SelectFdr(alpha=0.1).fit(X=X_train, y=y_train)
                        # feature_scores = selector.scores_
                        X_train_fs, X_test_fs = selector.transform(X_train), selector.transform(X_test)

                    elif fs == 'mrmr_classif':
                        selected_features = mrmr_classif(X=X_train, y=pd.Series(y_train), K=k)
                        X_train_fs, X_test_fs = X_train[selected_features], X_test[selected_features]

                    elif fs == 'relief':
                        selector = relief.Relief(n_features=k).fit(X_train, y_train)
                        X_train_fs, X_test_fs = selector.transform(X_train), selector.transform(X_test)

                    else:
                        estimator = SVC(kernel="linear")
                        selector = RFE(estimator, n_features_to_select=k).fit(X_train, y_train)
                        X_train_fs, X_test_fs = selector.transform(X_train), selector.transform(X_test)

                    end_time_fs_method = time.time()
                    time_of_fs = end_time_fs_method - start_time_fs_method

                    # clf fit
                    start_time_fit_clf = time.time()
                    clf_list[clf_name].fit(X=X_train_fs, y=y_train)
                    end_time_fit_clf = time.time()
                    time_of_fit_clf = end_time_fit_clf - start_time_fit_clf

                    # clf predict
                    pred_y = clf_list[clf_name].predict(X_test_fs)
                    matthews_corrcoef_score = matthews_corrcoef(y_true=y_test, y_pred=pred_y)
                    accuracy = accuracy_score(y_true=y_test, y_pred=pred_y)
                    clf_pred_prob = clf_list[clf_name].predict_proba(X_test_fs)[:, 1]
                    PR_AUC_score = roc_auc_score(y_test, clf_pred_prob)
                    auc_score = average_precision_score(y_test, clf_pred_prob)

                    st = f'{data_set},{clf_name},{fs},{k}'
                    dict_for_runs[st]["matthews_corrcoef"] = (dict_for_runs[st]["matthews_corrcoef"] + matthews_corrcoef_score)/2
                    dict_for_runs[st]["accuracy"] = (dict_for_runs[st]["accuracy"] + accuracy)/2
                    dict_for_runs[st]["PR-AUC"] = (dict_for_runs[st]["PR-AUC"] + PR_AUC_score)/2
                    dict_for_runs[st]["AUC"] = (dict_for_runs[st]["AUC"] + auc_score)/2
                    dict_for_runs[st]["fs_time"] = (dict_for_runs[st]["fs_time"] + time_of_fs)/2
                    dict_for_runs[st]["fit_clf_time"] = (dict_for_runs[st]["fit_clf_time"] + time_of_fit_clf)/2

dict_to_turn_to_df = {
    "Dataset Name": [],
    "Number of samples": [],
    "Original Number of features": [],
    "Filtering Algorithm": [],
    "Filtering time": [],
    "Learning algorithm": [],
    "fit time": [],
    "Number of features selected": [],
    "CV Method": [],
    "matthews_corrcoef Value": [],
    "accuracy Value": [],
    "PR-AUC Value": [],
    "AUC Value": [],
}

# build the df
for key in dict_for_runs.keys():
    data_set, clf_name, fs, k = key.split(",")
    dict_to_turn_to_df["Dataset Name"].append(data_set)
    if data_set == 'scikit_feature_datasets/leukemia.mat':
        dict_to_turn_to_df["Number of samples"].append(71)
        dict_to_turn_to_df["Original Number of features"].append(7070)

    elif data_set == 'scikit_feature_datasets/colon.mat':
        dict_to_turn_to_df["Number of samples"].append(62)
        dict_to_turn_to_df["Original Number of features"].append(2000)

    elif data_set == 'ARFF/Breast.arff':
        dict_to_turn_to_df["Number of samples"].append(97)
        dict_to_turn_to_df["Original Number of features"].append(24482)

    else:
        dict_to_turn_to_df["Number of samples"].append(60)
        dict_to_turn_to_df["Original Number of features"].append(7130)

    dict_to_turn_to_df["Filtering Algorithm"].append(fs)
    dict_to_turn_to_df["Filtering time"].append(dict_for_runs[key]["fs_time"])
    dict_to_turn_to_df["Learning algorithm"].append(clf_name)
    dict_to_turn_to_df["fit time"].append(dict_for_runs[key]["fit_clf_time"])
    dict_to_turn_to_df["Number of features selected"].append(k)
    dict_to_turn_to_df["CV Method"].append("10FoldCV")
    dict_to_turn_to_df["matthews_corrcoef Value"].append(dict_for_runs[key]["matthews_corrcoef"])
    dict_to_turn_to_df["accuracy Value"].append(dict_for_runs[key]["accuracy"])
    dict_to_turn_to_df["PR-AUC Value"].append(dict_for_runs[key]["PR-AUC"])
    dict_to_turn_to_df["AUC Value"].append(dict_for_runs[key]["AUC"])

# save the data
df = pd.DataFrame.from_dict(dict_to_turn_to_df)
df.to_csv("results.csv")