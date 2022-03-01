import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from .MLP import MLP
from ..utils.eval import get_test_metrics, get_val_metrics

class StandardClassifier:
    def __init__(self, method = 'knn', data = None, input_dim = None, params = {}):
        '''
        One class used to handle data, fit models and evaluate them.

        Method: possible methods are
        kNN: k nearest neighbour
        svm-lin: linear support vector machine
        svm-rbf: support vector machine with radial basis function kernel
        logistic: logistic regression
        tree: decision tree
        forest: random forest
        adaboost: boosted trees
        mlp: multi-layer perceptron (neural network)
        catboost: boosted trees
        tabnet: explainable neural network architecture for tabular data
        '''
        self.input_dim = input_dim
        if params == {}:
            params = self.get_default_params(method)
            print('Using default parameters: {}'.format(params))
        self.model_fitted = False
        self.params = params
        self.method = method
        # threshold values for calculating ROC curves
        self.thresh = np.arange(0,1.02,0.01)

        self.init_classifier(method, params)
        if data is not None:
            self.load_data(data)
            self.fit()

        # AUC ROC is not calcualated for svm
        if 'svm' in method:
            self.calc_auc = False
        else:
            self.calc_auc = True

    def init_classifier(self, method, params):
        '''
        Initialise the model.
        '''
        if method == 'knn':
            self.model = KNeighborsClassifier(**params)
        elif method == 'svm-lin':
            self.model = SVC(kernel = 'linear', **params)
        elif method == 'svm-rbf':
            self.model = SVC(kernel = 'rbf', **params)
        elif method == 'logistic':
            self.model = LogisticRegression(**params)
        elif method == 'tree':
            self.model = DecisionTreeClassifier(**params)
        elif method == 'forest':
            self.model = RandomForestClassifier(**params)
        elif method == 'adaboost':
            self.model = AdaBoostClassifier(**params)
        elif method == 'mlp':
            self.model = MLP(**params)
        elif method == 'catboost':
            self.model = CatBoostClassifier(**params)
        elif method == 'tabnet':
            self.model = TabNetClassifier(**params)
        else:
            raise ValueError('Method {} not implemented'.format(method))

    def load_data(self, data):
        '''
        Load data.
        '''
        print('Loading data...')
        self.train_data = data['train']
        self.validation_data = data['eval']
        self.test_data = data['test']
        self.test_galaxies = list(set(data['test']['galaxy']))
        self.model_fitted = False
        self.pred = None
        self.probs = None

    def get_default_params(self, method):
        '''
        If no params file is given, load default params for models.
        '''
        if method == 'knn':
            return {'n_neighbors': 1}
        elif method == 'logistic':
            return {'max_iter': 2000, 'random_state': 424242}
        elif method == 'tree':
             return {'criterion': 'entropy', 'min_samples_split':500, 'random_state': 4242422}
        elif method == 'forest':
            return {'random_state': 42424}
        elif method == 'adaboost':
            return {'n_estimators': 100, 'random_state': 4242, 'base_estimator': RandomForestClassifier()}
        elif method == 'mlp':
            return {'neurons_per_layer': [self.input_dim, 100, 100, 1], 'epochs': 60,
                    'dropout_rate': 0.05, 'learning_rate': 1e-3, 'weight_decay': 1e-5, 'batchsize': 500}
        elif method == 'catboost':
            return {'iterations': 100, 'random_seed': 63, 'learning_rate': 0.5}
        elif method == 'tabnet':
            return {'n_steps': 5, 'n_d': 10, 'n_a': 10}
        else:
            return {}

    def fit(self):
        '''
        Calls fit function of model to train on data.
        '''
        print('Fitting model...')
        if self.method == 'mlp':
            self.model.fit(self.train_data['inputs'], self.train_data['labels'], self.validation_data['inputs'], self.validation_data['labels'])
        elif self.method == 'catboost':
            self.model.fit(self.train_data['inputs'], self.train_data['labels'], eval_set=(self.validation_data['inputs'], self.validation_data['labels']), verbose=False, plot=False)
        elif self.method == 'tabnet':
            self.model.fit(self.train_data['inputs'], self.train_data['labels'], eval_set=[(self.validation_data['inputs'], self.validation_data['labels'])])
        else:
            self.model.fit(self.train_data['inputs'], self.train_data['labels'])
        self.model_fitted = True

    def val(self):
        '''
        Evaluates model on validation data.
        '''
        pred = self.model.predict(self.validation_data['inputs'])
        if self.calc_auc == True:
            probs = self.model.predict_proba(self.validation_data['inputs'])[:,1]
        else:
            probs = None

        self.stats_validation = get_val_metrics(self.validation_data['labels'], pred, probs=probs, thresh=self.thresh)

    def test(self, p=0.5):
        '''
        Evaluates model on test data.
        '''
        assert(self.model_fitted == True)
        if self.calc_auc == True:
            self.probs = self.model.predict_proba(self.test_data['inputs'])[:,1]
        else:
            self.probs = None
        if p == 0.5 or self.calc_auc == False:
            self.pred = self.model.predict(self.test_data['inputs'])
        else:
            self.pred = self.probs>p

        self.stats_galaxies, self.stats_all, self.auc_curve_fdr, self.auc_curve_fpr, self.false_positives, self.false_negatives, self.found_GCs = get_test_metrics(self.test_galaxies, self.test_data['galaxy'], self.test_data['ID'], self.test_data['labels'], self.pred, probs=self.probs, thresh=self.thresh)
