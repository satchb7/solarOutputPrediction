from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials


class randomForest():
    def __init__(self, x, y, iterations, trainSplit, validateSplit):
        self.seed = 2
        self.trial = Trials()
        self.x = x
        self.y = y
        self.iterations = iterations
        self.trainSplit = trainSplit
        self.validateSplit = validateSplit
        self.xFullTrain, self.xTest = self.segment(self.x, self.trainSplit)
        self.yFullTrain, self.yTest = self.segment(self.y, self.trainSplit)
        self.xTrain, self.xValidate = self.segment(self.xFullTrain, self.validateSplit)
        self.yTrain, self.yValidate = self.segment(self.yFullTrain, self.validateSplit)

    def segment(self, data, split):
        main, test = train_test_split(data, test_size = split, random_state = 1, shuffle = False)

        return main, test

    def objective(self, params):
        est = int(params['n_estimators'])
        md = int(params['max_depth'])
        msl = int(params['min_samples_leaf'])
        mss = int(params['min_samples_split'])

        model=RandomForestRegressor(n_estimators=est,max_depth=md,min_samples_leaf=msl,min_samples_split=mss)
        model.fit(self.xTrain,self.yTrain)
        pred=model.predict(self.xValidate)
        score=mean_squared_error(self.yValidate,pred)
        return score

    def optimize(self, trial):
        params={'n_estimators':hp.uniform('n_estimators',100,500),
           'max_depth':hp.uniform('max_depth',5,20),
           'min_samples_leaf':hp.uniform('min_samples_leaf',1,5),
           'min_samples_split':hp.uniform('min_samples_split',2,6),
           }
        best=fmin(fn=self.objective, space = params, algo = tpe.suggest,trials = trial, max_evals = self.iterations, rstate = np.random.default_rng(self.seed))
        return best

    def optimizeModel(self):
        best = self.optimize(self.trial)

        self.maxDepth = best['max_depth']
        self.nEstimators = best['n_estimators']
        self.minSampleLeaf = best['min_samples_leaf']
        self.minSamplesSplit = best['min_samples_split']

    def __str__(self):
        ret = 'Max Depth: ' + str(self.maxDepth) + '\n'
        ret += 'Min Samples per Leaf: ' + str(self.minSampleLeaf) + '\n'
        ret += 'Min Samples Split: ' + str(self.minSamplesSplit) + '\n'
        ret += 'Number of Estimators: ' + str(self.nEstimators) + '\n'

        return ret

    def buildBest(self):
        bestFit = RandomForestRegressor(max_depth = int(self.maxDepth), n_estimators = int(self.nEstimators), min_samples_leaf = int(self.minSampleLeaf), min_samples_split = int(self.minSamplesSplit), random_state = 0)
        bestFit = bestFit.fit(self.xTrain, self.yTrain)

        self.pred = bestFit.predict(self.xTest)
        self.bestMse = np.mean((self.pred - self.yTest)**2)
        self.bestR2 = bestFit.score(self.xTest, self.yTest)
