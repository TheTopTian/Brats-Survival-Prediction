from dataset import SurCombineFeaturing,give_me_all
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,VotingRegressor
from sklearn.linear_model import LinearRegression,BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor


threshold=[10*30,15*30]

def bucket(x,threshold):
    _x = np.zeros_like(x)
    for i,t in enumerate(threshold):
        _x[x>t] = i+1
    return _x

def metric(y,p):
    _y = bucket(y,threshold)
    _p = bucket(p,threshold)
    return f1_score(_y,_p,average='micro')    
    # return accuracy_score(_y,_p)


def standarlize(x):
    return (x-x.mean(0))/x.std(0)


def prepare():
    dataset = SurCombineFeaturing()
    train_loader = dataset.loader('train')
    test_loader  = dataset.loader('valid')
    x_train,y_train = give_me_all(train_loader)
    x_test ,y_test  = give_me_all(test_loader)
    x = np.concatenate([x_train,x_test],0)
    y = np.concatenate([y_train,y_test],0)
    return x,y

def common(model):
    x,y = prepare()
    scores = cross_validate(model,x,y,scoring=make_scorer(metric,greater_is_better=True))
    return scores['test_score']




class Regression:

    @staticmethod
    def DecisionTree():
        return common(DecisionTreeRegressor())
      
    @staticmethod
    def RandomForest():
        return common(RandomForestRegressor())

    @staticmethod
    def Linear():
        return common(LinearRegression())

    @staticmethod
    def Pipeline():
        return common(make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3)))

    @staticmethod
    def MLP():
        return common(MLPRegressor(hidden_layer_sizes=(128,64,32,8),max_iter=400))

    @staticmethod
    def AdaBoost():
        return common(AdaBoostRegressor())

    @staticmethod
    def GradientBoosting():
        return common(GradientBoostingRegressor(learning_rate=0.1))


class Visualizer:
    @staticmethod
    def scatter():
        methods = [i for i in dir(Regression) if not i.startswith('__')]
        data = {
            'name':methods,
            'score':[],
            'std':[]
        }
        for method in methods:
            result = getattr(Regression,method)()
            data['score'].append(result.mean())
            data['std'].append(result.std())

        fig,ax = plt.subplots(1,1,figsize=(12,8))
        sns.scatterplot(data=data,x='name',y='score',hue='score',ax=ax)
        ax.grid()
        # ax.set_xticklabels(data['name'])
        ax.set_title('Method-F1 score')
        plt.show()
    @staticmethod
    def barplot():
        methods = [i for i in dir(Regression) if not i.startswith('__')]
        data = {
            'name' :[],
            'score':[]
        }
        for method in methods:
            result = getattr(Regression,method)()
            data['name'].extend([method] * len(result))
            data['score'].extend(result.tolist())
        sns.set(font_scale = 2.0)
        fig,ax = plt.subplots(1,1,figsize=(12,8))
        sns.barplot(data=data,x='name',y='score',ax=ax,ci=None)
        plt.xlabel('Regression Methods')
        plt.xticks(rotation=30)
        plt.ylabel('Accuracy')
        ax.set_facecolor('white')
        ax.grid(color = 'gray')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        plt.savefig('Regression_Methods.png', bbox_inches = 'tight')
        plt.show()
        


if __name__ == '__main__':
    Visualizer.barplot()