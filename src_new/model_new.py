from dataset_Combination_4images_sel import SurCombineFeaturing,give_me_all
from sklearn import linear_model
import numpy as np
import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
# from sksurv.linear_model import CoxPHSurvivalAnalysis

def predict_using(model):
    dataset = SurCombineFeaturing()
    train_loader = dataset.loader('train')
    test_loader  = dataset.loader('test')
    x,y = give_me_all(train_loader)
    # a = [(x) for x in y]
    # y = np.array(a, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    model.fit(x,y)
    x = give_me_all(test_loader)
    p = model.predict(x)
    data = {
        'name':test_loader.dataset.ids,
        'survival_days':list(map(int,p))
    }
    df = pd.DataFrame.from_dict(data)
    df.to_csv('./output.csv',header=False,index=False)



if __name__ == '__main__':
    # predict_using(RandomForestRegressor())
    # predict_using(linear_model.Lasso(alpha=0.1))
    # predict_using(HistGradientBoostingRegressor())
    # predict_using(ElasticNet(max_iter = 1000))
    predict_using(make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3)))