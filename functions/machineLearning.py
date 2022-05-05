from sklearn.discriminant_analysis import   LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import  train_test_split
import numpy as np

max_deph=np.arange(1,7,1)
n_estimators=np.arange(1,10,1)
learning_rate=np.arange(0.001,1,0.005)
min_samples_split=np.arange(1,10,1)
C=np.arange(0.1,100,0.5)
cv=3
CLASSOFICATION_METRIC="f1"
scorer=make_scorer(f1_score,average="macro")
def get_train_test(X,y,train_size=0.7,shuffle= False):
     X_train, X_test, y_train, y_test = train_test_split(
                                X,
                                y,
                                train_size   = float(train_size),
                                random_state = 1234,
                                shuffle      = shuffle
                            )
     return X_train, X_test, y_train, y_test
def random_search(gbrt,param_grid,X_train,y_train):
    grid_search = RandomizedSearchCV(gbrt ,param_grid, cv=cv,
    scoring=scorer)
    grid_search.fit(X_train,y_train)
    params=grid_search.best_params_
    return params
def fit_predict(gbrt,X_train,y_train,X_test,y_test):
    gbrt.fit(X_train,y_train)
    y_pred=gbrt.predict(X_test)
    p=precision_score(y_test,y_pred,average="macro")
    r=recall_score(y_test,y_pred,average="macro")
    f1=f1_score(y_test,y_pred,average="macro")
    ac=accuracy_score(y_test,y_pred)
    return gbrt,p,r,f1,ac
#los siguientes modelos seleccionan parametros con randomSearch
def gradientBoostingClassify(X_train,y_train,X_test,y_test):
    gbrt = GradientBoostingClassifier()
    param_grid=[{"max_depth":max_deph,"n_estimators":n_estimators,"learning_rate":learning_rate}]
    params=random_search(gbrt,param_grid,X_train,y_train)
    gbrt = GradientBoostingClassifier(max_depth=params['max_depth'], n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
    return fit_predict(gbrt,X_train,y_train,X_test,y_test)
def AdaBoostingClassify(X_train,y_train,X_test,y_test):
    gbrt =  AdaBoostClassifier()
    param_grid=[{"n_estimators":n_estimators,"learning_rate":learning_rate}]
    params=random_search(gbrt,param_grid,X_train,y_train)
    gbrt = AdaBoostClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
    return fit_predict(gbrt,X_train,y_train,X_test,y_test)
def RandomForestClassify(X_train,y_train,X_test,y_test):
    gbrt = RandomForestClassifier()
    param_grid=[{"max_depth":max_deph,"n_estimators":n_estimators,"min_samples_split":min_samples_split}]
    params=random_search(gbrt,param_grid,X_train,y_train)
    gbrt = RandomForestClassifier(max_depth=params['max_depth'], n_estimators=params['n_estimators'],min_samples_split=params['min_samples_split'])
    return fit_predict(gbrt,X_train,y_train,X_test,y_test)
def DecissionTreeClassify(X_train,y_train,X_test,y_test):
    gbrt =  DecisionTreeClassifier()
    param_grid=[{"max_depth":max_deph,"min_samples_split":min_samples_split}]
    params=random_search(gbrt,param_grid,X_train,y_train)
    gbrt = DecisionTreeClassifier(max_depth=params['max_depth'],min_samples_split=params['min_samples_split'])
    return fit_predict(gbrt,X_train,y_train,X_test,y_test)
def LinearSVMClassify(X_train,y_train,X_test,y_test):
    gbrt =  LinearSVC( multi_class="crammer_singer")
    param_grid=[{"C":C}]
    params=random_search(gbrt,param_grid,X_train,y_train)
    gbrt = LinearSVC(C=params['C'],multi_class="crammer_singer")
    return fit_predict(gbrt,X_train,y_train,X_test,y_test)
def LogisticRegressionClassify(X_train,y_train,X_test,y_test):
    log_reg = LogisticRegression(multi_class="multinomial")
    log_reg.fit(X_train, y_train,X_train,y_train)
    y_pred=log_reg.predict(X_test)
    p = precision_score(y_test, y_pred, average="macro")
    r = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    ac=accuracy_score(y_test,y_pred)
    params={}
    return log_reg,p,r,f1,ac,params
def LinearDiscriminantClassify(X_train,y_train,X_test,y_test):
    clf = LinearDiscriminantAnalysis()
    y_pred=clf.predict(X_test)
    p = precision_score(y_test, y_pred, average="macro")
    r = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    ac=accuracy_score(y_test,y_pred)
    params={}
    return clf,p,r,f1,ac,params