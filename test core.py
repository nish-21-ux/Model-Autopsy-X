import numpy as np, pandas as pd, time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from stress import *
from metrics import *
from attribution import *
from drift import *
from failure_v2 import *

X,y=make_classification(n_samples=5000,n_features=8,n_informative=5,n_redundant=2,random_state=42)
X=pd.DataFrame(X,columns=[f"Feature_{i}" for i in range(1,9)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=42)
m=make_pipeline(SimpleImputer(strategy="median"),RandomForestClassifier(n_estimators=150,random_state=42)).fit(Xtr,ytr)
std=Xtr.std(); med=Xtr.median()
imp=pd.Series(m[-1].feature_importances_,index=X.columns).sort_values(ascending=False)
print(imp.round(3).to_dict())
drift_feats=list(imp.index[:2])
sev={"Feature Drift":1.5,"Measurement Noise":1.0,"Missing Values":0.2}
t=time.time()
phi,tot,tab=shapley_fault_attribution(m,Xte,yte,COMPONENT_ORDER,sev,std,drift_feats,n_seeds=3)
print("shapley",{k:round(v,4) for k,v in phi.items()},"sum",round(sum(phi.values()),4),"total",round(tot,4),round(time.time()-t,1),"s")
assert abs(sum(phi.values())-tot)<1e-9
Xs=apply_components(Xte,set(COMPONENT_ORDER),sev,0,std,drift_feats)
print("restore",feature_restoration(m,Xte,Xs,yte,drift_feats))
p=m.predict_proba(Xs)[:,1]
print({k:round(v,3) for k,v in metric_suite(yte,p).items()})
print(bootstrap_ci(yte,p))
pb=make_pipeline(SimpleImputer(),RandomForestClassifier(n_estimators=20,random_state=1)).fit(Xtr,ytr).predict_proba(Xs)[:,1]
print(paired_bootstrap(yte,p,pb))
print(degradation_auc([0,1,2,3],[.9,.8,.7,.5]),breaking_point([0,1,2,3],[.9,.8,.7,.5],0.75))
t=time.time()
print(univariate_drift(Xtr.sample(1500,random_state=0),Xs).round(3).head(4))
auc,imp2=classifier_two_sample_test(Xtr,Xs); print("c2st",round(auc,3),imp2.head(3).round(3).to_dict(),"null",round(null_drift_score(Xtr),3),round(time.time()-t,1),"s")
tab_,cats=paired_failure_breakdown(m,Xte,Xs,yte); print(tab_)
err=(m.predict(Xs)!=yte).astype(int)
print(feature_error_auc(Xs,err,med).head(3)); print(error_rules(Xs,err,med)[:400]); print(confidence_profile(p,yte))