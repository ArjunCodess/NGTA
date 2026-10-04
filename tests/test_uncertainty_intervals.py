import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.uncertainty_intervals import weighted_uncertainty_statistics
from src.robustness import selective_risk


def test_weighted_cluster_statistics_match_duplicate_case_reconstruction():
    y=np.array([0,1,1,0,1,0])
    p=np.array([.8,.9,.1,.2,.7,.6])
    score=np.array([.1,.1,.5,.4,.5,.9])
    index=np.array([0,0,1,2,2,2,5])
    order=np.argsort(score,kind="stable")
    starts=np.r_[0,np.flatnonzero(np.diff(score[order])!=0)+1]
    errors=(p>=.5)!=y
    area,auc=weighted_uncertainty_statistics(errors,np.bincount(index,minlength=len(y)),order,starts)
    np.testing.assert_allclose(area,selective_risk(y[index],p[index],score[index])[1],atol=1e-15)
    np.testing.assert_allclose(auc,roc_auc_score(errors[index],score[index]),atol=1e-15)
