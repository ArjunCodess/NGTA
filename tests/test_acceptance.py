import numpy as np
from src.acceptance import external_compatibility, robustness_acceptance
from src.robustness import mask_observed_values
import pandas as pd


def test_compatibility_margin_direction_and_robustness_harm():
    y=np.tile([0,1],10)
    good=np.where(y, .9,.1)
    bad=np.where(y,.6,.4)
    groups=np.repeat(range(5),4)
    report=external_compatibility(y,good,good,groups,40)
    assert report['accepted'] and report['brier_upper_95']==0
    assert not external_compatibility(y,bad,good,groups,40)['accepted']
    rates=[0,.1,.7]
    assert robustness_acceptance(y,[good,good,good],[good,bad,bad],rates,groups,40)['accepted']
    assert not robustness_acceptance(y,[good,bad,bad],[good,good,good],rates,groups,40)['accepted']


def test_outcome_mask_is_nested_and_cannot_change_existing_missingness():
    frame=pd.DataFrame({'x':np.r_[np.nan,np.ones(199)]})
    y=np.repeat([0,1],100)
    _,low=mask_observed_values(frame,['x'],.1,np.random.default_rng(1),'outcome_dependent_simulation',y)
    shifted,high=mask_observed_values(frame,['x'],.5,np.random.default_rng(1),'outcome_dependent_simulation',y)
    assert np.all(~low|high)
    assert not high[0,0] and pd.isna(shifted.iloc[0,0])
    assert high[y==1].sum()>high[y==0].sum()
