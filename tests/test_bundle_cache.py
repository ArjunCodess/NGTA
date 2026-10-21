from types import SimpleNamespace
import pandas as pd
from src.bundle_cache import load_cached_bundle
from src.wids_loader import WIDSPreprocessor, WIDS_CONTINUOUS_COLUMNS


def test_cache_reuses_only_identical_source_and_options(tmp_path):
    source = tmp_path / 'data'
    source.mkdir()
    csv = source / 'wids_icu.csv'
    csv.write_text('x\n1\n')
    calls=[]
    def loader(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(value=len(calls))
    arguments=dict(dataset='wids',data_dir=source,batch_size=4,seed=0,cache_dir=tmp_path/'cache',imputation='knn')
    assert load_cached_bundle(loader,**arguments).value == 1
    assert load_cached_bundle(loader,**arguments).value == 1
    csv.write_text('x\n2\n')
    assert load_cached_bundle(loader,**arguments).value == 2
    arguments['imputation']='median'
    assert load_cached_bundle(loader,**arguments).value == 3
    assert len(calls)==3


def test_median_imputation_fits_training_only_and_preserves_observed_rules():
    frame=pd.DataFrame({c:[1.,2.,3.,4.] for c in WIDS_CONTINUOUS_COLUMNS})
    frame=frame.assign(encounter_id=range(4),hospital_death=[0,1]*2,elective_surgery=[0,1]*2,gender=['M','F']*2,age=[30,40,50,60])
    processor=WIDSPreprocessor(imputation='median').fit(frame)
    holdout=frame.copy()
    holdout.loc[0,'age']=float('nan')
    encoded=processor.transform_components(holdout)
    assert encoded.numeric_imputed[0,processor.numeric_columns.index('age')] == 45
    assert encoded.rule_triggers[0,2] == 0
