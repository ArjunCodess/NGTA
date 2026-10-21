import numpy as np
import pytest
from src.recalibration import validation_recalibrator, probability_logits


def test_validation_calibration_uses_only_designated_labels():
    labels=np.array([0,0,1,1])
    p=np.array([.1,.3,.6,.8])
    model=validation_recalibrator(labels,p)
    before=model.coef_.copy()
    first=model.predict_proba(probability_logits([.2,.7]))[:,1]
    other_test_labels=np.array([1,0])
    assert other_test_labels.size==first.size
    np.testing.assert_array_equal(model.coef_,before)
    assert np.all((first>=0)&(first<=1))
    with pytest.raises(ValueError):
        validation_recalibrator([0,0,0,0],p)
    with pytest.raises(ValueError):
        probability_logits([.2,float('inf')])
