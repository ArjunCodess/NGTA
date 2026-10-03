"""Paired, clustered research acceptance tests with explicit margin directions."""
import numpy as np
from sklearn.metrics import roc_auc_score
from .evaluation import paired_bootstrap_indices


def external_compatibility(labels, nars, mc, groups=None, iterations=1000, seed=0):
    y=np.asarray(labels)
    left=np.asarray(nars,dtype=float)
    right=np.asarray(mc,dtype=float)
    if left.shape!=y.shape or right.shape!=y.shape or not np.isfinite([left,right]).all() or np.any((np.array([left,right])<0)|(np.array([left,right])>1)):
        raise ValueError("Compatibility requires aligned finite probabilities")
    indices=paired_bootstrap_indices(y,iterations,np.random.default_rng(seed),groups)
    brier=(left-y)**2-(right-y)**2
    observed_auc=float(roc_auc_score(y,left)-roc_auc_score(y,right))
    brier_samples=np.array([brier[i].mean() for i in indices])
    auc_samples=np.array([roc_auc_score(y[i],left[i])-roc_auc_score(y[i],right[i]) for i in indices])
    upper=float(np.percentile(brier_samples,97.5))
    lower=float(np.percentile(auc_samples,2.5))
    return dict(brier_nars_minus_mc=float(brier.mean()),brier_upper_95=upper,brier_margin=.001,
                auc_nars_minus_mc=observed_auc,auc_lower_95=lower,auc_margin=-.01,
                accepted=bool(upper<.001 and lower>-.01),sampling_unit='cluster' if groups is not None else 'case',
                interpretation='compatibility margins from the research plan; not superiority or clinical certification')


def robustness_acceptance(labels, nars_by_rate, mc_by_rate, rates, groups=None, iterations=1000, seed=0):
    y=np.asarray(labels)
    rates=np.asarray(rates,dtype=float)
    left=np.asarray(nars_by_rate,dtype=float)
    right=np.asarray(mc_by_rate,dtype=float)
    if (left.shape != right.shape or left.shape!=(len(rates),len(y)) or len(rates)<2
            or rates[0]!=0 or np.any(np.diff(rates)<=0) or not np.isfinite([left,right]).all()
            or np.any((np.array([left,right])<0)|(np.array([left,right])>1))):
        raise ValueError("Robustness requires aligned probabilities at increasing rates starting at zero")
    delta=(left-y)**2-(right-y)**2
    degradation=delta-delta[0]
    integrated=np.trapezoid(degradation,rates,axis=0)/(rates[-1]-rates[0])
    indices=paired_bootstrap_indices(y,iterations,np.random.default_rng(seed),groups)
    samples=np.array([integrated[i].mean() for i in indices])
    upper=float(np.percentile(samples,97.5))
    tail=2.5/len(rates)
    levels=[]
    for rate,loss in zip(rates,delta):
        bootstrap=np.array([loss[i].mean() for i in indices])
        levels.append(dict(rate=float(rate),brier_nars_minus_mc=float(loss.mean()),
                           upper_family_95=float(np.percentile(bootstrap,100-tail))))
    no_harm=all(row['upper_family_95']<=.001 for row in levels)
    return dict(integrated_degradation_nars_minus_mc=float(integrated.mean()),integrated_upper_95=upper,
                levels=levels,mask_harm_margin=.001,accepted=bool(upper<0 and no_harm),
                sampling_unit='cluster' if groups is not None else 'case',
                interpretation='paired degradation versus MC-only; Bonferroni per-level harm intervals; research criterion')
