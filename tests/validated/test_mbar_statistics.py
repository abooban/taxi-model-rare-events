import math
import json
from pathlib import Path
from dataclasses import replace
import unittest
import numpy as np
from experiments.validated.config import ModelConfig, Ensemble, Bias, Provenance
from experiments.validated.mbar import SamplingWindow, fit_mbar, EnsembleMismatch, ConvergenceError, OverlapError
from experiments.validated.reconstruct import summarize
from experiments.validated.statistics import (inefficiency, indicator_diagnostics, chain_comparison,
                                             burnin_diagnostics, ratio_influence, wilson_interval)
from experiments.validated.resampling import full_refit_bootstrap

COSTS=np.array([6.,8.,12.,20.]);TRUTH=np.array([.1,.25,.5,.15])

def window(costs,beta,identifier,*,independent=False,ensemble=Ensemble()):
    return SamplingWindow(ModelConfig(),ensemble,Bias(beta),Provenance("synthetic_exact_law",identifier),
                          costs,np.asarray(costs,dtype=np.int64),np.zeros(len(costs),dtype=np.int64),independent)

def exact_windows():
    betas=[0,math.log(2)/2,-math.log(2)/2]
    # Independently rational fixture counts from S2.
    fixture=json.loads((Path(__file__).parent/"fixtures/legacy_estimator_case.json").read_text())
    counts=[row["raw_counts"] for row in fixture["windows"]]
    return [window(np.repeat(COSTS,h),b,str(i)) for i,(h,b) in enumerate(zip(counts,betas))]

def independent_newton(costs,betas,counts):
    """Independent convex-likelihood solve, not the self-consistency iteration."""
    b=np.asarray(betas);counts=np.asarray(counts);u=b[:,None]*costs
    reference=int(np.flatnonzero(b==0)[0]);free=np.array([i for i in range(len(b)) if i!=reference])
    f=np.zeros(len(b))
    for _ in range(100):
        a=np.log(counts)[:,None]+f[:,None]-u
        den=np.logaddexp.reduce(a,axis=0)
        r=np.exp(a-den)
        grad=r.sum(axis=1)-counts
        if np.max(abs(grad))<1e-10:return f
        H=np.diag(r.sum(axis=1))-r@r.T
        delta=np.linalg.solve(H[np.ix_(free,free)],grad[free])
        objective=float(den.sum()-counts@f)
        scale=1.
        while scale>1e-8:
            new=f.copy();new[free]-=scale*delta
            value=float(np.logaddexp.reduce(np.log(counts)[:,None]+new[:,None]-u,axis=0).sum()-counts@new)
            if value<=objective+1e-9:f=new;break
            scale/=2
    raise AssertionError("independent Newton solve did not converge")

class MbarTests(unittest.TestCase):
    def test_exact_rational_fixture_and_independent_solver(self):
        fit=fit_mbar(exact_windows())
        np.testing.assert_allclose([fit.weights.normalized_weights[fit.weights.costs==c].sum() for c in COSTS],TRUTH,atol=2e-11)
        z=np.exp(-np.array(fit.betas)[:,None]*COSTS)@TRUTH
        np.testing.assert_allclose(fit.f,-np.log(z),atol=2e-10)
        second=independent_newton(fit.weights.costs,fit.betas,fit.state_counts)
        np.testing.assert_allclose(fit.f,second,atol=2e-9)
        np.testing.assert_allclose(fit.overlap.sum(axis=1),1,atol=2e-10)
        self.assertTrue(fit.connected);self.assertGreater(fit.spectral_gap,.05)
        self.assertLess(fit.residual,1e-10)
        report=summarize(fit.weights)
        self.assertAlmostEqual(report.total_mass,1,places=10)
        self.assertAlmostEqual(report.overflow_mass,0)

    def test_actual_counts_duplicate_reference_and_cap_errors(self):
        windows=exact_windows()
        with self.assertRaises(ValueError):fit_mbar(windows+windows)
        shared=("pcg32-xsh-rr-64-32",123456)
        with self.assertRaises(ValueError):
            fit_mbar([replace(w,stream_identity=shared) for w in windows])
        with self.assertRaises(EnsembleMismatch):fit_mbar(windows[1:])
        capped=window(windows[1].costs,windows[1].bias.beta,"capped",ensemble=Ensemble("finite_M",80))
        with self.assertRaises(EnsembleMismatch):fit_mbar([windows[0],capped])
        with self.assertRaises(ConvergenceError):fit_mbar(windows,max_iterations=1)
        # Splitting one chain changes bookkeeping, not retained n or the fitted law.
        costs=windows[0].costs
        split=[window(costs[:10],0,"s1"),window(costs[10:],0,"s2"),*windows[1:]]
        np.testing.assert_allclose(fit_mbar(split).f,fit_mbar(windows).f,atol=1e-10)

    def test_disconnected_or_near_singular_inputs_fail_closed(self):
        windows=[window(np.full(100,6.),0,"zero"),
                 window(np.full(100,1000.),-.1,"negative")]
        with self.assertRaises((ConvergenceError,OverlapError)):
            fit_mbar(windows,max_iterations=100)

    def test_full_refit_bootstrap_degenerate_tail_not_zero_error(self):
        rng=np.random.Generator(np.random.PCG64(12017))
        windows=[]
        for i,beta in enumerate((0,.2,-1/6)):
            p=TRUTH*np.exp(-beta*COSTS);p/=p.sum()
            windows.append(window(rng.choice(COSTS,1000,p=p),beta,str(i),independent=True))
        out=full_refit_bootstrap(windows,blocks=1,replicates=25,thresholds=(20.,60.))
        self.assertTrue(out["normalizers_refitted"])
        self.assertGreater(out["f_standard_error"][0],0)
        self.assertIsNone(out["observables"]["P(C>=60)"]["standard_error"])
        self.assertGreater(out["observables"]["mean_C"]["standard_error"],0)

class StatisticsTests(unittest.TestCase):
    def test_iid_ar1_constant_and_rank_drift_checks(self):
        rng=np.random.Generator(np.random.PCG64(8721));noise=rng.normal(size=60000)
        self.assertLess(abs(inefficiency(noise)["g"]-1),.2)
        ar=np.zeros(len(noise))
        for i in range(1,len(ar)):ar[i]=.8*ar[i-1]+noise[i]
        self.assertLess(abs(inefficiency(ar)["g"]-9),2)
        self.assertIsNone(inefficiency(np.zeros(20))["ess"])
        self.assertIsNone(indicator_diagnostics(np.zeros(20))["standard_error"])
        good=chain_comparison([noise[:10000],noise[10000:20000],noise[20000:30000],noise[30000:40000]])
        self.assertLess(good["rank_split_rhat"],1.01)
        bad=chain_comparison([noise[:10000],noise[10000:20000]+2])
        self.assertGreater(bad["rank_split_rhat"],1.2)
        drift=burnin_diagnostics(np.r_[noise[:10000],noise[10000:20000]+2],2000)
        self.assertGreater(abs(drift["first_last_standardized_drift"]),20)

    def test_ratio_influence_joint_covariance_and_zero_binomial_events(self):
        value,series=ratio_influence(np.array([0.,1.,2.,3.]),np.log([1.,2.,3.,4.]))
        self.assertAlmostEqual(value,2)
        self.assertAlmostEqual(float(series.mean()),0)
        # An observed zero is not proof of a zero probability or zero uncertainty.
        low,high=wilson_interval(0,1000)
        self.assertEqual(low,0);self.assertGreater(high,.003)

if __name__=="__main__":unittest.main()
