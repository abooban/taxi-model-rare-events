import ctypes as ct
import itertools
import math
import subprocess
import unittest
import numpy as np
from experiments.validated.build import build
from experiments.validated.conditional import log_tail, expected_cost, iid_log_summary, tilted_prefix
from experiments.validated.bounds import cap_bound
from experiments.validated.config import ModelConfig

class Model(ct.Structure):
    _fields_ = [(name,ct.c_double) for name in ("rate","L","c","b")]
class Prefix(ct.Structure):
    _fields_ = [("pre_cost",ct.c_double),("remainder",ct.c_double),("K",ct.c_uint64)]

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = build()
        cls.lib = ct.CDLL(str(cls.folder/"libcore.so"))
        cls.lib.taxi_log_accept.argtypes = [ct.c_double]*3+[ct.c_uint64]*2+[ct.c_int]
        cls.lib.taxi_log_accept.restype = ct.c_double
        cls.lib.taxi_log_tail.argtypes = [Model,Prefix,ct.c_double,ct.c_int]
        cls.lib.taxi_log_tail.restype = ct.c_double

    def test_native_core(self):
        subprocess.run([str(self.folder/"core-tests")], check=True, timeout=15, capture_output=True)

    def test_exhaustive_suffix_detailed_balance_and_missing_factor_counterexample(self):
        prob = {1:.7,3:.3}
        def paths(x=()):
            return [x] if sum(x)>=4 else sum((paths(x+(v,)) for v in prob),[])
        states = paths()
        cost = lambda x: sum(1+2*max(v-2,0) for v in x)
        base = lambda x: math.prod(prob[v] for v in x)
        for beta in (0,.5,.2,-1/6):
            p = np.array([base(x)*math.exp(-beta*cost(x)) for x in states]); p/=p.sum()
            matrices=[]
            for correct in (True,False):
                P=np.zeros((len(states),len(states)))
                for i,x in enumerate(states):
                    for j,y in enumerate(states):
                        q=sum(base(y[k:])/len(x) for k in range(min(len(x),len(y))) if x[:k]==y[:k])
                        loga=self.lib.taxi_log_accept(beta,cost(x),cost(y),len(x),len(y),int(correct))
                        P[i,j]=q*math.exp(min(0,loga))
                    P[i,i]+=1-P[i].sum()
                matrices.append(p[:,None]*P)
                np.testing.assert_allclose(P.sum(axis=1),1,atol=1e-14)
            self.assertLess(np.max(abs(matrices[0]-matrices[0].T)),1e-14)
            self.assertGreater(np.max(abs(matrices[1]-matrices[1].T)),1e-4)

    def test_fixed_positional_balance_independent_of_active_prefix(self):
        states=list(itertools.product((1,3), repeat=3)); w=np.array([.6,.3,.1])
        def cost(x):
            c=s=0
            for v in x:
                c+=1+2*max(v-2,0); s+=v
                if s>=4:break
            return c,s
        base=lambda x: math.prod(.7 if v==1 else .3 for v in x)
        p=np.array([base(x)*math.exp(-.5*cost(x)[0]) if cost(x)[1]>=4 else 0 for x in states]);p/=p.sum()
        P=np.zeros((8,8))
        for i,x in enumerate(states):
            if p[i]==0:continue
            for a,b in itertools.combinations(range(3),2):
                subset=w[a]*w[b]*(1/(1-w[a])+1/(1-w[b]))
                for vals in itertools.product((1,3),repeat=2):
                    y=list(x);y[a],y[b]=vals;y=tuple(y);j=states.index(y)
                    q=subset*base(vals)
                    alpha=0 if p[j]==0 else math.exp(min(0,self.lib.taxi_log_accept(.5,cost(x)[0],cost(y)[0],1,1,0)))
                    P[i,j]+=q*alpha;P[i,i]+=q*(1-alpha)
        flux=p[:,None]*P
        self.assertLess(np.max(abs(flux-flux.T)),1e-14)

    def test_conditional_formula_atoms_weak_strict_and_extreme_logs(self):
        for model in (ModelConfig(),ModelConfig(b=0),ModelConfig(eta_c=0),ModelConfig(lambda_=2,b=3)):
            m=Model(model.lambda_,model.L,model.eta_c,model.b)
            for pre,a in ((0,10),(5,1),(5,2),(6,0.2)):
                base=pre+1+model.b*max(a-model.eta_c,0)
                for z in (0,base,base+1,20,60,2000):
                    for strict in (False,True):
                        c=self.lib.taxi_log_tail(m,Prefix(pre,a,0),z,strict)
                        self.assertEqual(c,float(log_tail(pre,a,z,model,strict=strict)))
        self.assertLess(float(log_tail(5,1,2000)), -900)
        self.assertAlmostEqual(expected_cost(),13.97737623120548)

    def test_conditional_against_independent_exponential_quadrature(self):
        # Exact integration of piecewise-constant event integrand on O>=0:
        # independently find its jump by bisection, then integrate exp(-lambda O).
        for a in (.1,1.,2.,7.):
            for pre in (0.,6.,12.):
                for z in (10.,20.,30.,60.):
                    tariff=lambda o: pre+1+2*max(a+o-2,0)
                    if tariff(0)>=z: numerical=1.
                    else:
                        lo,hi=0.,1000.
                        for _ in range(80):
                            mid=(lo+hi)/2
                            if tariff(mid)>=z:hi=mid
                            else:lo=mid
                        numerical=math.exp(-hi)
                    self.assertAlmostEqual(math.exp(float(log_tail(pre,a,z))),numerical,delta=2e-15)

    def test_cap_bounds_and_scaled_iid_covariance(self):
        self.assertLess(cap_bound(80,-1/6)["total_variation_upper_bound"],3e-37)
        self.assertLess(cap_bound(80,-1/6)["mean_cost_difference_upper_bound"],1e-30)
        self.assertGreater(cap_bound(2,-1/6)["total_variation_upper_bound"],.1)
        with self.assertRaises(ValueError):cap_bound(80,-.5)
        logs=np.log(np.array([[.1,.2],[.2,.4],[.3,.6],[.4,.8]]))-700
        s=iid_log_summary(logs)
        np.testing.assert_allclose(s["relative_standard_error"],[math.sqrt(1/60)/.5]*2,rtol=1e-12)
        self.assertFalse(any(s["unavailable"]))
        tiny=iid_log_summary(np.array([-1000.,-1001.,-1002.]))
        self.assertTrue(tiny["linear_underflow"][0])
        self.assertTrue(math.isfinite(tiny["log_estimate"][0]))
        self.assertTrue(math.isfinite(tiny["log_standard_error"][0]))

    def test_terminal_transform_and_cost_moment_against_numerical_quadrature(self):
        nodes,weights=np.polynomial.legendre.leggauss(128)
        for model in (ModelConfig(),ModelConfig(eta_c=0),ModelConfig(b=0),ModelConfig(lambda_=2,b=3)):
            for beta in (0.,.5,.2,-1/6):
                for a in (.1,1.,2.,7.):
                    rate=model.lambda_+beta*model.b
                    split=max(model.eta_c-a,0)
                    intervals=[(0,split),(split,split+60/rate)]
                    A=J=0.
                    for lo,hi in intervals:
                        if hi==lo:continue
                        o=(nodes+1)*(hi-lo)/2+lo
                        h=1+model.b*np.maximum(a+o-model.eta_c,0)
                        density=np.exp(math.log(model.lambda_)-model.lambda_*o-beta*h)
                        A+=float(weights@density)*(hi-lo)/2
                        J+=float(weights@(density*h))*(hi-lo)/2
                    loga,mean,_=tilted_prefix(0.,a,1,beta,model)
                    self.assertAlmostEqual(math.exp(float(loga)),A,delta=1e-12*max(1,A))
                    self.assertAlmostEqual(float(mean),J/A,delta=2e-12*max(1,J/A))

if __name__=="__main__":unittest.main()
