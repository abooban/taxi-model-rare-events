from dataclasses import replace
from fractions import Fraction
import ast
import math
import unittest
import numpy as np

from experiments.validated.config import Ensemble, HistogramSpec, ModelConfig, ReconstructionConfig
from experiments.validated.reconstruct import (
    EnsembleMismatch, NumericalRangeError, SampleWindow, balance_weights, linear_mass,
    logsumexp, raw_count_log_masses, stabilize, summarize)
from fixture_support import COUNTS, COSTS, FIXTURE, H, ROOT, RUNS, TARGET, TRUTH, exact_windows, run


class ExactRegressionTests(unittest.TestCase):
    def test_fixture_normalizers_and_counts_with_exact_fraction_arithmetic(self):
        p = [Fraction(v) for v in FIXTURE["probability_fractions"]]
        self.assertEqual(sum(p), 1)
        for record in FIXTURE["windows"]:
            m = record["beta_log2_over_2_multiplier"]
            factors = [Fraction(2)**(-m*(c//2)) for c in FIXTURE["costs"]]
            z = sum(a*b for a, b in zip(p, factors))
            self.assertEqual(z, Fraction(record["normalizer_fraction"]))
            self.assertEqual(record["raw_counts"],
                             [record["sample_count"]*a*b/z for a, b in zip(p, factors)])
            self.assertEqual(sum(record["raw_counts"]), record["sample_count"])

    def test_raw_count_formula_recovers_exact_law_without_normalization(self):
        logs = raw_count_log_masses(H, COUNTS, RUNS, TARGET, COSTS)
        np.testing.assert_allclose(np.exp(logs), TRUTH, rtol=3e-14, atol=0)

    def test_per_observation_balance_recovers_exact_law_and_mass(self):
        weights = balance_weights(exact_windows(), TARGET)
        report = summarize(weights)
        np.testing.assert_allclose(list(report.atom_mass.values()), TRUTH, rtol=3e-14)
        np.testing.assert_array_equal(report.bin_mass, [0, 0])
        self.assertAlmostEqual(report.total_mass, 1, places=14)
        self.assertEqual(report.normalization, "absolute_known_normalizer")

    def test_balance_expectation_identity_is_independently_rational(self):
        p = [Fraction(v) for v in FIXTURE["probability_fractions"]]
        for j, c in enumerate(FIXTURE["costs"]):
            den = sum(Fraction(w["sample_count"]) * Fraction(2)**(
                -w["beta_log2_over_2_multiplier"]*(c//2)) /
                Fraction(w["normalizer_fraction"]) for w in FIXTURE["windows"])
            self.assertEqual(sum(Fraction(w["raw_counts"][j]) for w in FIXTURE["windows"])/den, p[j])

    def test_archived_estimator_is_preserved_and_wrong_even_after_unit_normalization(self):
        # Read syntax only: importing this archived script would write a PNG.
        tree = ast.parse((ROOT/"python/final_distribution.py").read_text())
        additions = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.AugAssign)]
        self.assertIn("numerator += weighted_counts", additions)
        self.assertTrue(any(s.startswith("P_combined /=") for s in additions))
        den = sum(n*np.exp(-r.bias.beta*COSTS-r.log_normalizer) for r, n in zip(RUNS, COUNTS))
        numerator = H[0].astype(float)
        for row, r in zip(H[1:], RUNS[1:]):
            a = r.bias.beta*COSTS
            numerator += row*np.exp(a-a.max())
        legacy = numerator/den
        normalized = legacy/legacy.sum()
        self.assertAlmostEqual(normalized.sum(), 1)
        self.assertGreater(float(np.max(abs(normalized-TRUTH))), .03)
        ratios = legacy/TRUTH
        self.assertGreater(float(ratios.max()/ratios.min()), 1.5)

    def test_single_window_formula_and_direct_only_limit(self):
        for window in exact_windows():
            weights = balance_weights([window], TARGET)
            expected = window.config.log_normalizer + (
                window.config.bias.beta*window.costs) - math.log(len(window.costs))
            np.testing.assert_allclose(weights.log_weights, expected, atol=2e-15)
        direct = balance_weights([exact_windows()[0]], TARGET)
        np.testing.assert_allclose(np.exp(direct.log_weights), 1/20, rtol=2e-15)

    def test_duplicating_rows_updates_n_and_preserves_exact_reconstruction(self):
        windows = exact_windows()
        w = windows[1]
        windows[1] = SampleWindow(w.config, np.tile(w.costs, 2), np.tile(w.steps, 2),
                                  np.tile(w.long_steps, 2))
        result = balance_weights(windows, TARGET)
        self.assertEqual(result.window_counts, (20, 1478, 238))
        np.testing.assert_allclose(list(summarize(result).atom_mass.values()), TRUTH, rtol=3e-14)

    def test_split_identical_window_equals_unsplit_empirical_result(self):
        window = exact_windows()[1]
        split = [SampleWindow(window.config, window.costs[s], window.steps[s], window.long_steps[s])
                 for s in (slice(0, 123), slice(123, None))]
        a = summarize(balance_weights([window], TARGET))
        b = summarize(balance_weights(split, TARGET))
        np.testing.assert_allclose(list(a.atom_mass.values()), list(b.atom_mass.values()), rtol=3e-14)

    def test_preweighted_counts_and_inconsistent_n_rejected(self):
        for counts, ns in ((H.astype(float), COUNTS), (H, [19, 739, 238]),
                           (H, [20, 739]), (H, [20, 0, 238])):
            with self.subTest(ns=ns), self.assertRaises(ValueError):
                raw_count_log_masses(counts, ns, RUNS, TARGET, COSTS)

    def test_empty_category_has_zero_mass(self):
        h = np.column_stack([H, np.zeros(3, dtype=int)])
        logs = raw_count_log_masses(h, COUNTS, RUNS, TARGET, [*COSTS, 30])
        self.assertEqual(logs[-1], -math.inf)
        self.assertEqual(linear_mass(logs[-1]), 0)

    def test_continuous_bin_center_error_removed_by_observation_weights(self):
        # p=(1/2,1/2), bias ratio across the two costs is exactly 1/2.
        c = np.array([6.2, 7.8])
        beta = math.log(2)/(c[1]-c[0])
        z = .5*math.exp(-beta*c[0]) + .5*math.exp(-beta*c[1])
        config = run(beta, math.log(z))
        target = replace(TARGET, histogram=HistogramSpec((6, 8)))
        window = SampleWindow(config, [6.2, 6.2, 7.8], [6, 6, 6], [1, 1, 1])
        exact = summarize(balance_weights([window], target))
        self.assertAlmostEqual(exact.bin_mass[0], 1, places=14)
        center = np.exp(raw_count_log_masses([[3]], [3], [config], target, [7]))[0]
        self.assertGreater(abs(center-1), .05)
        refined = np.exp(raw_count_log_masses([[2, 1]], [3], [config], target, c))
        np.testing.assert_allclose(refined, [.5, .5], rtol=2e-14)
        finer = summarize(balance_weights([window], replace(target, histogram=HistogramSpec((6, 7, 8)))))
        self.assertAlmostEqual(sum(finer.bin_mass), exact.bin_mass[0], places=14)


class MassAndSupportTests(unittest.TestCase):
    def test_atoms_continuous_integer_cost_edges_and_outside_mass_partition(self):
        target = replace(TARGET, histogram=HistogramSpec((8, 10, 20, 40)))
        c = [6, 8, 8, 8.5, 10, 12, 20, 40, 41]
        n = [6, 8, 6, 6, 10, 6, 10, 10, 10]
        long = [0, 0, 1, 1, 0, 1, 1, 1, 1]
        report = summarize(balance_weights([SampleWindow(run(), c, n, long)], target))
        np.testing.assert_allclose(report.bin_mass, np.array([2, 1, 2])/9, rtol=3e-14)
        self.assertEqual(set(report.atom_mass), {8, 10})
        np.testing.assert_allclose(list(report.atom_mass.values()), [1/9, 1/9], rtol=3e-14)
        self.assertAlmostEqual(report.underflow_mass, 1/9)
        self.assertAlmostEqual(report.overflow_mass, 1/9)
        self.assertAlmostEqual(report.reported_mass, 7/9)
        self.assertAlmostEqual(report.diagnostics["partition_log_mass"], report.total_log_mass)
        self.assertAlmostEqual(float(np.dot(report.continuous_density, [2, 10, 20])),
                               sum(report.bin_mass))

    def test_atom_outside_range_is_not_lost_or_range_renormalized(self):
        target = replace(TARGET, histogram=HistogramSpec((7, 10)))
        window = SampleWindow(run(), [6, 8, 12], [6, 8, 12], [0, 0, 0])
        report = summarize(balance_weights([window], target), normalize=True)
        self.assertAlmostEqual(report.reported_mass, 1/3)
        self.assertAlmostEqual(report.underflow_mass, 1/3)
        self.assertAlmostEqual(report.overflow_mass, 1/3)
        self.assertAlmostEqual(report.total_mass, 1)

    def test_flat_tariff_atoms_include_above_threshold_steps(self):
        model = ModelConfig(b=0)
        target = ReconstructionConfig(model, Ensemble(), HistogramSpec((1, 4)))
        window = SampleWindow(run(model=model), [1, 2, 3], [1, 2, 3], [1, 2, 0])
        report = summarize(balance_weights([window], target))
        np.testing.assert_array_equal(report.bin_mass, [0])
        self.assertEqual(set(report.atom_mass), {1, 2, 3})

    def test_known_normalizer_finite_sample_mass_error_is_visible(self):
        # A possible nonrepresentative sample from an exact biased discrete law.
        window = SampleWindow(RUNS[1], [6], [6], [0])
        weights = balance_weights([window], TARGET)
        absolute = summarize(weights)
        self.assertGreater(abs(absolute.total_mass-1), .1)
        self.assertAlmostEqual(absolute.diagnostics["absolute_mass_error"], absolute.total_mass-1)
        normalized = summarize(weights, normalize=True)
        self.assertAlmostEqual(normalized.total_mass, 1)
        self.assertEqual(normalized.source_total_log_mass, absolute.total_log_mass)
        self.assertEqual(normalized.normalization, "self_normalized_full_sample")

    def test_no_observations_in_reporting_range_retains_overflow(self):
        target = replace(TARGET, histogram=HistogramSpec((1, 5)))
        report = summarize(balance_weights([exact_windows()[0]], target))
        self.assertEqual(report.reported_mass, 0)
        self.assertAlmostEqual(report.overflow_mass, 1)

    def test_finite_M_common_target_works_but_mixed_caps_are_rejected(self):
        target = replace(TARGET, ensemble=Ensemble("finite_M", 40))
        windows = [SampleWindow(replace(w.config, ensemble=target.ensemble),
                                w.costs, w.steps, w.long_steps) for w in exact_windows()]
        np.testing.assert_allclose(list(summarize(balance_weights(windows, target)).atom_mass.values()),
                                   TRUTH, rtol=3e-14)
        with self.assertRaises(EnsembleMismatch):
            balance_weights(windows, TARGET)
        w = windows[0]
        windows[0] = SampleWindow(replace(w.config, ensemble=Ensemble("finite_M", 80)),
                                  w.costs, w.steps, w.long_steps)
        with self.assertRaises(EnsembleMismatch):
            balance_weights(windows, target)

    def test_model_mismatch_unknown_metadata_and_cap_violation_fail(self):
        window = exact_windows()[0]
        with self.assertRaises(EnsembleMismatch):
            balance_weights([window], replace(TARGET, model=ModelConfig(L=9)))
        with self.assertRaises(ValueError):
            SampleWindow(None, [8], [8], [0])
        with self.assertRaises(EnsembleMismatch):
            SampleWindow(run(ensemble=Ensemble("finite_M", 6)), [8], [8], [0])

    def test_invalid_observations_and_empty_inputs_rejected(self):
        for c, n, long in (([], [], []), ([8], [8, 9], [0]), ([math.nan], [8], [0]),
                           ([math.inf], [8], [0]), ([8], [8.0], [0]), ([8], [8], [-1]),
                           ([8], [8], [9]), ([8], [9], [0]), ([8.5], [8], [0]),
                           ([5], [5], [0]), ([8], [True], [0]),
                           (np.array([np.longdouble("1e400")]), [6], [1])):
            with self.subTest(c=c, n=n, long=long), self.assertRaises(ValueError):
                SampleWindow(run(), c, n, long)
        with self.assertRaises(ValueError):
            balance_weights([], TARGET)


class NumericalTests(unittest.TestCase):
    def test_stabilizing_shift_restores_absolute_mass_and_cancels_in_ratio(self):
        weights = balance_weights(exact_windows(), TARGET)
        high = float(max(weights.log_weights))
        for shift in (None, high-100, high, high+100):
            stable = weights.stabilized(shift)
            np.testing.assert_allclose(stable.values/stable.values.sum(),
                                       weights.normalized_weights, rtol=4e-14, atol=0)
            self.assertAlmostEqual(stable.restored_log_total, weights.log_total_mass, places=13)

    def test_extreme_exponents_stable_in_common_mixture(self):
        # Exact point law at 20: Z_beta=exp(-20000), never formed linearly.
        direct = SampleWindow(run(), [20], [20], [0])
        biased = SampleWindow(run(1000, -20000), [20], [20], [0])
        report = summarize(balance_weights([direct, biased], TARGET))
        self.assertAlmostEqual(report.atom_mass[20], 1)
        self.assertTrue(np.all(np.isfinite(report.bin_mass)))

    def test_tiny_category_preserved_as_log_mass_when_linear_weights_underflow(self):
        # Toy log probabilities (-14000, log(1-exp(-14000))) give nearly
        # balanced biased probabilities; the small baseline mass is log-only.
        window = SampleWindow(run(1000, math.log(2)-20000), [6, 20], [6, 20], [0, 0])
        weights = balance_weights([window], TARGET)
        report = summarize(weights)
        self.assertTrue(math.isfinite(report.atom_log_mass[6]))
        self.assertLess(report.atom_log_mass[6], -1000)
        self.assertEqual(report.atom_mass[6], 0)
        self.assertEqual(report.diagnostics["linear_underflow_categories"], 1)
        self.assertEqual(weights.stabilized().underflow_count, 1)

    def test_unrepresentable_absolute_output_is_explicit_and_logs_survive(self):
        # Equal baseline masses at 6 and 20, beta=1000. Observing the second
        # state is possible but exceptionally rare under this biased law.
        window = SampleWindow(run(1000, -6000-math.log(2)), [6, 20], [6, 20], [0, 0])
        weights = balance_weights([window], TARGET)
        report = summarize(weights)
        self.assertGreater(report.total_log_mass, 1000)
        self.assertIsNone(report.diagnostics["absolute_total_mass"])
        with self.assertRaises(NumericalRangeError):
            _ = report.total_mass
        normalized = summarize(weights, normalize=True)
        self.assertAlmostEqual(normalized.total_mass, 1)
        self.assertEqual(normalized.source_total_log_mass, report.source_total_log_mass)

    def test_numerical_domain_errors_are_clear(self):
        with self.assertRaises(NumericalRangeError):
            stabilize([0, -1], -1000)
        with self.assertRaises(NumericalRangeError):
            _ = stabilize([0, -1], 1000).restored_log_total
        self.assertEqual(logsumexp([]), -math.inf)
        for values in ([math.nan], [math.inf]):
            with self.assertRaises(NumericalRangeError):
                logsumexp(values)
        # A possible very rare observation from an equal two-point baseline
        # at 6 and 13: absolute bin mass fits, its narrow-bin density does not.
        narrow = replace(TARGET, histogram=HistogramSpec((13, math.nextafter(13, math.inf))))
        observed = SampleWindow(run(100, -600-math.log(2)), [13], [6], [1])
        report = summarize(balance_weights([observed], narrow))
        self.assertTrue(math.isfinite(report.bin_mass[0]))
        self.assertTrue(math.isfinite(report.continuous_log_density[0]))
        with self.assertRaises(NumericalRangeError):
            _ = report.continuous_density
        # Even log arithmetic needs a finite representable beta*C.
        window = SampleWindow(run(1e308, -1), [20], [20], [0])
        with self.assertRaisesRegex(NumericalRangeError, "beta.C"):
            balance_weights([window], TARGET)
