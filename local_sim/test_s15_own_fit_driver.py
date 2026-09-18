"""Label-free contract checks for the bounded own-fit scheduler."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
import s15_own_fit_driver as D


class OwnFitSchedule(unittest.TestCase):
    def test_dry_plan_is_disjoint_and_capped(self):
        p=D.plan()
        self.assertEqual(p["ordered_seeds"],list(D.SEEDS))
        self.assertEqual(set(p["fit_indices"])&set(p["validation_indices"]),set())
        self.assertEqual(p["scheduled_evaluations"],19)
        self.assertLessEqual(p["scheduled_evaluations"],p["max_evaluations"])
        self.assertEqual(p["full_network_runs"],228)

    def test_scheduler_never_reads_validation_before_freeze(self):
        calls=[]
        def fake(split,x,index):
            calls.append((split,x,index))
            return 1.0+.01*(x[0]-.75)**2+.02*(x[1]-1.1)**2
        chosen,base,held_base,held,history=D.coordinate_fit(fake)
        self.assertEqual(len(calls),19)
        self.assertEqual([c[2] for c in calls],list(range(19)))
        self.assertEqual([c[0] for c in calls],['fit']*17+['validation']*2)
        self.assertEqual(calls[0][1],D.X0)
        self.assertEqual(calls[-2][1],D.X0)
        self.assertEqual(calls[-1][1],chosen)
        self.assertTrue(all(lo<=v<=hi for v,(lo,hi) in zip(chosen,D.BOUNDS)))
        self.assertGreater(base,0)
        self.assertGreater(held_base,0)
        self.assertGreater(held,0)

    def test_fixed_steps_and_dimensionless_ridge(self):
        calls=[]
        D.coordinate_fit(lambda split,x,index: calls.append((split,x)) or 1.0)
        # Every nondefault trial is exactly one declared coordinate step from
        # the incumbent. With flat scores, ridge retains the default.
        self.assertEqual(calls[1][1],(.75,1.,0.,1.))
        self.assertEqual(calls[2][1],(1.25,1.,0.,1.))
        self.assertEqual(calls[-1][1],D.X0)
        self.assertAlmostEqual(D.objective(1.,1.,D.X0),1.)
        self.assertAlmostEqual(D.objective(1.,1.,(.75,1.,0.,1.)),1.+.001*.25**2)
        with self.assertRaises(ValueError):
            D.objective(1.,0.,D.X0)

    def test_design_spec_discloses_disabled_local_seconds(self):
        spec=D.design_spec()
        deviations=spec.deviations_from_round()
        self.assertEqual(spec.round_tag,"v2-phase2")
        self.assertEqual(spec.lam,0.)
        self.assertIsNone(spec.residual_wall_time_limit_s)
        self.assertIsNone(spec.wall_time_limit_s)
        self.assertIn("residual_wall_time_limit_s",deviations)
        self.assertIn("wall_time_limit_s",deviations)
        self.assertIn("predict_timeout_s",deviations)
        self.assertIn("setup_timeout_s",deviations)


if __name__=="__main__":
    unittest.main()
