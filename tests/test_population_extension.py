"""Scientific regression tests against the supplied population-extension PDF.

Run with: PYTHONPATH=src python3 -m unittest discover -s tests
          -p 'test_population_extension.py'
"""
import unittest

import numpy as np
from scipy.integrate import quad

from wnt_pinn.population.model import (
    NOMINAL, Parameters, Protocol, exposure, rhs,
)
from wnt_pinn.population.simulation import compare_at, maintenance_events, paired


class PopulationExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Parameters()
        cls.control, cls.treated = paired(cls.p)

    def test_reported_severe_population_ratios(self):
        """Table 8 and Sections 10.1--10.2: external numerical targets."""
        row = compare_at(self.control, self.treated, self.p, 88.)
        for key, expected in {'NT_ratio': .957, 'NS_ratio': .696, 'FS_ratio': .728,
                              'FA_ratio': .790, 'IH_ratio': 1.514, 'b_ratio': .719,
                              'h5_ratio': 4.116, 'h13_ratio': .677}.items():
            with self.subTest(key=key):
                self.assertAlmostEqual(row[key], expected, delta=.001)

    def test_reported_rebound_time(self):
        """Table 9: 94.9 h is relative to simulation baseline, not ATRA onset."""
        events = maintenance_events(self.control, self.treated, self.p)
        self.assertAlmostEqual(events['crossing_h'], 94.9, delta=.1)
        self.assertAlmostEqual(events['NS_ratio'], .685, delta=.001)
        self.assertAlmostEqual(events['crossing_h'], events['nadir_NS']['time_h'], delta=1e-4)

    def test_positive_boundary_and_capacity_are_invariant(self):
        """No compartment can flow outward at zero or above carrying capacity."""
        rng = np.random.default_rng(829)
        for _ in range(20):
            y = np.r_[rng.uniform(.01,3.,7), rng.uniform(0.,1.,4), rng.dirichlet([1,1,1])]
            for j in range(14):
                boundary = y.copy()
                boundary[j] = 0.
                self.assertGreaterEqual(rhs(60., boundary, self.p, NOMINAL)[j], -1e-14)
            self.assertLessEqual(rhs(60., y, self.p, NOMINAL)[11:].sum(), 1e-14)
            for j in range(7,11):
                boundary = y.copy()
                boundary[j] = 1.
                self.assertLessEqual(rhs(60., boundary, self.p, NOMINAL)[j], 0.)

    def test_equal_input_area(self):
        """Integrated smooth windows retain the same area despite rounded edges."""
        protocols = [NOMINAL, Protocol('front',1.5,((40.,64.,2.),)),
                     Protocol('pulses',1.5,tuple((x,x+6.,2.) for x in (40.,54.,68.,82.)))]
        for protocol in protocols:
            area = quad(lambda t: protocol.amplitude*exposure(t,self.p,protocol),
                        -200.,300.,epsabs=1e-9,points=[40.,46.,54.,60.,64.,68.,74.,82.,88.])[0]
            self.assertAlmostEqual(area,72.,places=7)

    def test_switching_conserves_total_stem_population(self):
        """Changing only switching can redistribute L/A, never their total flux."""
        y = self.treated.sol(60.)
        changed = self.p.changed(qLA0=.1,qLAC=.2,qAL0=.15,qALB=.05)
        base = rhs(60.,y,self.p,NOMINAL)
        altered = rhs(60.,y,changed,NOMINAL)
        self.assertGreater(abs(base[11]-altered[11]),.001)
        self.assertAlmostEqual(base[11:13].sum(),altered[11:13].sum(),places=14)


if __name__ == '__main__':
    unittest.main()
