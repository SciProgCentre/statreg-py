from unittest import TestCase, skip
from statreg import FourierBasis, CubicSplines
from statreg.examples import getOpticsKernels
import numpy as np


class TestBasis(TestCase):

    def setUp(self):
        self.listOfBasis = [
            FourierBasis(0, 1, 10),
            CubicSplines(np.linspace(-1, 1, 10)),
            ]

    def test_omega(self):
        for i, basis in enumerate(self.listOfBasis):
            with self.subTest(i=i):
                basis.omega(0)
                self.assertTrue(True)



    def test_discretizeKernel(self):
        kenrel = getOpticsKernels("gaussian")
        ys = np.linspace(-1, 1, 10)
        for i, basis in enumerate(self.listOfBasis):
            with self.subTest(i=i):
                basis.discretizeKernel(kenrel, ys)
                self.assertTrue(True)



