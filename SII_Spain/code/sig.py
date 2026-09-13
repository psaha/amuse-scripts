import numpy as np

class SIG():
          
      def sqVisi(self, u, v, x, y, d, lam, cv):

          x = x / d
          y = y / d

          u = u / lam
          v = v / lam

          V = np.zeros_like(u, dtype=complex)

          for xi, yi, ci in zip(x, y, cv):
              phase = np.exp(-2j*np.pi*(u*xi + v*yi))
              V += ci * phase

          V /= np.sum(cv)

          return np.abs(V)**2

