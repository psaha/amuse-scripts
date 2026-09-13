from scipy.special import gamma
import numpy as np

class SPH():
      
      def W(self, x, y, z, h):
          r = np.sqrt(x**2 + y**2 + z**2)
          w = (1.0 / (h * np.sqrt(np.pi))) ** 3 * np.exp(-(r**2) / h**2)
          return w
          
      def gradW(self, x, y, z, h):
          r = np.sqrt(x**2 + y**2 + z**2)
          n = -2 * np.exp(-(r**2) / h**2) / h**5 / (np.pi) ** (3 / 2)
          wx = n * x
          wy = n * y
          wz = n * z
          return wx, wy, wz
          
      def getPairwiseSeparations(self, ri, rj):
          M = ri.shape[0]
          N = rj.shape[0]
  
          rix = ri[:, 0].reshape((M, 1))
          riy = ri[:, 1].reshape((M, 1))
          riz = ri[:, 2].reshape((M, 1))

          rjx = rj[:, 0].reshape((N, 1))
          rjy = rj[:, 1].reshape((N, 1))
          rjz = rj[:, 2].reshape((N, 1))

          dx = rix - rjx.T 
          dy = riy - rjy.T
          dz = riz - rjz.T
          return dx, dy, dz
          
      def getDensity(self, r, pos, m, h):
          M = r.shape[0]
          dx, dy, dz = self.getPairwiseSeparations(r, pos)
          rho = np.sum(m * self.W(dx, dy, dz, h), 1).reshape((M, 1))
          return rho
          
      def getPressure(self, rho, k, n):
          P = k * rho ** (1 + 1 / n)
          return P
          
      def getHarmonic(self, k, n, M, R):
          gm1 = 2 * k * (1 + n) * np.pi**(-3/(2 * n)) / R**2
          gm2 = M * gamma(5 / 2 + n)
          gm3 = R**3 * gamma(1 + n)
          lmbda = gm1 * (gm2 / gm3)**(1 / n)
          return lmbda
          
      def getPressureAcc(self, pos, m, h, k, n):
          rho = self.getDensity(pos, pos, m, h)
          P = self.getPressure(rho, k, n)

          dx, dy, dz = self.getPairwiseSeparations(pos, pos)
          dWx, dWy, dWz = self.gradW(dx, dy, dz, h)

          ax = -np.sum(m*(P/rho**2 + P.T/rho.T**2)*dWx, axis=1)
          ay = -np.sum(m*(P/rho**2 + P.T/rho.T**2)*dWy, axis=1)
          az = -np.sum(m*(P/rho**2 + P.T/rho.T**2)*dWz, axis=1)
          return np.column_stack((ax, ay, az))
          
      def getHarAcc(self, pos, lmbda):
          N = pos.shape[0]
          a = np.zeros((N,3))
          c = np.average(pos, axis=0)
          
          a = -lmbda * (pos - c)
          return a
          
      def getGravAcc(self, pos, M1, M2, N1, G):
          N = pos.shape[0]
          a = np.zeros((N,3))
          c1 = np.average(pos[:N1], axis=0)
          c2 = np.average(pos[N1:], axis=0)
          r12 = c2 - c1
          r = np.linalg.norm(r12)             # the length of r12
          eps = 1e-8
          aorb1 = G * M2 * r12 / (r**3 + eps)
          aorb2 = G * M1 * r12 / (r**3 + eps)
          
          a[:N1] = aorb1
          a[N1:] = -aorb2
          return a
          
      def getDampAcc(self, vel, N1, nu1, nu2):

          dAcc = np.zeros_like(vel)

          vcom1 = np.mean(vel[:N1], axis=0)
          vcom2 = np.mean(vel[N1:], axis=0)

          dAcc[:N1] = -nu1*(vel[:N1]-vcom1)
          dAcc[N1:] = -nu2*(vel[N1:]-vcom2)

          return dAcc
          
      def getAcc(self, pos, vel, M1, M2, N1, lmbda1, lmbda2, h1, h2, G, k1, k2, n1, n2, nu1, nu2, relax=True):
          N = pos.shape[0]
          a = np.zeros((N,3))
          
          N2 = N - N1 
          m1 = M1/N1
          m2 = M2/N2
          
          # accelaration due to Pressure
          a[:N1] += self.getPressureAcc(pos[:N1], m1, h1, k1, n1)
          a[N1:] += self.getPressureAcc(pos[N1:], m2, h2, k2, n2)
          
          # accelaration due to self-potential
          a[:N1] += self.getHarAcc(pos[:N1], lmbda1)
          a[N1:] += self.getHarAcc(pos[N1:], lmbda2)
          
          if relax:
             # Damping only during relaxation
             a += self.getDampAcc(vel,N1,nu1,nu2)
             a += self.getGravAcc(pos, M1, M2, N1, G)
          else:
             # Orbital gravity only during binary evolution
             a += self.getGravAcc(pos, M1, M2, N1, G) 
          
          return a
          
      def getSurface(self, pos, cval, dis):    
          sorted_indices = np.argsort(pos[:, 2])
          dis_sq = dis**2
          
          accepted = np.zeros((len(pos), 2))
          accepted_count = 0
          cv = []
          for s in sorted_indices:
              p = pos[s, :2]
              if accepted_count > 0:
                 acc_dist = np.sum((accepted[:accepted_count] - p)**2, axis=1)
                 if np.any(acc_dist < dis_sq):
                    continue
              
              accepted[accepted_count] = p
              accepted_count += 1
              cv.append(cval[s])             
          return accepted[:accepted_count, 0], accepted[:accepted_count, 1], np.asarray(cv)
          
      def sqVisi(self, u, v, x, y, d, lam, cv):

          theta_x = x / d
          theta_y = y / d

          u_lam = u / lam
          v_lam = v / lam

          V = np.zeros_like(u_lam, dtype=np.complex128)

          norm = np.sum(cv)

          for xi, yi, ci in zip(theta_x, theta_y, cv):
              phase = np.exp(-2j*np.pi*(u_lam*xi + v_lam*yi))
              V += ci * phase

          V /= norm

          return np.abs(V)**2
  

