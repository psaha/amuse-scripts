import scipy.optimize as sciopt
import numpy as np

class POSVEL():
          
      def parameter(self, fname, Nobj, *obj):
           par =  "#M (Msun), P (days), a (au), e, I (deg), Omega (deg), epoch (days), omega (deg)"
           f = open(fname, "w")
           f.write(par)                                           
           f.write('\n')     
           for i in range(Nobj):
               for j in obj[i]:
                   f.write(j + str('\t')) 
               f.write('\n')              
           f.close()
           
      def baryCen(self, m, x, N):
           """       
           """
           cm = 0*m
           cm[0] = m[0]                                                            
           for n in range(1,N):
               cm[n] = cm[n-1] + m[n]                                              
           cx = 0*x                                                                
           for n in range(1,N):
               x[n] += cx[n-1]                                                     
               cx[n] = (cm[n-1]*cx[n-1] + m[n]*x[n])/cm[n];                        
           b = cx[N-1]                                                            
           x[:] -= b                                                               
           return x
           
      def rotate(self, x, y, omega, I, Omega):             
           cs, sn = np.cos(omega), np.sin(omega)
           x, y, z = x*cs - y*sn, x*sn + y*cs, 0
           cs, sn = np.cos(I), np.sin(I)
           x,y,z = x, y*cs - z*sn, y*sn + z*cs
           Omega += np.pi/2                                                           # rotate (north, east)
           cs,sn = np.cos(Omega), np.sin(Omega)
           x,y,z = x*cs - y*sn, x*sn + y*cs, z
           return np.array((x,y,z))
           
      def posvel(self, P, a, e, omega, I, Omega, mean_an):
           ec = (1-e*e)**.5
           mean_an = mean_an % (2*np.pi)                                              
           psi = sciopt.brentq(lambda psi: psi - e*np.sin(psi) - mean_an, 0, 2*np.pi)    
           cs,sn = np.cos(psi), np.sin(psi)
           x,y = cs-e, ec*sn                                                       
           vx,vy = -sn/(1-e*cs), ec*cs/(1-e*cs)                                   
           pos = a * self.rotate(x,y,omega,I,Omega)                                     
           vel = 2*np.pi * a/P * self.rotate(vx,vy,omega,I,Omega)                          
           return pos, vel
           
      def getposvel(self, fname, now, N):                               
           fil = open(fname)
           fil.readline()
           star = []
           for n in range(N):
               pars = fil.readline().split()
               pars = [float(s) for s in pars]                                    
               star.append(pars)                                                   
           c = 299792458
           days = 86400
           au = 149597870700 / c
           deg = np.pi/180            
           mass = np.zeros(N)
           pos = np.zeros((N,3))      
           vel = np.zeros((N,3))           
           for n in range(N):
               mass[n] = float(star[n][0])                                 
           for n in range(1,N): 
               P,a,e,I,Omega,ep,omega = star[n][1:]                                
               mean_an = 2*np.pi * (now-ep)/P                                         
               P *= days                                                           
               a *= au                                                             
               I *= deg                                                            
               Omega *= deg                                                        
               omega *= deg                                                        
               pos[n],vel[n] = self.posvel(P,a,e,omega,I,Omega,mean_an)                 
           return mass, self.baryCen(mass,pos,N), self.baryCen(mass,vel,N)
           
