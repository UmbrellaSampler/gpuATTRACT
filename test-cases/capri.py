import sys
import numpy as np

name = sys.argv[1]
template = 'out_'+name+'-sorted-dr'
lrmsd = np.loadtxt(template+'.lrmsd',unpack=True, usecols=[1])
irmsd = np.loadtxt(template+'.irmsd',unpack=True, usecols=[1])
fnat = np.loadtxt(template+'.fnat',unpack=True, usecols=[0])


onestar, twostar, threestar = 0,0,0
i1, i2, i3 = ([] for i in range(3))
for i in range(len(irmsd)):
  if (irmsd[i] <= 1.0 or lrmsd[i] <= 1.0) and fnat[i] >= 0.5:
    threestar += 1
    i3.append(i)
  elif (irmsd[i] <= 2.0 or lrmsd[i] <= 5.0) and fnat[i] >= 0.3:
    twostar += 1
    i2.append(i)   
  elif (irmsd[i] <= 4.0 or lrmsd[i] <= 10.0) and fnat[i] >= 0.3:
    onestar += 1
    i1.append(i)
    
print "***:",threestar,i3,"**:",twostar,i2,"*:",onestar,i1,"Best IRMSD:",np.amin(irmsd),"Best LRMSD:",np.amin(lrmsd),"Best fnat:",np.amax(fnat)
