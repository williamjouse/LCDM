from getdist import plots
from getdist import loadMCSamples
import pylab as plt
import matplotlib
matplotlib.use('TkAgg')

plt.rcParams['text.usetex']=True




chains1 = loadMCSamples('chains/SN/cumtrapz-run')
chains2 = loadMCSamples('chains/SN/f2py-run')
chains3 = loadMCSamples('chains/SN/quad-run')

for chains in [chains1, chains2, chains3]:
    p = chains.getParams()
    chains.addDerived((p.Obh2 + p.Och2)/(p.H0/100.)**2, name='Om', label='\Omega_m')


g = plots.getSinglePlotter()
g.triangle_plot([chains1, chains2, chains3], ['H0', 'Om'], ['cumtrapz', 'F2py', 'quad'], filled=[True])
g.export('figures/SN-conf.pdf')

for chains in [chains11, chains22, chains33]:
	result = chains.getTable(limit=1).tableTex()
	print('SNeIa results', result)

chains11 = loadMCSamples('chains/BAO/cumtrapz-run')
chains22 = loadMCSamples('chains/BAO/f2py-run')
chains33 = loadMCSamples('chains/BAO/quad-run')

for chains in [chains11, chains22, chains33]:
    p = chains.getParams()
    chains.addDerived((p.Obh2 + p.Och2)/(p.H0/100.)**2, name='Om', label='\Omega_m')


g = plots.getSinglePlotter()
g.triangle_plot([chains11, chains22, chains33], ['H0', 'Om'], ['cumtrapz', 'F2py', 'quad'], filled=[True])
g.export('figures/BAO-conf.pdf')

plt.show()

for chains in [chains11, chains22, chains33]:
	result = chains.getTable(limit=1).tableTex()
	print('BAO results', result)


