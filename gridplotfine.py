from pylab import *
ion()

d = log(np.load("dgridfine.npy"))/log(10.0)

figure(1,figsize=(6,6))
imshow(d,vmin = -1.0,vmax=1.0,interpolation="nearest",cmap = cm.gray)
xtlabels=["0.001","0.002","0.005","0.01","0.02","0.05","0.1","0.2","0.5"]
xtvals = range(9)
ytlabels=["10","20","50","100","200","500","1000","2000","5000"]
ytvals = range(9)
xticks(xtvals,xtlabels,fontsize=14,rotation=15)
yticks(ytvals,ytlabels,fontsize=14)

xlabel(r"step size ratio $\rho_\lambda$",fontsize=14)
ylabel(r"step size ratio $\rho_\mu$",fontsize=14)
title("Data RMSE (fine search grid)",fontsize=14)

subplots_adjust(left=0.16,bottom=0.11,top=0.95,right=0.97)
savefig("figs/finegrid.png",dpi=300)
