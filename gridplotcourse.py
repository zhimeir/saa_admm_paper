from pylab import *
ion()

d = log(np.load("dgridmupcourse.npy"))/log(10.0)

figure(1,figsize=(6,6))
imshow(d,vmin = -1.0,vmax=2.0,interpolation="nearest",cmap = cm.gray)
xtlabels=[r"$10^{-4}$",r"$10^{-3}$",r"$10^{-2}$",r"$10^{-1}$",r"$1$",r"$10^{1}$",r"$10^{2}$",r"$10^{3}$",r"$10^{4}$"]
xtlabels=[r"$10^{-4}$","",r"$10^{-2}$","",r"$1$","",r"$10^{2}$","",r"$10^{4}$"]
xtvals = range(9)
ytlabels=[r"$10^{-4}$",r"$10^{-3}$",r"$10^{-2}$",r"$10^{-1}$",r"$1$",r"$10^{1}$",r"$10^{2}$",r"$10^{3}$",r"$10^{4}$"]
ytlabels=[r"$10^{-4}$","",r"$10^{-2}$","",r"$1$","",r"$10^{2}$","",r"$10^{4}$"]
ytvals = range(9)
xticks(xtvals,xtlabels,fontsize=14)
yticks(ytvals,ytlabels,fontsize=14)

xlabel(r"step size ratio $\rho_\lambda$",fontsize=14)
ylabel(r"step size ratio $\rho_\mu$",fontsize=14)
title("Data RMSE (course search grid)",fontsize=14)

subplots_adjust(left=0.15,bottom=0.08,top=0.95,right=0.97)

savefig("figs/coursegrid.png",dpi=300)
