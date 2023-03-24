from pylab import *
ion()

figure(1,figsize=(6,4))
#mlaa data rmse
d1 = np.load("results9cm_noiseless/datarmse.npy")
#admm-saa data rmse
d2 = np.load("saaresults9cm_noiseless/datarmse.npy")
#admm-saa rho=1 data rmse
d3 = np.load("saaresults9cm_noiseless/datarmse11.npy")

loglog(d1,"k-",linewidth=2.0,label="MLAA")
loglog(d2,"b--",linewidth=2.0,label=r"ADMM-SAA ($\rho_\lambda=0.01$, $\rho_\mu=100$)")
loglog(d3,"r-.",linewidth=2.0,label=r"ADMM-SAA ($\rho_\lambda=\rho_\mu=1$)")
xlabel("iteration number",fontsize=14)
ylabel("normalized data RMSE",fontsize=14)
xticks(fontsize=12)
yticks(fontsize=12)
axis([1,1000,0.001,5])

legend(loc="lower left",fontsize=12)
subplots_adjust(left=0.13,bottom=0.14,top=0.97,right=0.97)

savefig("figs/icrmsecomp.png",dpi=300)
