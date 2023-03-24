from pylab import *
ion()

figure(1,figsize=(6,4))
#admm-saa data rmse
d1 = np.load("saaresults9cm_noiseless/imagelrmse.npy")
#admm-saa mu TV
d2 = np.load("saaresults9cm_noiseless/imagelrmse_TVmu100.npy")
#admm-saa lambda TV
d3 = np.load("saaresults9cm_noiseless/imagelrmse_TVlam100.npy")
#admm-saa both TV consts.
d4 = np.load("saaresults9cm_noiseless/imagelrmse_TVmu100_TVlam100.npy")

loglog(d1,"k-",linewidth=2.0,label="no TV constraint")
loglog(d2,"b--",linewidth=2.0,label="attenuation TV constraint")
loglog(d3,"r-.",linewidth=2.0,label="activity TV constraint")
loglog(d4,"g:",linewidth=2.0,label="both TV constraints")
xlabel("iteration number",fontsize=14)
ylabel("normalized activity RMSE",fontsize=14)
xticks(fontsize=12)
yticks(fontsize=12)
axis([1,1000,0.001,1])

legend(loc="lower left",fontsize=12)
subplots_adjust(left=0.13,bottom=0.14,top=0.97,right=0.97)

savefig("figs/iclrmsecompTV.png",dpi=300)
