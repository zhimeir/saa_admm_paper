from pylab import *
ion()
im0 = np.load("UWDRO1_slice40_activity.npy")[::2,::2]*1.
#im0 *= 1.e6/im0.sum()

sim = im0.sum()

storeiterations=array([1,2,3,4,5,6,8,10,12,14,16,18,20,25,30,35,40,45,50,60,70,80,90,100])
subiter = [0,1,4,7,12,18,23]

im1 = np.load("saaresults9cm_noisy/lambdamean_TVmu100_TVlam100.npy")
istd1 = np.load("saaresults9cm_noisy/lambdastd_TVmu100_TVlam100.npy")


im = np.load("results9cm_noisy/lambdamean.npy")
istd = np.load("results9cm_noisy/lambdastd.npy")

data1 = []
for i in range(len(im1)):
   nmean = sqrt( ((im1[i]-im0)**2).sum() )/sqrt( (im0**2).sum() )
   nstd =  sqrt((istd1[i]**2).sum()) /sqrt( (im0**2).sum() )
   data1.append([nmean,nstd])

data = []
for i in range(len(im)):
   nmean = sqrt( ((im[i]-im0)**2).sum() )/sqrt( (im0**2).sum() )
   nstd =  sqrt((istd[i]**2).sum()) /sqrt( (im0**2).sum() )
   data.append([nmean,nstd])

data1 = array(data1)

data = array(data)[:7]

figure(1,figsize=(6,5))
plot(data1[:,0],data1[:,1],'b:',linewidth=2.0,label=r"ADMM-SAA ($\gamma_\lambda= 1.0$, $\gamma_\mu = 1.0$)")

plot(data[:,0],data[:,1],'k-',linewidth=2.0,label="MLAA")

legend(loc="best",fontsize=14)

subdata1 =data1[subiter]
tshifts1=array([[0.0,0.005],[0.0,0.005],[-0.02,0.005],[0.0,-0.015],[-0.04,-0.015],[0.008,-0.013],[-0.085,0.002]])
tshifts=array([[-0.02,-0.013],[-0.04,-0.01],[-0.04,-0.006],[0.005,0.005],[0.01,0.005],[0.015,0.002],[-0.09,-0.01]])
plot(subdata1[:,0],subdata1[:,1],'bo')

xlabel("normalized bias",fontsize=14)
xticks(fontsize=14)

ylabel("normalized std.",fontsize=14)
yticks(fontsize=14)

piter = storeiterations[subiter].tolist()
print(piter)
for i,txt in enumerate(piter):
   print(i,txt)
   annotate(str(txt),(subdata1[i,0]+tshifts1[i,0],subdata1[i,1]+tshifts1[i,1]),fontsize=14,color="b")

plot(data[:,0],data[:,1],'ko')
for i,txt in enumerate(piter):
   print(i,txt)
   annotate(str(txt),(data[i,0]+tshifts[i,0],data[i,1]+tshifts[i,1]),fontsize=14)

axis([0.0,1.0,-0.01,0.25])

subplots_adjust(left= 0.14,bottom= 0.12,top=0.95,right=0.95)

savefig("figs/bias_std.png",dpi=300)
