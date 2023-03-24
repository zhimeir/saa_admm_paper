from pylab import *
ion()
im0 = np.load("UWDRO1_slice40_activity.npy")[::2,::2]*1.
#im0 *= 1.e6/im0.sum()

sim = im0.sum()

storeiterations=array([1,2,3,4,5,6,8,10,12,14,16,18,20,25,30,35,40,45,50,60,70,80,90,100])
subiter = [0,1,4,7,12,18,23]
subiter = [7,12,18,23]

im1 = np.load("saaresults9cm_noisy/lambdamean_TVmu100_TVlam100.npy")
istd1 = np.load("saaresults9cm_noisy/lambdastd_TVmu100_TVlam100.npy")
im2 = np.load("saaresults9cm_noisy/lambdamean_TVmu100.npy")
istd2 = np.load("saaresults9cm_noisy/lambdastd_TVmu100.npy")
im3 = np.load("saaresults9cm_noisy/lambdamean_TVlam100.npy")
istd3 = np.load("saaresults9cm_noisy/lambdastd_TVlam100.npy")


data1 = []
data2 = []
data3 = []
for i in range(len(im1)):
   nmean = sqrt( ((im1[i]-im0)**2).sum() )/sqrt( (im0**2).sum() )
   nstd =  sqrt((istd1[i]**2).sum()) /sqrt( (im0**2).sum() )
   data1.append([nmean,nstd])
   nmean = sqrt( ((im2[i]-im0)**2).sum() )/sqrt( (im0**2).sum() )
   nstd =  sqrt( (istd2[i]**2).sum()) /sqrt( (im0**2).sum() )
   data2.append([nmean,nstd])
   nmean = sqrt( ((im3[i]-im0)**2).sum() )/sqrt( (im0**2).sum() )
   nstd =  sqrt( (istd3[i]**2).sum()) /sqrt( (im0**2).sum() )
   data3.append([nmean,nstd])


data1 = array(data1)
data2 = array(data2)
data3 = array(data3)


figure(1,figsize=(6,5))
plot(data1[7:,0],data1[7:,1],'b-',linewidth=2.0,label="att. and act. TV constraints")
plot(data2[7:,0],data2[7:,1],'g--',linewidth=2.0,label="att. TV constraint only")
plot(data3[7:,0],data3[7:,1],'r:',linewidth=2.0,label="act. TV constraint only")


legend(loc="best",fontsize=14)

subdata1 =data1[subiter]
subdata2 =data2[subiter]
subdata3 =data3[subiter]
tshifts1=array([[0.0,0.005],[-0.02,0.01],[-0.01,0.01],[-0.048,-0.01]])
plot(subdata1[:,0],subdata1[:,1],'bo')
plot(subdata2[:,0],subdata2[:,1],'go')
plot(subdata3[:,0],subdata3[:,1],'rv')

xlabel("normalized bias",fontsize=14)
xticks(fontsize=14)

ylabel("normalized std.",fontsize=14)
yticks(fontsize=14)

piter = storeiterations[subiter].tolist()
print(piter)
for i,txt in enumerate(piter):
   print(i,txt)
   annotate(str(txt),(subdata1[i,0]+tshifts1[i,0],subdata1[i,1]+tshifts1[i,1]),fontsize=14,color="b")


axis([0.0,0.52,0.0,0.5])

subplots_adjust(left= 0.14,bottom= 0.12,top=0.95,right=0.95)

savefig("figs/bias_std_admm.png",dpi=300)
