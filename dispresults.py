from pylab import *
ion()

tlam = np.load("UWDRO1_slice40_activity.npy")[::2,::2]
tmu = np.load("UWDRO1_slice40_attenuation.npy")[::2,::2]

imlam10 = np.load("results10cm/lambda500.npy")
immu10 = np.load("results10cm/mu500.npy")

imlam10n = np.load("results10cm_noisy/lambda10.npy")
immu10n = np.load("results10cm_noisy/mu10.npy")


lmin = -35000.0
lmax = 0.
mmin=0.05
mmax=0.12
figure(1,figsize=(9,6))
subplot(2,3,1)
imshow(-tlam[12:100,20:128-20],cmap = cm.gray,interpolation="nearest",vmin=lmin,vmax=lmax)
xticks([])
yticks([])
text(10,15,"Truth",fontsize=14)
subplot(2,3,4)
imshow(tmu[12:100,20:128-20],cmap = cm.gray,interpolation="nearest",vmin=mmin,vmax=mmax)
xticks([])
yticks([])

subplot(2,3,2)
imshow(-imlam10[12:100,20:128-20],cmap = cm.gray,interpolation="nearest",vmin=lmin,vmax=lmax)
xticks([])
yticks([])
text(10,15,"FWHM = 10 cm",fontsize=14)
subplot(2,3,5)
imshow(immu10[12:100,20:128-20],cmap = cm.gray,interpolation="nearest",vmin=mmin,vmax=mmax)
xticks([])
yticks([])

subplot(2,3,3)
imshow(-imlam10n[12:100,20:128-20],cmap = cm.gray,interpolation="nearest",vmin=lmin,vmax=lmax)
xticks([])
yticks([])
text(10,15,"FWHM = 10 cm",fontsize=14)
text(10,25,"noisy",fontsize=14)
subplot(2,3,6)
imshow(immu10n[12:100,20:128-20],cmap = cm.gray,interpolation="nearest",vmin=mmin,vmax=mmax)
xticks([])
yticks([])

subplots_adjust(left=0,right=1,top=1,bottom=0,wspace=0.005,hspace=0.005)

figure(2)

im10 = np.load("results10cm_noisy/lambda10.npy")
im20 = np.load("results10cm_noisy/lambda20.npy")
im50 = np.load("results10cm_noisy/lambda50.npy")

plot(tlam[64],label="true")
plot(im10[64],label="10 iter")
plot(im20[64],label="20 iter")
plot(im50[64],label="50 iter")
title("activity profile, FWHM = 10cm, noisy")
legend(loc="best")

