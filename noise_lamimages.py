from pylab import *
ion()

im1 = np.load("saaresults9cm_noisy/lambda10_TVmu100_TVlam100.npy")/32768.0
im2 = np.load("saaresults9cm_noisy/lambda20_TVmu100_TVlam100.npy")/32768.0
im3 = np.load("saaresults9cm_noisy/lambda50_TVmu100_TVlam100.npy")/32768.0
im4 = np.load("saaresults9cm_noisy/lambda100_TVmu100_TVlam100.npy")/32768.0

mim1 = np.load("results9cm_noisy/lambda10.npy")/32768.0
mim2 = np.load("results9cm_noisy/lambda20.npy")/32768.0
mim3 = np.load("results9cm_noisy/lambda50.npy")/32768.0
mim4 = np.load("results9cm_noisy/lambda100.npy")/32768.0

vmm= 0.0
vmx= 1.1
figure(1,figsize=(4,8))
subplot(4,2,1)
imshow(mim1[24:104,24:104],vmin = vmm, vmax = vmx, cmap = cm.gray, interpolation = "nearest")
xticks([])
yticks([])
subplot(4,2,2)
imshow(im1[24:104,24:104],vmin = vmm, vmax = vmx, cmap = cm.gray, interpolation = "nearest")
xticks([])
yticks([])
subplot(4,2,3)
imshow(mim2[24:104,24:104],vmin = vmm, vmax = vmx, cmap = cm.gray, interpolation = "nearest")
xticks([])
yticks([])
subplot(4,2,4)
imshow(im2[24:104,24:104],vmin = vmm, vmax = vmx, cmap = cm.gray, interpolation = "nearest")
xticks([])
yticks([])
subplot(4,2,5)
imshow(mim3[24:104,24:104],vmin = vmm, vmax = vmx, cmap = cm.gray, interpolation = "nearest")
xticks([])
yticks([])
subplot(4,2,6)
imshow(im3[24:104,24:104],vmin = vmm, vmax = vmx, cmap = cm.gray, interpolation = "nearest")
xticks([])
yticks([])
subplot(4,2,7)
imshow(mim4[24:104,24:104],vmin = vmm, vmax = vmx, cmap = cm.gray, interpolation = "nearest")
xticks([])
yticks([])
subplot(4,2,8)
imshow(im4[24:104,24:104],vmin = vmm, vmax = vmx, cmap = cm.gray, interpolation = "nearest")
xticks([])
yticks([])
subplots_adjust(left = 0, bottom = 0,top = 1,right= 1,wspace =0.01,hspace = 0.01)
savefig("figs/actimages.png",dpi=300)
