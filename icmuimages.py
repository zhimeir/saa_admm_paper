from pylab import *
ion()

im1 = np.load("saaresults9cm_noiseless/mu1000.npy")
im2 = np.load("saaresults9cm_noiseless/mu1000_TVmu100.npy")
im3 = np.load("saaresults9cm_noiseless/mu1000_TVlam100.npy")
im4 = np.load("saaresults9cm_noiseless/mu1000_TVmu100_TVlam100.npy")

vmm= 0.075
vmx= 0.115
figure(1,figsize=(8,8))
subplot(221)
imshow(im1[24:104,24:104],vmin = vmm, vmax = vmx, cmap = cm.gray, interpolation = "nearest")
xticks([])
yticks([])
subplot(222)
imshow(im2[24:104,24:104],vmin = vmm, vmax = vmx, cmap = cm.gray, interpolation = "nearest")
xticks([])
yticks([])
subplot(223)
imshow(im3[24:104,24:104],vmin = vmm, vmax = vmx, cmap = cm.gray, interpolation = "nearest")
xticks([])
yticks([])
subplot(224)
imshow(im4[24:104,24:104],vmin = vmm, vmax = vmx, cmap = cm.gray, interpolation = "nearest")
xticks([])
yticks([])
subplots_adjust(left = 0, bottom = 0,top = 1,right= 1,wspace =0.01,hspace = 0.01)
savefig("figs/attimagesic.png")
