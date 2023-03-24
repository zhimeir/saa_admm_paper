from pylab import *
ion()
import scipy
from scipy import ndimage
gf = ndimage.gaussian_filter

im1 = gf(np.load("UWDRO1_slice40_activity.npy"),0.0)[::2,::2]
im2 = gf(np.load("UWDRO1_slice40_attenuation.npy"),0.0)[::2,::2]

vmm1= 0.0
vmx1= 40000
vmm2= 0.075
vmx2= 0.115
figure(1,figsize=(8,4))
subplot(121)
imshow(im1[24:104,24:104],vmin = vmm1, vmax = vmx1, cmap = cm.gray, interpolation = "nearest")
xticks([])
yticks([])
subplot(122)
imshow(im2[24:104,24:104],vmin = vmm2, vmax = vmx2, cmap = cm.gray, interpolation = "nearest")
xticks([])
yticks([])
subplots_adjust(left = 0, bottom = 0,top = 1,right= 1,wspace =0.01)
#savefig("figs/phantom.png")
