from tofsetup import *


sinoatt = sinogram.copy()
sinoatt.fill(0.)
print("Starting projection ...")
t1 = time.time()
circularParallelBeamProjection(testmu,sinoatt)
print("Projection took ",time.time()-t1," seconds.")
attfactor = exp(-sinoatt)

print("Starting fast TOF projection ...")
tofsino = zeros([ntof,nviews,nbins])
t1 = time.time()
TOFprojection(testlambda,tofsino)
print("Full Fast projection took ",time.time()-t1," seconds.")


truess = tofsino.sum(axis=0)

tofdata = attfactor*tofsino
#tofdata[tofdata<small] = small

datafactor = totalcounts/tofdata.sum() #scale the data to the desired count level
print("datafactor: ",datafactor)
tofdata *= datafactor
tofsino *= datafactor
testlambda  *= datafactor
truess *= datafactor
#tofdata = random.poisson(tofdata).astype("float64") #UNCOMMENT for noise and COMMENT out for no noise
print("tofdata: ",tofdata.min(),tofdata.max())
print("tofsum: ",tofdata.sum())



ltotalcounts = testlambda.sum()
print("Total Counts: ",ltotalcounts)

sblambda = 1.0
#sblambda = 1.0
mulamratio=1.0e5
sblambda =0.02
mulamratio = 2.5e4
#sblambda = 0.05
#mulamratio=2.0e4
#sblambda = 0.02
#mulamratio=0.6e5
shrink = 1.0
#sbmu = 7500.0
sbmu = sblambda*mulamratio
sigmu = shrink*sbmu/lproj
taumu = shrink*1.0/(sbmu*lproj)

siglambda = shrink*sblambda/ltofproj
taulambda = shrink*1.0/(sblambda*ltofproj)
print("sblambda: ",sblambda," sbmu: ",sbmu)

worktofsino = 0.*tofsino
ulambda = 0.*tofsino
ylambda = 0.*tofsino

imlambda = testlambda*0.
wimlambda = testlambda*0.

worksino = 0.*sinoatt
umu = 0.*sinoatt
ymu = 0.*sinoatt

immu = testmu*0.
wimmu = testmu*0.

niter = 101

newtiter = 20

datarmses = []
storeiterations=[1,2,5,10,20,50,100,200,500]
hessmin = 1.e10
recymu = 0
for j in range(niter):
# x-update
   TOFprojection(imlambda,worktofsino)
   arg1 = ulambda+siglambda*(worktofsino -ylambda)
   TOFbackProjection(arg1,wimlambda)

# counts constraint block
   nu  = ((imlambda.sum() -ltotalcounts)/taulambda - wimlambda.sum())/fovmask.sum()
#   nu = 0.

#   print("tc 1 :",imlambda.sum(), "true: ",ltotalcounts)
   imlambda =imlambda -taulambda*(nu*fovmask + wimlambda)
   TOFprojection(imlambda,worktofsino)

#   print("tc  2:",imlambda.sum(), "true: ",ltotalcounts)
#   input("hi")

   circularParallelBeamProjection(immu,worksino)
   arg1 = umu +sigmu*( worksino -ymu )
   circularParallelBeamBackProjection(arg1,wimmu)
   immu =immu - wimmu*taumu
   immu *= mumask
   circularParallelBeamProjection(immu,worksino)

   ylambdat = 1.*ylambda
   ymut = 1.*ymu
# y-update
#   aq = 1.0 + siglambda
   aq = siglambda
   cq = -tofdata
   ymunewt = ymu*0.
   ymunewttrack=[]
   gradtrack=[]
   hesstrack=[]
   hessmask = ymu*0.
   hessmask.fill(1.)
   for i in range(newtiter):
      ymuterm = ones([ntof])[:,newaxis,newaxis]*qexp(-ymunewt)
#      ymuterm = ones([ntof])[:,newaxis,newaxis]*exp(-ymunewt)

#      bq = ymuterm - ylambdat - ulambda - siglambda*worktofsino #  ylambdat or tofsino for test
      bq = ymuterm - ulambda - siglambda*worktofsino #  ylambdat or tofsino for test
#      bq = ymuterm - tofsino - ulambda - siglambda*worktofsino #  ylambdat or tofsino for test

      ylambda = (-bq + sqrt(bq*bq - 4.*aq*cq))/(2.*aq)

      ymutermprime = -ones([ntof])[:,newaxis,newaxis]*qexp(-ymunewt,1)
#      ymutermprime = -ones([ntof])[:,newaxis,newaxis]*exp(-ymunewt)

      ylambdaprime = ymutermprime*(-1 + bq/sqrt(bq*bq - 4.*aq*cq))/(2.*aq)

#      grad = (-qexp(-ymunewt,1)*ylambda + tofdata).sum(axis=0) + ntof*(hfunc(ymunewt,1)-hfunc(ymut,1)) - umu + sigmu*(ymunewt - worksino)
      grad = (-qexp(-ymunewt,1)*ylambda + tofdata).sum(axis=0) - umu + sigmu*(ymunewt - worksino)
#      grad = (-exp(-ymunewt)*ylambda + tofdata).sum(axis=0) - umu + sigmu*(ymunewt - worksino)
#      grad = (-qexp(-ymunewt,1)*ylambda + tofdata).sum(axis=0) + ntof*(hfunc(ymunewt,1)-hfunc(sinoatt,1)) - umu + sigmu*(ymunewt - worksino)
      ymuterm1 = ylambda*qexp(-ymunewt,2)
#      ymuterm1 = ylambda*exp(-ymunewt)
      ymuterm2 = ylambdaprime*qexp(-ymunewt,1)
#      ymuterm2 = ylambdaprime*exp(-ymunewt)

#      hess = ymuterm1.sum(axis=0) - ymuterm2.sum(axis=0) + ntof*hfunc(ymunewt,2) + sigmu
      hess = ymuterm1.sum(axis=0) - ymuterm2.sum(axis=0) + sigmu
#      hess = ymuterm1.sum(axis=0) + sigmu

      hessmask[hess<0.] = 0.
      if hess.min() < hessmin:
         hessmin = hess.min()
         rayindex = unravel_index(hess.argmin(),hess.shape)
         ulami = ulambda[:,rayindex[0],rayindex[1]]*1.
         tofi = tofdata[:,rayindex[0],rayindex[1]]*1.
         wtofsi = worktofsino[:,rayindex[0],rayindex[1]]*1.
         umui = umu[rayindex]*1.
         wsi = worksino[rayindex]*1.
         recymu = 1
         print(i," hessmin: ",ymunewt[rayindex])

     
      ymunewttrack.append(ymunewt*1.)
      gradtrack.append(grad*1.)
      hesstrack.append(hess*1.)

      ymunewt = ymunewt - grad/hess
#      ymunewt[ymunewt<0.] = 0.
#      print(i," grad mag: ",sqrt((grad**2).sum())," ymu min.: ",ymunewt.min())

#   input("hi")
   
#   ymu = ymunewt*1.
   print("hessmask size: ",(hessmask.sum()/(1.*nviews*nbins)))
   ymu[hessmask>0.5] = ymunewt[hessmask>0.5]*1.
   if recymu:
      ymuvals = array(ymunewttrack)[:,rayindex[0],rayindex[1]]
      gradvals = array(gradtrack)[:,rayindex[0],rayindex[1]]
      hessvals = array(hesstrack)[:,rayindex[0],rayindex[1]]
      recymu = 0
#   ymuterm = ones([ntof])[:,newaxis,newaxis]*qexp(-ymu)
#   bq = ymuterm - ulambda - siglambda*worktofsino #  ylambdat or tofsino for test
#   ylambda = (-bq + sqrt(bq*bq - 4.*aq*cq))/(2.*aq)
   
# u-update
   TOFprojection(imlambda,worktofsino)
   ulambda = ulambda + siglambda*(worktofsino - ylambda)

   circularParallelBeamProjection(immu,worksino)
   umu = umu +sigmu*(worksino-ymu)

   datarmse = sqrt(( (tofsino-worktofsino)**2. ).sum())/sqrt(1.*ntof*nviews*nbins)
   imagelrmse = sqrt(( (testlambda-imlambda)**2. ).sum())/sqrt(1.*nx*ny)
   imagemrmse = sqrt(( (testmu-immu)**2. ).sum())/sqrt(1.*nx*ny)
   datarmses.append(datarmse)
   print("iter: ",j," ",datarmse," ",imagelrmse," ",imagemrmse," ",hessmin)
#   input("step")

   if j+1 in storeiterations:
      save(saaresultsfile+"lambda"+str(j+1)+".npy",imlambda/datafactor) #save the images with datafactor removed in order to be able to compare with phantom
      save(saaresultsfile+"mu"+str(j+1)+".npy",immu)


print("ymuvals: ",ymuvals)
def ymupot(ymu):
   aq = siglambda
   cq = -tofi
   bq = qexps(-ymu) - ulami - siglambda*wtofsi
   ylambda = (-bq + sqrt(bq*bq - 4.*aq*cq))/(2.*aq)
   ylambdaprime = -qexps(-ymu)*(-1. + bq/sqrt(bq*bq - 4.*aq*cq))/(2.*aq)
   dphi = (-qexps(-ymu,1)*ylambda + tofi).sum() -umui + sigmu*(ymu-wsi)
   dphi2 = (qexps(-ymu,2)*ylambda -qexps(-ymu,1)*ylambdaprime).sum() + sigmu
   return dphi,dphi2

def ymupot2(ymu):
   aq = siglambda
   cq = -tofi
   bq = qexps(-ymu*0.) - ulami - siglambda*wtofsi
   ylambda = (-bq + sqrt(bq*bq - 4.*aq*cq))/(2.*aq)
   dphi = (-qexps(-ymu,1)*ylambda + tofi).sum() -umui + sigmu*(ymu-wsi)
   dphi2 = (qexps(-ymu,2)*ylambda ).sum() + sigmu
   return dphi,dphi2


save(saaresultsfile+"datarmse.npy",array(datarmses))
