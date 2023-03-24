from tofsetup import *
small = 1.e-5


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

sblambda = 1.0e5
mulamratio=100.0
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
   aq = siglambda
   cq = -tofdata
   hessmin = 1.e10
   hessmin2 = 1.e10
   ymunewt = ymu*1.
   ylambdanewt = ylambda*1.
   for i in range(newtiter):
      gradlambda = (ylambdanewt)*qexp(-ymunewt) \
          -tofdata +(ylambdanewt)*( -ulambda +siglambda*(ylambdanewt-worktofsino))
      gradmu = (-qexp(-ymunewt,1)*ylambdanewt + tofdata).sum(axis=0) - umu + sigmu*(ymunewt - worksino)

      hesslambda2 = tofdata/maximum(abs(ylambdanewt),small) +siglambda*ylambdanewt
      hessmu2= (ylambdanewt*qexp(-ymunewt,2)).sum(axis=0) + sigmu
      hesslambdamub= -(ylambdanewt)*qexp(-ymunewt,1) 
      hesslambdamuc= -ones([ntof])[:,newaxis,newaxis]*qexp(-ymunewt,1) 

      cab = ((hesslambdamuc*hesslambdamub)/hesslambda2).sum(axis=0)
      cau = (gradlambda*hesslambdamuc/hesslambda2).sum(axis=0) 
      
      delmu = (gradmu-cau)/(hessmu2 -cab)
      dellambda = (gradlambda - hesslambdamub*delmu)/hesslambda2


      ylambdanewt = ylambdanewt - dellambda
      ymunewt = ymunewt - delmu
      print(i," grad lambda mag: ",sqrt((gradlambda**2).sum())," grad mu mag: ",sqrt((gradmu**2).sum()))

   input("hi")
   ylambda = ylambdanewt*1.
   ymu = ymunewt*1.
#   ymu[ymu<0.] = 0.
   
# u-update
   TOFprojection(imlambda,worktofsino)
   ulambda = ulambda + siglambda*(worktofsino - ylambda)

   circularParallelBeamProjection(immu,worksino)
   umu = umu +sigmu*(worksino-ymu)

   datarmse = sqrt(( (tofsino-worktofsino)**2. ).sum())/sqrt(1.*ntof*nviews*nbins)
   imagelrmse = sqrt(( (testlambda-imlambda)**2. ).sum())/sqrt(1.*nx*ny)
   imagemrmse = sqrt(( (testmu-immu)**2. ).sum())/sqrt(1.*nx*ny)
   datarmses.append(datarmse)
   print("iter: ",j," ",datarmse," ",imagelrmse," ",imagemrmse," ",hessmin," ",hessmin2)
#   input("step")

   if j+1 in storeiterations:
      save(saaresultsfile+"lambda"+str(j+1)+".npy",imlambda/datafactor) #save the images with datafactor removed in order to be able to compare with phantom
      save(saaresultsfile+"mu"+str(j+1)+".npy",immu)

save(saaresultsfile+"datarmse.npy",array(datarmses))
