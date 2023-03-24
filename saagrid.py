from tofsetup import *

print("Ntof: ",ntof)
input("hi")

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


niter = 101

newtiter = 10
nuv = 100

storeiterations=[1,2,5,10,20,50,100,200,500]

ltotalcounts = testlambda.sum()
print("Total Counts: ",ltotalcounts)

dgrid=[]
#sbmu = sblambda*mulamratio
for sbmu in [10.,20.,50.,100.,200.,500.,1000.,2000.,5000.]:
#for sbmu in [1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4]:
   dgridrow = []
   for sblambda in [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5]:
#   for sblambda in [1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4]:
      sigmu = sbmu/lproj
      taumu = 1.0/(sbmu*lproj)

      siglambda = sblambda/ltofproj
      taulambda = 1.0/(sblambda*ltofproj)
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

      datarmses = []
      for j in range(niter):
# x-update
         TOFprojection(imlambda,worktofsino)
         arg1 = ulambda+siglambda*(worktofsino -ylambda)
         TOFbackProjection(arg1,wimlambda)
#   print("wimlambda: ",sqrt( (wimlambda**2.).sum())," ",sqrt( ((wimlambda-(wimlambda.sum()/fovmask.sum()))**2.).sum()))

# counts constraint block
         nu  = ((imlambda.sum() -ltotalcounts)/taulambda - wimlambda.sum())/fovmask.sum()
#   nu = 0.

#   print("tc 1 :",imlambda.sum(), "true: ",ltotalcounts)
         imlambda =imlambda -taulambda*(nu*fovmask + wimlambda)
#         imlambda[imlambda<0.]=0.
         TOFprojection(imlambda,worktofsino)

#   print("tc  2:",imlambda.sum(), "true: ",ltotalcounts)
#   input("hi")

         circularParallelBeamProjection(immu,worksino)
         arg1 = umu +sigmu*( worksino -ymu )
         circularParallelBeamBackProjection(arg1,wimmu)
#   print("wimmu: ",sqrt( (wimmu**2.).sum()))
         immu =immu - wimmu*taumu
#         immu[immu<0.]=0.
         immu *= mumask
         circularParallelBeamProjection(immu,worksino)

         ylambdat = 1.*ylambda
         ymut = 1.*ymu
# y-update
#         aq = 1.0 + siglambda
         aq = siglambda #biconvex
         cq = -tofdata
         for k in range(nuv):
#            ylambdat = 1.*ylambda
#            ymut = 1.*ymu
            ymuterm = ones([ntof])[:,newaxis,newaxis]*qexp(-ymu)
#      ymuterm = ones([ntof])[:,newaxis,newaxis]*exp(-ymu)
#            bq = ymuterm - ylambdat - ulambda - siglambda*worktofsino #  ylambdat or tofsino for test
            bq = ymuterm - ulambda - siglambda*worktofsino # biconvex
#            bq = ymuterm - tofsino - ulambda - siglambda*worktofsino #  ylambdat or tofsino for test
            ylambdaold = ylambda*1.
            ylambda = (-bq + sqrt(bq*bq - 4.*aq*cq))/(2.*aq)
            ylambdaold = ylambda*1.

            ymunewt = ymu*0.
            for i in range(newtiter):
#               grad = (-qexp(-ymunewt,1)*ylambdaold + tofdata).sum(axis=0) + ntof*(hfunc(ymunewt,1)-hfunc(ymut,1)) - umu + sigmu*(ymunewt - worksino)
               grad = (-qexp(-ymunewt,1)*ylambdaold + tofdata).sum(axis=0) - umu + sigmu*(ymunewt - worksino) #biconvex
#         grad = (-exp(-ymunewt)*ylambdaold + tofdata).sum(axis=0) - umu + sigmu*(ymunewt - worksino)
#               grad = (-qexp(-ymunewt,1)*ylambdaold + tofdata).sum(axis=0) + ntof*(hfunc(ymunewt,1)-hfunc(sinoatt,1)) - umu + sigmu*(ymunewt - worksino)
               ymuterm = ylambdaold*qexp(-ymunewt,2)
#         ymuterm = ylambdaold*exp(-ymunewt)
#               hess = ymuterm.sum(axis=0) + ntof*hfunc(ymunewt,2) + sigmu
               hess = ymuterm.sum(axis=0) + sigmu #biconvex
#               hessmin = minimum(hessmin,hess.min())
               ymunewt = ymunewt - grad/hess
#         print(i," grad mag: ",sqrt((grad**2).sum()))
            ymuold = ymu*1.
            ymu = ymunewt*1.
            ymu[ymu<0.] = 0.
            ylambdadist = sqrt( ( (ylambda-ylambdaold)**2).sum())
            ymudist = sqrt( ( (ymu-ymuold)**2).sum())
#      print("ylambda change: ",ylambdadist, "ymu change: ",ymudist)
   
# u-update
#   input("hi")
         TOFprojection(imlambda,worktofsino)
         ulambda = ulambda + siglambda*(worktofsino - ylambda)

         circularParallelBeamProjection(immu,worksino)
         umu = umu +sigmu*(worksino-ymu)

         datarmse = sqrt(( (tofdata-worktofsino*exp(-worksino))**2. ).sum())/sqrt(1.*ntof*nviews*nbins)
         imagelrmse = sqrt(( (testlambda-imlambda)**2. ).sum())/sqrt(1.*nx*ny)
         imagemrmse = sqrt(( (testmu-immu)**2. ).sum())/sqrt(1.*nx*ny)
         datarmses.append(datarmse)
#   input("step")

         if j+1 in storeiterations:
            print("iter: ",j+1," ",datarmse," ",imagelrmse," ",imagemrmse," ",ymu.min())
#            save(saaresultsfile+"lambda"+str(j+1)+".npy",imlambda/datafactor) #save the images with datafactor removed in order to be able to compare with phantom
#            save(saaresultsfile+"mu"+str(j+1)+".npy",immu)
      dgridrow.append(datarmse)
   dgrid.append(dgridrow)
#save(saaresultsfile+"datarmse.npy",array(datarmses))
