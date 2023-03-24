from tofsetup import *

print("Ntof: ",ntof)
input("hi")

tvmuswitch = 0
tvlamswitch = 1

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

ptvmu = 100
tgx,tgy = gradim(testmu)
tvmu = numu*sqrt(tgx**2 + tgy**2).sum() *(1.*ptvmu/100.0)

ptvlam = 100
tgx,tgy = gradim(testlambda)
tvlambda = nulam*sqrt(tgx**2 + tgy**2).sum() *(1.*ptvlam/100.0)
print("TVs: ",tvmu,tvlambda)
tvstring = "_TVmu"+str(ptvmu)+"_TVlam"+str(ptvlam)


niter = 51

newtiter = 10
nuv = 100

storeiterations=[1,2,5,10,20,50,100,200,500]

ltotalcounts = testlambda.sum()
print("Total Counts: ",ltotalcounts)

dgrid=[]
#sbmu = sblambda*mulamratio
for sbmu in [10.,20.,50.,100.,200.,500.,1000.,2000.,5000.]:
#for sbmu in [100.,200.,500.]:
#for sbmu in [1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4]:
   dgridrow = []
   for sblambda in [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5]:
#   for sblambda in [0.005,0.01,0.02]:
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
      ugradlamx = testlambda*0.
      ygradlamx = testlambda*0.
      ugradlamy = testlambda*0.
      ygradlamy = testlambda*0.

      worksino = 0.*sinoatt
      umu = 0.*sinoatt
      ymu = 0.*sinoatt
      ugradmux = testlambda*0.
      ygradmux = testlambda*0.
      ugradmuy = testlambda*0.
      ygradmuy = testlambda*0.

      immu = testmu*0.
      wimmu = testmu*0.

      datarmses = []
      for j in range(niter):
# x-update
         TOFprojection(imlambda,worktofsino)
         arg1 = ulambda+siglambda*(worktofsino -ylambda)
         TOFbackProjection(arg1,wimlambda)
         tgx,tgy = gradim(imlambda)
         tgx *= nulam
         tgy *= nulam

         argx = ugradlamx + siglambda*(tgx - ygradlamx)
         argy = ugradlamy + siglambda*(tgy - ygradlamy)
         wgradimlam = mdiv(argx,argy)
         wgradimlam *= nulam*fovmask
         tvlamest  = sqrt(tgx**2 + tgy**2).sum()
#   print("wimlambda: ",sqrt( (wimlambda**2.).sum())," ",sqrt( ((wimlambda-(wimlambda.sum()/fovmask.sum()))**2.).sum()))

# counts constraint block
   #      nu  = ((imlambda.sum() -ltotalcounts)/taulambda - wimlambda.sum())/fovmask.sum()
         nu  = ((imlambda.sum() -ltotalcounts)/taulambda - wimlambda.sum() - wgradimlam)/fovmask.sum()
#   nu = 0.

#   print("tc 1 :",imlambda.sum(), "true: ",ltotalcounts)
        # imlambda =imlambda -taulambda*(nu*fovmask + wimlambda)
         imlambda =imlambda -taulambda*(nu*fovmask + wimlambda + wgradimlam)
#         imlambda[imlambda<0.]=0.
         TOFprojection(imlambda,worktofsino)
         workgradlamx, workgradlamy = gradim(imlambda)
         workgradlamx *= nulam
         workgradlamy *= nulam


#   print("tc  2:",imlambda.sum(), "true: ",ltotalcounts)
#   input("hi")

         circularParallelBeamProjection(immu,worksino)
         arg1 = umu +sigmu*( worksino -ymu )
         circularParallelBeamBackProjection(arg1,wimmu)
         tgx,tgy = gradim(immu)
         tgx *= numu
         tgy *= numu
         argx = ugradmux + sigmu*(tgx - ygradmux)
         argy = ugradmuy + sigmu*(tgy - ygradmuy)
         wgradimmu = mdiv(argx,argy)
         wgradimmu *= numu*fovmask
#   print("wimmu: ",sqrt( (wimmu**2.).sum()))
         immu =immu - wimmu*taumu - wgradimmu*taumu
#         immu[immu<0.]=0.
         immu *= mumask
         circularParallelBeamProjection(immu,worksino)
         workgradmux, workgradmuy = gradim(immu)
         workgradmux *= numu
         workgradmuy *= numu

         tvmuest  = sqrt(tgx**2 + tgy**2).sum()

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
   
         ptilx= ugradlamx/siglambda + workgradlamx
         ptily= ugradlamy/siglambda + workgradlamy
         ptilmag = sqrt(ptilx**2 + ptily**2)
         tempvex=(ptilmag).flatten()*1.
         if tvlamswitch:
            resvex = euclidean_proj_l1ball(tempvex, s=tvlambda)
         else:
            resvex = tempvex*1.
         resvex.shape = nx,ny
         ptilx[ptilmag>0]=ptilx[ptilmag>0]*resvex[ptilmag>0]/ptilmag[ptilmag>0]
         ptily[ptilmag>0]=ptily[ptilmag>0]*resvex[ptilmag>0]/ptilmag[ptilmag>0]
         ygradlamx = ptilx*1.
         ygradlamy = ptily*1.

         ptilx= ugradmux/sigmu + workgradmux
         ptily= ugradmuy/sigmu + workgradmuy
         ptilmag = sqrt(ptilx**2 + ptily**2)
         tempvex=(ptilmag).flatten()*1.
         if tvmuswitch:
            resvex = euclidean_proj_l1ball(tempvex, s=tvmu)
         else:
            resvex = tempvex*1.
         resvex.shape = nx,ny
         ptilx[ptilmag>0]=ptilx[ptilmag>0]*resvex[ptilmag>0]/ptilmag[ptilmag>0]
         ptily[ptilmag>0]=ptily[ptilmag>0]*resvex[ptilmag>0]/ptilmag[ptilmag>0]
         ygradmux = ptilx*1.
         ygradmuy = ptily*1.


# u-update
#   input("hi")
         TOFprojection(imlambda,worktofsino)
         ulambda = ulambda + siglambda*(worktofsino - ylambda)
         tgx,tgy = gradim(imlambda)
         tgx *= nulam
         tgy *= nulam
         ugradlamx = ugradlamx + siglambda*(tgx - ygradlamx)
         ugradlamy = ugradlamy + siglambda*(tgy - ygradlamy)

         circularParallelBeamProjection(immu,worksino)
         umu = umu +sigmu*(worksino-ymu)
         tgx,tgy = gradim(immu)
         tgx *= numu
         tgy *= numu
         ugradmux = ugradmux + sigmu*(tgx - ygradmux)
         ugradmuy = ugradmuy + sigmu*(tgy - ygradmuy)


         datarmse = sqrt(( (tofdata-worktofsino*exp(-worksino))**2. ).sum())/sqrt(1.*ntof*nviews*nbins)
         imagelrmse = sqrt(( (testlambda-imlambda)**2. ).sum())/sqrt(1.*nx*ny)
         imagemrmse = sqrt(( (testmu-immu)**2. ).sum())/sqrt(1.*nx*ny)
         datarmses.append(datarmse)
#   input("step")

         if j+1 in storeiterations:
            print("iter: ",j+1," ",datarmse," ",imagelrmse," ",imagemrmse," ",ymu.min())
            print("TV lam: ",tvlamest,":",tvlambda," TV mu: ",tvmuest,":",tvmu)
#            save(saaresultsfile+"lambda"+str(j+1)+".npy",imlambda/datafactor) #save the images with datafactor removed in order to be able to compare with phantom
#            save(saaresultsfile+"mu"+str(j+1)+".npy",immu)
      dgridrow.append(datarmse)
   dgrid.append(dgridrow)
save(saaresultsfile+"datarmse.npy",array(dgrid))
