from tofsetup import *

print("Ntof: ",ntof)

tvmuswitch = 1
tvlamswitch = 0

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
tofdata0 = attfactor*tofsino
datafactor = totalcounts/tofdata0.sum() #scale the data to the desired count level
tofdata0 *= datafactor
testlambda  = testlambda*datafactor
truess *= datafactor
print("datafactor: ",datafactor)

ptvmu = 100
tgx,tgy = gradim(testmu)
tvmu = numu*sqrt(tgx**2 + tgy**2).sum() *(1.*ptvmu/100.0)

ptvlam = 100
tgx,tgy = gradim(testlambda)
tvlambda = nulam*sqrt(tgx**2 + tgy**2).sum() *(1.*ptvlam/100.0)
print("TVs: ",tvmu,tvlambda)
tvstring = ""
if tvmuswitch:
   tvstring = tvstring + "_TVmu"+str(ptvmu)
if tvlamswitch:
   tvstring = tvstring + "_TVlam"+str(ptvlam)

ltotalcounts = testlambda.sum()
print("Total Counts: ",ltotalcounts)

ntrials = 100
niter = 101
newtiter = 10
nuv = 100
storeiterations=[1,2,3,4,5,6,8,10,12,14,16,18,20,25,30,35,40,45,50,60,70,80,90,100]
nims = len(storeiterations)
imagesum = zeros([nims,nx,ny])
imagesos = zeros([nims,nx,ny])
drmse = []
ilrmse = []
imrmse = []
afrmse = []
for nt in range(ntrials):
   print("trial: ",nt)

   tofdata = tofdata0*1.

   tofdata = random.poisson(tofdata).astype("float64") #UNCOMMENT for noise and COMMENT out for no noise

# grid search results 9cm
# Both TV constraints
   if tvmuswitch and tvlamswitch:
      sbmu = 50
      sblambda = 0.01
# mu TV constraint only
   if tvmuswitch and not tvlamswitch:
      sbmu = 100 
      sblambda = 0.01
# lambda TV constraint only
   if tvlamswitch and not tvmuswitch:
      sbmu = 200 
      sblambda = 0.01

   sigmu = sbmu/lproj
   taumu = 1.0/(sbmu*lproj)

   siglambda = sblambda/ltofproj
   taulambda = 1.0/(sblambda*ltofproj)

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
   imagelrmses = []
   imagemrmses = []
   attrmses = []
   nim = 0
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

# counts constraint block
      nu  = ((imlambda.sum() -ltotalcounts)/taulambda - wimlambda.sum() - wgradimlam)/fovmask.sum()
#      nu = 0.
      imlambda =imlambda -taulambda*(nu*fovmask + wimlambda + wgradimlam)

      TOFprojection(imlambda,worktofsino)
      workgradlamx, workgradlamy = gradim(imlambda)
      workgradlamx *= nulam
      workgradlamy *= nulam

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

      immu =immu - wimmu*taumu - wgradimmu*taumu
      immu *= mumask
      circularParallelBeamProjection(immu,worksino)
      workgradmux, workgradmuy = gradim(immu)
      workgradmux *= numu
      workgradmuy *= numu

      tvmuest  = sqrt(tgx**2 + tgy**2).sum()
#   print("TV lam: ",tvlamest,":",tvlambda," TV mu: ",tvmuest,":",tvmu)

      ylambdat = 1.*ylambda
      ymut = 1.*ymu
# y-update
      aq = siglambda
      cq = -tofdata
      hessmin = 1.e10
      for k in range(nuv):
         ymuterm = ones([ntof])[:,newaxis,newaxis]*qexp(-ymu)
         bq = ymuterm - ulambda - siglambda*worktofsino #  ylambdat or tofsino for test
         ylambdaold = ylambda*1.
         ylambda = (-bq + sqrt(bq*bq - 4.*aq*cq))/(2.*aq)
         ylambdaold = ylambda*1.

         ymunewt = ymu*0.
         for i in range(newtiter):
            grad = (-qexp(-ymunewt,1)*ylambdaold + tofdata).sum(axis=0) - umu + sigmu*(ymunewt - worksino)
            ymuterm = ylambdaold*qexp(-ymunewt,2)
            hess = ymuterm.sum(axis=0) + sigmu
            hessmin = minimum(hessmin,hess.min())
            ymunewt = ymunewt - grad/hess
         ymuold = ymu*1.
         ymumin = ymunewt.min()
         ymu = ymunewt*1.
         ymu[ymu<0.] = 0.
         ylambdadist = sqrt( ( (ylambda-ylambdaold)**2).sum())
         ymudist = sqrt( ( (ymu-ymuold)**2).sum())

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


      attfactorest = exp(-worksino)
      datarmse = sqrt(( (tofdata-worktofsino*attfactorest)**2. ).sum())/sqrt( (tofdata**2).sum() )
      imagelrmse = sqrt(( (testlambda-imlambda)**2. ).sum())/sqrt( (testlambda**2.).sum() )
      imagemrmse = sqrt(( (testmu-immu)**2. ).sum())/sqrt( (testmu**2.).sum() )
      attdiff = (truess)*(attfactor -  attfactorest)
      attrmse = sqrt(( attdiff**2. ).sum())/sqrt( ((truess*attfactor)**2).sum() )
      datarmses.append(datarmse)
      imagelrmses.append(imagelrmse)
      imagemrmses.append(imagemrmse)
      attrmses.append(attrmse)

      if j+1 in storeiterations:
         print("iter: ",j," ",datarmse," ",imagelrmse," ",imagemrmse," ",attrmse)
         imagesum[nim] += imlambda
         imagesos[nim] += imlambda**2
         nim += 1
   drmse.append(datarmses)
   ilrmse.append(imagelrmses)
   imrmse.append(imagemrmses)
   afrmse.append(attrmses)

for i in range(nims):
   ilmean = imagesum/ntrials
   stdiff =  imagesos - (imagesum**2.)/ntrials
   stdiff[stdiff < 0.] = 0.
   ilstd = sqrt( stdiff/(ntrials-1.) )
save(saaresultsfile+"drmse"+tvstring+".npy",array(drmse))
save(saaresultsfile+"ilrmse"+tvstring+".npy",array(ilrmse))
save(saaresultsfile+"imrmse"+tvstring+".npy",array(imrmse))
save(saaresultsfile+"afrmse"+tvstring+".npy",array(afrmse))
save(saaresultsfile+"lambdamean"+tvstring+".npy",ilmean/datafactor)
save(saaresultsfile+"lambdastd"+tvstring+".npy",ilstd/datafactor)
