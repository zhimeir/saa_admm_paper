from tofsetup import *

print("Ntof: ",ntof)

tvmuswitch = 1
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
tvstring = ""
if tvmuswitch:
   tvstring = tvstring + "_TVmu"+str(ptvmu)
if tvlamswitch:
   tvstring = tvstring + "_TVlam"+str(ptvlam)

ltotalcounts = testlambda.sum()
print("Total Counts: ",ltotalcounts)

# grid search results 9cm
# Both TV constraints
if tvmuswitch and tvlamswitch:
   sbmu = 100
   sblambda = 0.005
# mu TV constraint only
if tvmuswitch and not tvlamswitch:
   sbmu = 100 
   sblambda = 0.01
# lambda TV constraint only
if tvlamswitch and not tvmuswitch:
   sbmu = 200 
   sblambda = 0.01
print("sbmu: ",sbmu," sblambda",sblambda)

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

niter = 201

newtiter = 10
nuv = 100

datarmses = []
imagelrmses = []
imagemrmses = []
attrmses = []
storeiterations=[1,2,5,10,20,50,100,200,500,1000]
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
   nu  = ((imlambda.sum() -ltotalcounts)/taulambda - wimlambda.sum() - wgradimlam)/fovmask.sum()
#   nu = 0.

#   print("tc 1 :",imlambda.sum(), "true: ",ltotalcounts)
   imlambda =imlambda -taulambda*(nu*fovmask + wimlambda + wgradimlam)
#   imlambda[imlambda<0.]=0.
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
#   immu[immu<0.]=0.
   immu *= mumask
   circularParallelBeamProjection(immu,worksino)
   workgradmux, workgradmuy = gradim(immu)
   workgradmux *= numu
   workgradmuy *= numu

   tvmuest  = sqrt(tgx**2 + tgy**2).sum()
   print("TV lam: ",tvlamest,":",tvlambda," TV mu: ",tvmuest,":",tvmu)

   ylambdat = 1.*ylambda
   ymut = 1.*ymu
# y-update
#   aq = 1.0 + siglambda
   aq = siglambda
   cq = -tofdata
   hessmin = 1.e10
   for k in range(nuv):
#      ylambdat = 1.*ylambda
#      ymut = 1.*ymu
      ymuterm = ones([ntof])[:,newaxis,newaxis]*qexp(-ymu)
#      ymuterm = ones([ntof])[:,newaxis,newaxis]*exp(-ymu)
#      bq = ymuterm - ylambdat - ulambda - siglambda*worktofsino #  ylambdat or tofsino for test
      bq = ymuterm - ulambda - siglambda*worktofsino #  ylambdat or tofsino for test
#      bq = ymuterm - tofsino - ulambda - siglambda*worktofsino #  ylambdat or tofsino for test
      ylambdaold = ylambda*1.
      ylambda = (-bq + sqrt(bq*bq - 4.*aq*cq))/(2.*aq)
      ylambdaold = ylambda*1.

      ymunewt = ymu*0.
      for i in range(newtiter):
#         grad = (-qexp(-ymunewt,1)*ylambdaold + tofdata).sum(axis=0) + ntof*(hfunc(ymunewt,1)-hfunc(ymut,1)) - umu + sigmu*(ymunewt - worksino)
         grad = (-qexp(-ymunewt,1)*ylambdaold + tofdata).sum(axis=0) - umu + sigmu*(ymunewt - worksino)
#         grad = (-exp(-ymunewt)*ylambdaold + tofdata).sum(axis=0) - umu + sigmu*(ymunewt - worksino)
#         grad = (-qexp(-ymunewt,1)*ylambdaold + tofdata).sum(axis=0) + ntof*(hfunc(ymunewt,1)-hfunc(sinoatt,1)) - umu + sigmu*(ymunewt - worksino)
         ymuterm = ylambdaold*qexp(-ymunewt,2)
#         ymuterm = ylambdaold*exp(-ymunewt)
#         hess = ymuterm.sum(axis=0) + ntof*hfunc(ymunewt,2) + sigmu
         hess = ymuterm.sum(axis=0) + sigmu
         hessmin = minimum(hessmin,hess.min())
         ymunewt = ymunewt - grad/hess
#         print(i," grad mag: ",sqrt((grad**2).sum()))
      ymuold = ymu*1.
      ymumin = ymunewt.min()
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
   print("iter: ",j," ",datarmse," ",imagelrmse," ",imagemrmse," ",attrmse)
#   input("step")

   if j+1 in storeiterations:
      save(saaresultsfile+"lambda"+str(j+1)+tvstring+".npy",imlambda/datafactor) #save the images with datafactor removed in order to be able to compare with phantom
      save(saaresultsfile+"mu"+str(j+1)+tvstring+".npy",immu)

save(saaresultsfile+"datarmse"+tvstring+".npy",array(datarmses))
save(saaresultsfile+"imagelrmse"+tvstring+".npy",array(imagelrmses))
save(saaresultsfile+"imagemrmse"+tvstring+".npy",array(imagemrmses))
save(saaresultsfile+"attrmse"+tvstring+".npy",array(attrmses))
