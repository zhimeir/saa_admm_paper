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
tofdata[tofdata<small] = small

datafactor = totalcounts/tofdata.sum() #scale the data to the desired count level
print("datafactor: ",datafactor)
tofdata *= datafactor
testlambda  *= datafactor
truess *= datafactor
#tofdata = random.poisson(tofdata).astype("float64") #UNCOMMENT for noise and COMMENT out for no noise
print("tofdata: ",tofdata.min(),tofdata.max())
print("tofsum: ",tofdata.sum())


lordata = tofdata.sum(axis=0)
bplordata = testmu*0.
circularParallelBeamBackProjection(lordata,bplordata)
bplordata[bplordata<0.] = 0.
bppsi= testmu*0.

totalcounts = testlambda.sum()


oneim = testmu*0.
oneim.fill(1.)
ponesino = 0.*sinogram
wbponeimage = 0.*testlambda
workimage = 0.*wbponeimage

circularParallelBeamProjection(oneim,ponesino)

onedata = tofdata*1.
onedata.fill(1.)
t1 = time.time()
print("Starting fastbackprojection ...")
TOFbackProjection(onedata,wbponeimage)
print("Fast backprojection took ",time.time()-t1," seconds.")


worktofsino = 0.*tofsino

imlambda = testlambda*0.
imlambda.fill(1.)
immu = testmu*0.
immu.fill(0.)
attfactorest = attfactor*1.
attfactorest.fill(1.)
niter = 500
wbponeimage[wbponeimage<small] = small
datarmses = []
storeiterations=[1,2,5,10,20,50,100,200,500]
for j in range(niter):
# lambda update
   TOFprojection(imlambda,worktofsino)
   worktofdata = worktofsino*attfactorest   #put attfactor to test MLEM on lambda alone
   worktofdata[worktofdata<small]=small     #avoid divide by zero in em update
   
   emratio = tofdata/worktofdata
   datarmse = sqrt(( (tofdata-worktofdata)**2. ).sum())/sqrt(1.*ntof*nviews*nbins)
   datarmses.append(datarmse)
   imagelrmse = sqrt(( (testlambda-imlambda)**2. ).sum())/sqrt(1.*nx*ny)
   imagemrmse = sqrt(( (testmu-immu)**2. ).sum())/sqrt(1.*nx*ny)
   print("iter: ",j," ",datarmse," ",imagelrmse," ",imagemrmse)


   TOFbackProjection(emratio,workimage)
   imlambda *= (workimage/wbponeimage)
   imlambda *= totalcounts/imlambda.sum() #normalize to totalcounts

# mu-update
   TOFprojection(imlambda,worktofsino)

   psi = attfactorest*worktofsino.sum(axis=0)
#   psi = attfactorest*truess  #for testing MLTR alone
   psi[psi<0.]=0.
  
   circularParallelBeamBackProjection(ponesino*psi,bppsi)
   denom = bppsi*1.
   
   denom[denom<small]=small
   
   circularParallelBeamBackProjection(psi,bppsi)
   immu += mumask*(bppsi - bplordata)/denom 
#   immu[immu<0.]=0.
#   immu += (bppsi - bplordata)/denom 

   
   circularParallelBeamProjection(immu,sinogram)
   attfactorest = exp(-sinogram)
   if j+1 in storeiterations:
      save(resultsfile+"lambda"+str(j+1)+".npy",imlambda/datafactor) #save the images with datafactor removed in order to be able to compare with phantom
      save(resultsfile+"mu"+str(j+1)+".npy",immu)

save(resultsfile+"datarmse.npy",array(datarmses))
