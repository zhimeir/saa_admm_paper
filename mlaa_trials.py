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
tofdata0 = attfactor*tofsino
datafactor = totalcounts/tofdata0.sum() #scale the data to the desired count level
tofdata0 *= datafactor
testlambda  = testlambda*datafactor
truess *= datafactor

ltotalcounts = testlambda.sum()

ntrials = 100
niter = 201
storeiterations=[1,2,5,10,20,50,100,200]
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


   lordata = tofdata.sum(axis=0)
   bplordata = testmu*0.
   circularParallelBeamBackProjection(lordata,bplordata)
   bplordata[bplordata<0.] = 0.
   bppsi= testmu*0.

#   totalcounts = testlambda.sum()


   oneim = testmu*0.
   oneim.fill(1.)
   ponesino = 0.*sinogram
   wbponeimage = 0.*testlambda
   workimage = 0.*wbponeimage

   circularParallelBeamProjection(oneim,ponesino)

   onedata = tofdata*1.
   onedata.fill(1.)


   worktofsino = 0.*tofsino

   imlambda = testlambda*0.
   imlambda.fill(1.)
   immu = testmu*0.
   immu.fill(0.)
   attfactorest = attfactor*1.
   attfactorest.fill(1.)
   datarmses = []
   imagelrmses = []
   imagemrmses = []
   attrmses = []

   nim = 0
   for j in range(niter):
# lambda update
      TOFprojection(imlambda,worktofsino)
      worktofdata = worktofsino*attfactorest   #put attfactor to test MLEM on lambda alone
      worktofdata[worktofdata<small]=small     #avoid divide by zero in em update
   
      emratio = attfactorest*tofdata/worktofdata
   
      datarmse = sqrt(( (tofdata-worktofsino*attfactorest)**2. ).sum())/sqrt( (tofdata**2).sum() )
      imagelrmse = sqrt(( (testlambda-imlambda)**2. ).sum())/sqrt( (testlambda**2.).sum() )
      imagemrmse = sqrt(( (testmu-immu)**2. ).sum())/sqrt( (testmu**2.).sum() )
      attdiff = (truess)*(attfactor -  attfactorest)
      attrmse = sqrt(( attdiff**2. ).sum())/sqrt( ((truess*attfactor)**2).sum() )
      datarmses.append(datarmse)
      imagelrmses.append(imagelrmse)
      imagemrmses.append(imagemrmse)
      attrmses.append(attrmse)
#      print("iter: ",j," ",datarmse," ",imagelrmse," ",imagemrmse," ",attrmse)


      TOFbackProjection(emratio,workimage)
      TOFbackProjection(attfactorest*onedata,wbponeimage)
      wbponeimage[wbponeimage<small] = small
      imlambda *= (workimage/wbponeimage)
      imlambda *= ltotalcounts/imlambda.sum() #normalize to ltotalcounts

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
#      immu[immu<0.]=0.
#      immu += (bppsi - bplordata)/denom 

   
      circularParallelBeamProjection(immu,sinogram)
      attfactorest = exp(-sinogram)
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
   ilstd = sqrt( (imagesos - (imagesum**2.)/ntrials)/(ntrials-1.) )
save(resultsfile+"drmse.npy",array(drmse))
save(resultsfile+"ilrmse.npy",array(ilrmse))
save(resultsfile+"imrmse.npy",array(imrmse))
save(resultsfile+"afrmse.npy",array(afrmse))
save(resultsfile+"lambdamean.npy",ilmean/datafactor)
save(resultsfile+"lambdastd.npy",ilstd/datafactor)
