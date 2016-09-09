load_all("~/rpackages/mlr/")
load_all("~/rpackages/mlrMBO/")
library(OpenML)
library(batchtools)


types = c("C-svc", "spoc-svc", "kbb-svc")

OVERWRITE = TRUE


lrns = lapply(types, function(type) {
  lrn = makeLearner("classif.ksvm", id = type, type = type)
  
  ps = makeParamSet(
    makeNumericParam("C", -10, 10, trafo = function(x) 2^x),
    makeNumericParam("sigma", -10, 10, trafo = function(x) 2^x))
  
  mbo.ctrl = makeMBOControl()
  
  mbo.ctrl = setMBOControlTermination(mbo.ctrl, iters = 20, time.budget = 3600L)
  
  surrogate.lrn = makeLearner("regr.km", predict.type = "se")
  
  ctrl = mlr:::makeTuneControlMBO(learner = surrogate.lrn,
    mbo.control = mbo.ctrl, same.resampling.instance = FALSE)
  
  makeTuneWrapper(lrn, cv10, measures = acc, par.set = ps, control = ctrl)
})

omls = listOMLDataSets(number.of.instances = c(100, 5000), number.of.features = c(5, 100), number.of.missing.values = 0L,
  number.of.classes = c(3, 10))


unlink("registry/", recursive = TRUE)
reg = makeExperimentRegistry()
batchmark(learners = lrns, data.ids = omls$data.id[1:2], measures = acc, resamplings = cv2)
summarizeExperiments()
submitJobs()
getStatus()
res = reduceBatchmarkResults(findDone())
rmat = convertBMRToRankMatrix(res)
plotBMRSummary(res)
plotBMRBoxplots(res, acc, style = "violin")

