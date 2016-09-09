library(mlr)
library(mlrMBO)
library(OpenML)
library(batchtools)


types = c("C-svc", "spoc-svc", "kbb-svc")

OVERWRITE = TRUE

if (OVERWRITE)
  unlink("registry/", recursive = TRUE)

reg = makeExperimentRegistry(packages = c("mlr", "mlrMBO", "OpenML"))

reg$default.resources = list(walltime = 7200L, memory = 2000L, ntasks = 1L)

lrns = lapply(types, function(type) {
  lrn = makeLearner("classif.ksvm", id = type, type = type)
  
  ps = makeParamSet(
    makeNumericParam("C", -12, 12, trafo = function(x) 2^x),
    makeNumericParam("sigma", -12, 12, trafo = function(x) 2^x))
  
  mbo.ctrl = makeMBOControl()
  
  mbo.ctrl = setMBOControlTermination(mbo.ctrl, iters = 20, time.budget = 3600L)
  
  surrogate.lrn = makeLearner("regr.km", predict.type = "se")
  
  ctrl = mlr:::makeTuneControlMBO(learner = surrogate.lrn,
    mbo.control = mbo.ctrl, same.resampling.instance = FALSE)
  
  makeTuneWrapper(lrn, cv5, measures = acc, par.set = ps, control = ctrl)
})


omls = listOMLDataSets(number.of.instances = c(100, 5000), number.of.features = c(5, 100), number.of.missing.values = 0L,
  number.of.classes = c(3, 10))

batchmark(learners = lrns, data.ids = omls$data.id, measures = acc, resamplings = cv10)
summarizeExperiments()

