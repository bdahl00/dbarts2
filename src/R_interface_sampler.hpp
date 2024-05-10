#ifndef R_INTERFACE_SAMPLER_HPP
#define R_INTERFACE_SAMPLER_HPP

// basic sampler creation, running, and accessing

#include <external/Rinternals.h> // SEXP

extern "C" {
  
  SEXP create(SEXP control, SEXP model, SEXP data);
  SEXP run(SEXP fit, SEXP numBurnIn, SEXP numSamples);
  SEXP sampleTreesFromPrior(SEXP fit);
  SEXP sampleNodeParametersFromPrior(SEXP fit);
  
  SEXP setData(SEXP fit, SEXP data);
  SEXP setControl(SEXP fit, SEXP control);
  SEXP setModel(SEXP fit, SEXP model);
  SEXP setSpatialStructureFromLocations(SEXP fit, SEXP numNeighbors, SEXP locs, SEXP rangeParam, SEXP smoothnessParam); // bdahl addition
  SEXP setSpatialStructureFromNeighbors(SEXP fit, SEXP vecchiaIndices, SEXP vecchiaVals, SEXP vecchiaVars); // Eventually deprecate the extra arguments in the dbarts function
  SEXP setTestSpatialStructureFromLocations(SEXP fit, SEXP locs); // addSpatialStructureFromLocations MUST have been called first
  SEXP setTestSpatialStructureFromNeighbors(SEXP fit, SEXP testVecchiaIndices, SEXP testVecchiaVals); // bdahl addition
  
  SEXP predict(SEXP fit, SEXP x_test, SEXP offset_test);
  SEXP setResponse(SEXP fit, SEXP y);
  SEXP setOffset(SEXP fit, SEXP offset, SEXP updateScale);
  SEXP setWeights(SEXP fit, SEXP weights);
  SEXP setSigma(SEXP fit, SEXP sigma);
  SEXP setPredictor(SEXP fit, SEXP x, SEXP forceUpdate, SEXP updateCutPoints);
  SEXP updatePredictor(SEXP fit, SEXP x, SEXP cols, SEXP forceUpdate, SEXP updateCutPoints);
  SEXP setCutPoints(SEXP fitExpr, SEXP cutPointsExpr, SEXP colsExpr);
  SEXP setTestPredictor(SEXP fit, SEXP x_test);
  SEXP setTestOffset(SEXP fit, SEXP offset_test);
  SEXP setTestPredictorAndOffset(SEXP fit, SEXP x_test, SEXP offset_test);
  SEXP storeLatents(SEXP fit, SEXP result);
  
  SEXP updateTestPredictor(SEXP fit, SEXP x_test, SEXP cols);
   
  SEXP createState(SEXP fit);
  SEXP storeState(SEXP fit, SEXP state);
  SEXP restoreState(SEXP fit, SEXP state);
  
  SEXP getTrees(SEXP fit, SEXP chainIndices, SEXP sampleIndices, SEXP treeIndices);
  SEXP printTrees(SEXP fit, SEXP chainIndices, SEXP sampleIndices, SEXP treeIndices);
  
  double matern(double distance, double range, double smoothness);
}

#endif

