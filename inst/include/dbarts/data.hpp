#ifndef DBARTS_DATA_HPP
#define DBARTS_DATA_HPP

#include <cstddef> // size_t
#include <dbarts/cstdint.hpp>

#include <dbarts/types.hpp>

namespace dbarts {
  struct Data {
    const double* y;
    const double* x;
    const double* x_test;
    
    const double* weights;
    const double* offset;
    const double* testOffset;

    std::size_t* vecchiaIndices; // bdahl: Ideally would be const - easier in src/R_interface_common.cpp to leave as is
    const double* vecchiaVals;
    const double* vecchiaVars;
    std::size_t numNeighbors;
    
    std::size_t numObservations;
    std::size_t numPredictors;
    std::size_t numTestObservations;
    double sigmaEstimate;
    
    const VariableType* variableTypes;
    const std::uint32_t* maxNumCuts; // length = numPredictors if control.useQuantiles is true
    
    Data() :
      y(NULL), x(NULL), x_test(NULL), weights(NULL), offset(NULL), testOffset(NULL),
      vecchiaIndices(NULL), vecchiaVals(NULL), vecchiaVars(NULL), numNeighbors(0),
      numObservations(0), numPredictors(0), numTestObservations(0),
      sigmaEstimate(1.0), variableTypes(NULL), maxNumCuts(NULL)
    { }
    
    Data(const double* y,
         const double* x,
         const double* x_test,
         const double* weights,
         const double* offset,
         const double* testOffset,
         std::size_t numObservations,
         std::size_t numPredictors,
         std::size_t numTestObservations,
         double sigmaEstimate,
         const VariableType* variableTypes,
         const std::uint32_t* maxNumCuts) :
      y(y), x(x), x_test(x_test), weights(weights), offset(offset), testOffset(testOffset),
      vecchiaIndices(NULL), vecchiaVals(NULL), vecchiaVars(NULL), numNeighbors(0),
      numObservations(numObservations), numPredictors(numPredictors), numTestObservations(numTestObservations),
      sigmaEstimate(sigmaEstimate), variableTypes(variableTypes), maxNumCuts(maxNumCuts)
    {
    }
    
    Data(const double* y,
         const double* x,
         const double* x_test,
         const double* weights,
         const double* offset,
         const double* testOffset,
         std::size_t* vecchiaIndices,
         const double* vecchiaVals,
         const double* vecchiaVars,
         std::size_t numNeighbors,
         std::size_t numObservations,
         std::size_t numPredictors,
         std::size_t numTestObservations,
         double sigmaEstimate,
         const VariableType* variableTypes,
         const std::uint32_t* maxNumCust) :
      y(y), x(x), x_test(x_test), weights(weights), offset(offset), testOffset(testOffset),
      vecchiaIndices(vecchiaIndices), vecchiaVals(vecchiaVals), vecchiaVars(vecchiaVars), numNeighbors(numNeighbors),
      numObservations(numObservations), numPredictors(numPredictors), numTestObservations(numTestObservations),
      sigmaEstimate(sigmaEstimate), variableTypes(variableTypes), maxNumCuts(maxNumCuts)
    {
    }
  };
} // namespace dbarts

#endif // DBARTS_DATA_HPP
