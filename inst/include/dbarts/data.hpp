#ifndef DBARTS_DATA_HPP
#define DBARTS_DATA_HPP

#include <cstddef> // size_t
#include <dbarts/cstdint.hpp>
#include <Eigen/Sparse>

#include <dbarts/types.hpp>

namespace dbarts {
  struct Data {
    const double* y;
    const double* x;
    const double* x_test;
    
    const double* weights;
    const double* offset;
    const double* testOffset;

    std::size_t numNeighbors;
    Eigen::SparseMatrix<double> adjIMinusB;
    double range;
    double smoothness;
    const std::size_t* testNeighbors;
    const double* testNeighborDeviationWeights;

    std::size_t numObservations;
    std::size_t numPredictors;
    std::size_t numTestObservations;
    double sigmaEstimate;
    
    const VariableType* variableTypes;
    const std::uint32_t* maxNumCuts; // length = numPredictors if control.useQuantiles is true
    
    Data() :
      y(NULL), x(NULL), x_test(NULL), weights(NULL), offset(NULL), testOffset(NULL),
      numNeighbors(0), range(0.0), smoothness(0.0), testNeighbors(NULL), testNeighborDeviationWeights(NULL),
      numObservations(0), numPredictors(0), numTestObservations(0),
      sigmaEstimate(1.0), variableTypes(NULL), maxNumCuts(NULL)
    {
      Eigen::SparseMatrix<double> adjIMinusB(1,1);
      this->adjIMinusB = adjIMinusB; 
    }
    
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
      numNeighbors(0), range(0.0), smoothness(0.0), testNeighbors(NULL), testNeighborDeviationWeights(NULL),
      numObservations(numObservations), numPredictors(numPredictors), numTestObservations(numTestObservations),
      sigmaEstimate(sigmaEstimate), variableTypes(variableTypes), maxNumCuts(maxNumCuts)
    {
      Eigen::SparseMatrix<double> adjIMinusB(1,1);
      this->adjIMinusB = adjIMinusB; 
    }


    Data(const double* y,
         const double* x,
         const double* x_test,
         const double* weights,
         const double* offset,
         const double* testOffset,
         Eigen::SparseMatrix<double> adjIMinusB,
         std::size_t numNeighbors,
         std::size_t numObservations,
         std::size_t numPredictors,
         std::size_t numTestObservations,
         double sigmaEstimate,
         const VariableType* variableTypes,
         const std::uint32_t* maxNumCust) :
      y(y), x(x), x_test(x_test), weights(weights), offset(offset), testOffset(testOffset),
      adjIMinusB(adjIMinusB), numNeighbors(numNeighbors), range(0.0), smoothness(0.0),
      testNeighbors(NULL), testNeighborDeviationWeights(NULL),
      numObservations(numObservations), numPredictors(numPredictors), numTestObservations(numTestObservations),
      sigmaEstimate(sigmaEstimate), variableTypes(variableTypes), maxNumCuts(maxNumCuts)
    {
    }

  };
} // namespace dbarts

#endif // DBARTS_DATA_HPP
