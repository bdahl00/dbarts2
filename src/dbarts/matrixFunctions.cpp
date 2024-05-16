#include "config.hpp"
#include "matrixFunctions.hpp"

#include <cstddef>

#include <dbarts/bartFit.hpp>
#include "tree.hpp"
#include "node.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <chrono>

namespace dbarts {
  using std::size_t;

  double computeMarginalLogLikelihood(const BARTFit& fit, size_t chainNum, const Tree& tree, const double* R, double sigma) {
#define firstway 1
#if firstway
    auto t1 = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd IMinusBD = tree.calculateIMinusBD(fit);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Time to calculate IMinusBD: " <<  ms_double.count() << " milliseconds" << std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd IMinusBR = tree.calculateIMinusBR(fit, R);
    t2 = std::chrono::high_resolution_clock::now();
    ms_double = t2 - t1;
    std::cout << "Time to calculate IMinusBR: " << ms_double.count() << " milliseconds" << std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd DTLambdaD = IMinusBD.transpose() * IMinusBD;
    t2 = std::chrono::high_resolution_clock::now();
    ms_double = t2 - t1;
    std::cout << "Time to calculate DTLambdaD: " << ms_double.count() << " milliseconds" << std::endl;
    //Eigen::MatrixXd DTLambdaD(IMinusBD.cols(), IMinusBD.cols());
    //DTLambdaD.setZero().selfadjointView<Eigen::Lower>().rankUpdate(IMinusBD.transpose());
    auto Q = Eigen::MatrixXd::Identity(IMinusBD.cols(), IMinusBD.cols()); // needs to be updated with state.k somehow
    Eigen::MatrixXd fullCondVar = (sigma * sigma * Q + DTLambdaD).inverse();
    Eigen::VectorXd DTLambdaR = IMinusBD.transpose() * IMinusBR;
#else
    NodeVector bottomNodes(tree.getBottomNodes());
    std::size_t numBottomNodes = bottomNodes.size();
 
    // Calculate D
    Eigen::SparseMatrix<double> D(fit.data.numObservations, numBottomNodes);
    std::vector<int> numObsInNode(numBottomNodes);
    for (std::size_t colIndex = 0; colIndex < numBottomNodes; ++colIndex) {
      numObsInNode.at(colIndex) = static_cast<int>(bottomNodes[colIndex]->numObservations);
    }
    D.reserve(numObsInNode);
    for (std::size_t colIndex; colIndex < numBottomNodes; ++colIndex) {
      for (std::size_t nodeObsIndex = 0; nodeObsIndex < numObsInNode.at(colIndex); ++nodeObsIndex) {
        D.insert(bottomNodes[colIndex]->observationIndices[nodeObsIndex], colIndex) = 1;
      }
    }

    Eigen::MatrixXd DTLambdaD(numBottomNodes, numBottomNodes);
    DTLambdaD = D.transpose() * fit.data.Lambda.selfadjointView<Eigen::Lower>() * D;
    auto Q = Eigen::MatrixXd::Identity(numBottomNodes, numBottomNodes);
    Eigen::MatrixXd fullCondVar = (sigma * sigma * Q + DTLambdaD).inverse();
    Eigen::Map<const Eigen::VectorXd> Rvec(R, fit.data.numObservations);
    Eigen::VectorXd DTLambdaR = D.transpose() * fit.data.Lambda * Rvec;
#endif
    
    double part1 = Q.cols() * log(sigma) + log(fullCondVar.determinant()) / 2; // This will change once state.k is included
    double part2 = DTLambdaR.dot(fullCondVar * DTLambdaR) / (2 * sigma * sigma);
    return part1 + part2;
  }
}
