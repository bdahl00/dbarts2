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
#if !optimizedCache
  double computeMarginalLogLikelihood(const BARTFit& fit, size_t chainNum, const Tree& tree, const double* R, double sigma) {
    //auto t1 = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd IMinusBD = tree.calculateIMinusBD(fit);
    //auto t2 = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    //std::cout << "Time to calculate IMinusBD: " <<  ms_double.count() << " milliseconds" << std::endl;
    //t1 = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd IMinusBR = tree.calculateIMinusBR(fit, R);
    //t2 = std::chrono::high_resolution_clock::now();
    //ms_double = t2 - t1;
    //std::cout << "Time to calculate IMinusBR: " << ms_double.count() << " milliseconds" << std::endl;
    //t1 = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd DTLambdaD = IMinusBD.transpose() * IMinusBD;
    //t2 = std::chrono::high_resolution_clock::now();
    //ms_double = t2 - t1;
    //std::cout << "Time to calculate DTLambdaD: " << ms_double.count() << " milliseconds" << std::endl;
    //Eigen::MatrixXd DTLambdaD(IMinusBD.cols(), IMinusBD.cols());
    //DTLambdaD.setZero().selfadjointView<Eigen::Lower>().rankUpdate(IMinusBD.transpose());
    auto Q = Eigen::MatrixXd::Identity(IMinusBD.cols(), IMinusBD.cols()); // needs to be updated with state.k somehow
    Eigen::MatrixXd fullCondVar = (sigma * sigma * Q + DTLambdaD).inverse();
    Eigen::VectorXd DTLambdaR = IMinusBD.transpose() * IMinusBR;
    
    double part1 = Q.cols() * log(sigma) + log(fullCondVar.determinant()) / 2; // This will change once state.k is included
    double part2 = DTLambdaR.dot(fullCondVar * DTLambdaR) / (2 * sigma * sigma);
//std::cout << "############ Beginning of computeMarginalLogLikelihood block ############" << std::endl;
//std::cout << "fullCondVar, known correct method:\n" << fullCondVar << std::endl;
//std::cout << "fullCondVar.determinant(), known correct method:" << fullCondVar.determinant() << std::endl;
//std::cout << "Parts, approach 1: " << part1 << "\t" << part2 << std::endl;
    return part1 + part2;
  }
#else
  double computeMarginalLogLikelihood(const BARTFit& fit, size_t chainNum, const Tree& tree, const double* R, double sigma) {
    NodeVector bottomNodes(tree.getBottomNodes());
    std::size_t numBottomNodes = bottomNodes.size();
    Eigen::VectorXd IMinusBR = tree.calculateIMinusBR(fit, R);
    Eigen::VectorXd DTLambdaR(numBottomNodes);
    Eigen::MatrixXd DTLambdaD(numBottomNodes, numBottomNodes);
    for (std::size_t colIndex = 0; colIndex < numBottomNodes; ++colIndex) {
      DTLambdaR(colIndex) = (bottomNodes[colIndex]->IMinusBDCol).dot(IMinusBR);
      DTLambdaD(colIndex, colIndex) = (bottomNodes[colIndex]->IMinusBDCol).dot(bottomNodes[colIndex]->IMinusBDCol);
      for(std::size_t rowIndex = 0; rowIndex < colIndex; ++rowIndex) {
        double innerProd = (bottomNodes[colIndex]->IMinusBDCol).dot(bottomNodes[rowIndex]->IMinusBDCol);
        DTLambdaD(rowIndex, colIndex) = innerProd;
        DTLambdaD(colIndex, rowIndex) = innerProd;
      }
    }
//std::cout << "DTLambdaD\n" << DTLambdaD << std::endl;
    auto Q = Eigen::MatrixXd::Identity(numBottomNodes, numBottomNodes); // needs to be updated with state.k somehow
    Eigen::MatrixXd fullCondVar = (sigma * sigma * Q + DTLambdaD).inverse();
    double part1 = numBottomNodes * log(sigma) + log(fullCondVar.determinant()) / 2; // This will change once state.k is included
    double part2 = DTLambdaR.dot(fullCondVar * DTLambdaR) / (2 * sigma * sigma);
//std::cout << "Part 1: " << part1 << ", part 2: " << part2 << std::endl;
    return part1 + part2;
  }
#endif

  // Assumes the vectors are the right length, does not assume anything about the matrices
  void createSuiteOfMatrixObjects(const BARTFit& fit, const Tree& tree, std::vector<Eigen::VectorXd*>& IMinusBDObj, std::vector<std::size_t>& IMinusBDColIndices, Eigen::MatrixXd& DTLambdaD, Eigen::MatrixXd& fullCondVar, double sigma) {
std::cout << "createSuiteOfMatrixObjects entered" << std::endl;
    Eigen::MatrixXd IMinusBD = tree.calculateIMinusBD(fit);
    for (std::size_t colIndex = 0; colIndex < IMinusBD.cols(); ++colIndex) {
      IMinusBDObj.at(colIndex) = new Eigen::VectorXd(fit.data.numObservations);
      *(IMinusBDObj.at(colIndex)) = IMinusBD.col(colIndex);
      IMinusBDColIndices.at(colIndex) = colIndex;
    }
std::cout << "IMinusBDObj created" << std::endl;
    DTLambdaD = IMinusBD.transpose() * IMinusBD;
    auto Q = Eigen::MatrixXd::Identity(IMinusBD.cols(), IMinusBD.cols());
    fullCondVar = (sigma * sigma * Q + DTLambdaD).inverse();
std::cout << "createSuiteOfMatrixObjects successfully exited" << std::endl;
  }
}
