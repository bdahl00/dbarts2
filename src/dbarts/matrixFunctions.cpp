#include "config.hpp"
#include "matrixFunctions.hpp"

#include <cstddef>

#include <dbarts/bartFit.hpp>
#include "tree.hpp"
#include <Eigen/Dense>
#include <iostream>

namespace dbarts {
  using std::size_t;

  double computeMarginalLogLikelihood(const BARTFit& fit, size_t chainNum, const Tree& tree, const double* R, double sigma) {
    Eigen::MatrixXd IMinusBD = tree.calculateIMinusBD(fit);
    Eigen::VectorXd IMinusBR = tree.calculateIMinusBR(fit, R);
#if 1
    Eigen::MatrixXd DTLambdaD = IMinusBD.transpose() * IMinusBD;
    auto Q = Eigen::MatrixXd::Identity(IMinusBD.cols(), IMinusBD.cols()); // needs to be updated with state.k somehow
    Eigen::MatrixXd fullCondVar = (sigma * sigma * Q + DTLambdaD).inverse();
#endif
    //Eigen::MatrixXd fullCondVar = tree.commonFullCondVar;
    
    double part1 = Q.cols() * log(sigma) + log(fullCondVar.determinant()) / 2; // This will change once state.k is included
    //double part1 = tree.IMinusBD.cols() * log(sigma) / 2 + tree.fullCondVarHalfLogDeterminant / 2; 
    Eigen::VectorXd DTLambdaR = IMinusBD.transpose() * IMinusBR;
    double part2 = DTLambdaR.dot(fullCondVar * DTLambdaR) / (2 * sigma * sigma);
    //double part2 = sigma * sigma * DTLambdaR.dot(fullCondVar * DTLambdaR) / 2;
    return part1 + part2;
  }
}
