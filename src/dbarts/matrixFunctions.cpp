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
//std::cout << "calculateIMinusBD called" << std::endl;
    Eigen::MatrixXd IMinusBD = tree.calculateIMinusBD(fit);
// std::cout << "calculateIMinusBR called" << std::endl; 
    Eigen::VectorXd IMinusBR = tree.calculateIMinusBR(fit, R);
// std::cout << "External functions passed" << std::endl;
    Eigen::MatrixXd DTLambdaD = IMinusBD.transpose() * IMinusBD;
    auto Q = Eigen::MatrixXd::Identity(IMinusBD.cols(), IMinusBD.cols()); // needs to be updated with state.k somehow
    Eigen::MatrixXd fullCondVar = (sigma * sigma * Q + DTLambdaD).inverse();
    
    double part1 = Q.cols() * log(sigma) / 2 + log(fullCondVar.determinant()) / 2; // This will change once state.k is included
    Eigen::VectorXd DTLambdaR = IMinusBD.transpose() * IMinusBR;
    double part2 = DTLambdaR.dot(fullCondVar * DTLambdaR) / (2 * sigma * sigma);
    //double part2 = sigma * sigma * DTLambdaR.dot(fullCondVar * DTLambdaR) / 2;
//    std::cout << "Exponential part: " << part2 << std::endl;
    return part1 + part2;
  }
}
