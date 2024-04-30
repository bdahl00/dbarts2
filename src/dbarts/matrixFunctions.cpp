#include "config.hpp"
#include "matrixFunctions.hpp"

#include <cstddef>

#include <dbarts/bartFit.hpp>
#include "tree.hpp"
#include <Eigen/Dense>

namespace dbarts {
  using std::size_t;

  double computeMarginalLogLikelihood(const BARTFit& fit, size_t chainNum, const Tree& tree, const double* R, double sigma) {
    Eigen::MatrixXd IMinusBD = tree.calculateIMinusBD(fit);
    Eigen::VectorXd IMinusBR = tree.calculateIMinusBR(fit, R);
    Eigen::MatrixXd DTLambdaD = IMinusBD.transpose() * IMinusBD;
    auto Q = Eigen::MatrixXd::Identity(IMinusBD.cols(), IMinusBD.cols()); // needs to be updated with state.k somehow
    Eigen::MatrixXd fullCondVar = (sigma * sigma * Q + DTLambdaD).inverse();
    
    double part1 = Q.cols() * log(sigma) / 2 - log(fullCondVar.determinant()) / 2; // This will change once state.k is included
    Eigen::MatrixXd DTLambdaR = IMinusBD.transpose() * IMinusBR;
    double part2 = (DTLambdaR.transpose() * fullCondVar * DTLambdaR)(1,1) / (2 * sigma * sigma);
    return part1 + part2;
  }
}
