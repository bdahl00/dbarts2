#ifndef DBARTS_MATRIX_FUNCTIONS_HPP
#define DBARTS_MATRIX_FUNCTIONS_HPP

#include <cstddef>
#include <vector>
#include <Eigen/Dense>

namespace dbarts {
  struct BARTFit;
  struct Tree;

  double computeMarginalLogLikelihood(const BARTFit& fit, std::size_t chainNum, const Tree& tree, const double* R, double sigma);
  double computeMarginalLogLikelihood2(const BARTFit& fit, const Tree& tree, const std::vector<Eigen::VectorXd*>& IMinusBDObj, const std::vector<std::size_t>& IMinusBDColIndices, const Eigen::MatrixXd& fullCondVar, const double* R, double sigma);
  void createSuiteOfMatrixObjects(const BARTFit& fit, const Tree& tree, std::vector<Eigen::VectorXd*>& IMinusBDObj, std::vector<std::size_t>& IMinusBDColIndices, Eigen::MatrixXd& DTLambdaD, Eigen::MatrixXd& fullCondVar, double sigma);
}

#endif
