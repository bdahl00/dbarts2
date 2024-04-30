#ifndef DBARTS_MATRIX_FUNCTIONS_HPP
#define DBARTS_MATRIX_FUNCTIONS_HPP

#include <cstddef>

namespace dbarts {
  struct BARTFit;
  struct Tree;

  double computeMarginalLogLikelihood(const BARTFit& fit, std::size_t chainNum, const Tree& tree, const double* R, double sigma);
}

#endif
