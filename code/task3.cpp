#include <queso/Environment.h>
#include <queso/GslVector.h>
#include <queso/GslMatrix.h>
#include <queso/VectorSpace.h>
#include <queso/BoxSubset.h>
#include <queso/GaussianVectorRV.h>
#include <mpi.h>

int main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);

  QUESO::FullEnvironment env(MPI_COMM_WORLD, argv[1], "", NULL);

  // 1D is easier to play with
  QUESO::VectorSpace<> paramSpace(env, "", 1, NULL);
  QUESO::GslVector mins(paramSpace.zeroVector());
  QUESO::GslVector maxs(paramSpace.zeroVector());

  mins[0] = -INFINITY;  // Gaussians have infinite support
  maxs[0] = INFINITY;

  // Set up random variable
  QUESO::BoxSubset<> paramDomain("", paramSpace, mins, maxs);

  QUESO::GslVector mean(paramSpace.zeroVector());
  QUESO::GslMatrix cov(paramSpace.zeroVector());

  mean[0] = 1.0;
  cov(0, 0) = 1.0;  // 1x1 matrix

  QUESO::GaussianVectorRV<> randomVariable("", paramDomain, mean, cov);

  // Make draws
  QUESO::GslVector draw(paramSpace.zeroVector());
  QUESO::GslVector sampleMean(paramSpace.zeroVector());
  QUESO::GslVector sumSquares(paramSpace.zeroVector());
  QUESO::GslVector delta(paramSpace.zeroVector());
  QUESO::GslVector temp(paramSpace.zeroVector());

  unsigned int numDraws = 1000000;
  for (unsigned int i; i < numDraws; i++) {
    randomVariable.realizer().realization(draw);

    delta = draw;
    delta -= sampleMean;

    temp = delta;
    temp /= (double) (i + 1);
    sampleMean += temp;

    temp = draw;
    temp -= sampleMean;
    temp *= delta;
    sumSquares += temp;
  }
  temp = sumSquares;
  temp /= numDraws - 1;  // temp now contains the sample variance

  std::cout << "Sample mean is:" << std::endl;
  std::cout << sampleMean << std::endl;
  std::cout << "Sample variance is:" << std::endl;
  std::cout << temp << std::endl;

  MPI_Finalize();

  return 0;
}
