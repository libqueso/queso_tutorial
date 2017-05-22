#include <queso/Environment.h>
#include <queso/GslVector.h>
#include <queso/GslMatrix.h>
#include <queso/BoxSubset.h>
#include <queso/UniformVectorRV.h>
#include <queso/ScalarFunction.h>
#include <queso/VectorSet.h>
#include <queso/VectorSpace.h>
#include <queso/StatisticalInverseProblem.h>

double model(double D, double T, double beta)
{
  return D * std::pow(T, beta);
}

template<class V = QUESO::GslVector, class M = QUESO::GslMatrix>
class Likelihood : public QUESO::BaseScalarFunction<V, M>
{
public:

  Likelihood(const char * prefix, const QUESO::VectorSet<V, M> & domain)
    : QUESO::BaseScalarFunction<V, M>(prefix, domain),
      m_T(7, 0),
      m_y(7, 0),
      m_sigma(10.0)
  {
    m_T[0] = 313.7;
    m_T[1] = 314.9;
    m_T[2] = 375.2;
    m_T[3] = 474.7;
    m_T[4] = 481.0;
    m_T[5] = 573.5;
    m_T[6] = 671.1;

    m_y[0] = 4603.50;
    m_y[1] = 4638.15;
    m_y[2] = 6302.27;
    m_y[3] = 9505.89;
    m_y[4] = 9755.11;
    m_y[5] = 13239.08;
    m_y[6] = 17431.02;
  }

  virtual ~Likelihood()
  {
    // Deconstruct here
  }

  virtual double lnValue(const V & domainVector, const V * domainDirection,
      V * gradVector, M * hessianMatrix, V * hessianEffect) const
  {
    // D is domainVector[0]
    // beta is domainVector[1]
    double misfit = 0.0;
    double temp = 0.0;
    for (unsigned int i = 0; i < m_y.size(); i++) {
      // Evaluate model and compare with observation
      temp = domainVector[0] * std::pow(m_T[i], domainVector[1]) - m_y[i];

      // Increment misfit with current observation misfit
      misfit += temp * temp;
    }

    return -0.5 * misfit / (m_sigma * m_sigma);
  }

  virtual double actualValue(const V & domainVector, const V * domainDirection,
      V * gradVector, M * hessianMatrix, V * hessianEffect) const
  {
    return std::exp(this->lnValue(domainVector, domainDirection, gradVector,
    hessianMatrix, hessianEffect));
  }

private:
  std::vector<double> m_T;
  std::vector<double> m_y;
  double m_sigma;
};

int main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);

  QUESO::FullEnvironment env(MPI_COMM_WORLD, argv[1], "", NULL);

  unsigned int dim = 2;
  QUESO::VectorSpace<> paramSpace(env, "", dim, NULL);

  QUESO::GslVector mins(paramSpace.zeroVector());
  mins[0] = 0.0;
  mins[1] = -INFINITY;

  QUESO::GslVector maxs(paramSpace.zeroVector());
  maxs[0] = INFINITY;
  maxs[1] = INFINITY;

  QUESO::BoxSubset<> paramDomain("", paramSpace, mins, maxs);
  QUESO::UniformVectorRV<> priorRv("", paramDomain);

  Likelihood<> likelihood("", paramDomain);

  QUESO::GenericVectorRV<> postRv("post_", paramSpace);

  QUESO::StatisticalInverseProblem<> ip("", NULL, priorRv, likelihood, postRv);

  // Initial condition of the chain
  QUESO::GslVector paramInitials(paramSpace.zeroVector());
  paramInitials[0] = 1.0;
  paramInitials[1] = 1.0;

  QUESO::GslMatrix proposalCovMatrix(paramSpace.zeroVector());
  for (unsigned int i = 0; i < dim; i++) {
    proposalCovMatrix(i, i) = 0.0001;
  }

  ip.solveWithBayesMetropolisHastings(NULL, paramInitials, &proposalCovMatrix);

  MPI_Finalize();
  return 0;
}
