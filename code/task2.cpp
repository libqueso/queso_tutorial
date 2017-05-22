#include <queso/Environment.h>
#include <queso/VectorSpace.h>
#include <queso/GslVector.h>
#include <queso/GslMatrix.h>
#include <mpi.h>

int main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);

  QUESO::FullEnvironment env(MPI_COMM_WORLD, argv[1], "", NULL);

  QUESO::VectorSpace<> paramSpace(env, "", 2, NULL);

  QUESO::GslVector x(paramSpace.zeroVector());
  QUESO::GslMatrix A(paramSpace.zeroVector());

  x[0] = 1.0;
  x[1] = 2.0;

  A(0, 0) = 1.0;
  A(0, 1) = 2.0;
  A(1, 0) = 3.0;
  A(1, 1) = 4.0;

  QUESO::GslVector b(A.multiply(x));

  std::cout << "x is:" << std::endl;
  std::cout << x << std::endl;
  std::cout << "A is:" << std::endl;
  std::cout << A << std::endl;
  std::cout << "b is:" << std::endl;
  std::cout << b << std::endl;

  MPI_Finalize();

  return 0;
}
