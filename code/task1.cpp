#include <queso/Environment.h>
#include <mpi.h>

int main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);

  QUESO::FullEnvironment env(MPI_COMM_WORLD, argv[1], "", NULL);

  MPI_Finalize();

  return 0;
}
