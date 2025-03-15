/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "dispgrad_field.h"
#include "functions.h"
#include "cli_parser.h"

using namespace ExaDiS;
using namespace ExaDiS::tools;

/*---------------------------------------------------------------------------
 *
 *    Function:     dispgrad_field_grid
 *                  Computes the dislocation stress field on a regular grid
 *
 *-------------------------------------------------------------------------*/
template<class N>
std::vector<Mat33> dispgrad_field_grid(N* net, DispGradIso::Params params, std::vector<int> Ndim,
                                       std::vector<int> Nimg={1,1,1}, bool reg_conv=true)
{
    DispGradFieldGrid<N> dispgrad_field(net, params, Ndim, Nimg, reg_conv);
    
    std::vector<Mat33> gridval(Ndim[0]*Ndim[1]*Ndim[2]);
    for (int kx = 0; kx < Ndim[0]; kx++) {
        for (int ky = 0; ky < Ndim[1]; ky++) {
            for (int kz = 0; kz < Ndim[2]; kz++) {
                int kind = kz+Ndim[2]*ky+Ndim[1]*Ndim[2]*kx;
                gridval[kind] = dispgrad_field.gridval(kx, ky, kz);
            }
        }
    }
    return gridval;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     write_output
 *
 *-------------------------------------------------------------------------*/
void write_output(Cell& cell, std::vector<int> N, Mat33* gridval,
                  std::string field_prefix, std::string filename)
{
    if (N.size() == 1)
        N = {N[0], N[0], N[0]};
    else if (N.size() != 3)
        ExaDiS_fatal("Error: N must be a list of 1 or 3 integers in write_output()\n");
    
    // Open output file
    FILE* fp = fopen(filename.c_str(), "w");
    if (fp == NULL)
        ExaDiS_fatal("Error: cannot open output file %s\n", filename.c_str());
    printf("Writing file: %s\n", filename.c_str());
    
    fprintf(fp, "# Ngrid = %d %d %d\n", N[0], N[1], N[2]);
    fprintf(fp, "# x y z Uxx Uxy Uxz Uyx Uyy Uyz Uzx Uzy Uzz\n");
    
    for (int kx = 0; kx < N[0]; kx++) {
        for (int ky = 0; ky < N[1]; ky++) {
            for (int kz = 0; kz < N[2]; kz++) {
                int kind = kz+N[2]*ky+N[1]*N[2]*kx;
                Mat33* U = &gridval[kind];
                
                Vec3 s(1.0*(kx+0.5)/N[0], 1.0*(ky+0.5)/N[1], 1.0*(kz+0.5)/N[2]);
                Vec3 p = cell.real_position(s);
                
                fprintf(fp, "%e %e %e %e %e %e %e %e %e %e %e %e\n",
                p.x, p.y, p.z,
                U->xx(), U->xy(), U->xz(),
                U->yx(), U->yy(), U->yz(),
                U->zx(), U->zy(), U->zz());
            }
        }
    }
    
    fclose(fp);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     main
 *
 *-------------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
    ExaDiS::Initialize init(argc, argv);
    
    printf("----------------------------------------------------\n");
	printf("ExaDiS compute displacement gradient field\n");
	printf("----------------------------------------------------\n");
    
    std::string datafile;
    int N = 32;
    int Nxyz[3] = {0,0,0};
    double NU = 0.3;
    double a = 1.0;
    int Nim = 1;
    bool pbc[3] = {1,1,1};
    bool reg_conv = 1;
    std::string outname = "displacement_gradient.dat";
    
    cliParser parser(argc, argv);
    parser.add_argument(
        cliParser::STRING, &datafile, 1, "<dataFile>",
        "File containing the dislocation network (.data)"
    );
    parser.add_option(cliParser::REQUIRED, 
        cliParser::DOUBLE, &NU, 1, "-NU", "--poisson_ratio", 
        "Poisson ratio"
    );
    parser.add_option(cliParser::OPTIONAL, 
        cliParser::DOUBLE, &a, 1, "-a", "--core_radius", 
        "Dislocation core radius (b)"
    );
    parser.add_option(cliParser::OPTIONAL, 
        cliParser::INT, &N, 1, "-N", "--ngrid", 
        "Grid resolution [N,N,N] to compute the displacement gradient field"
    );
    parser.add_option(cliParser::OPTIONAL, 
        cliParser::INT, &Nxyz[0], 3, "-Nxyz", "--ngridxyz", 
        "Grid resolution [Nx,Ny,Nz] to compute the displacement gradient field"
    );
    parser.add_option(cliParser::OPTIONAL, 
        cliParser::STRING, &outname, 1, "-o", "--output", 
        "Output file name"
    );
    parser.add_option(cliParser::OPTIONAL, 
        cliParser::INT, &Nim, 1, "-Nimg", "--nimages", 
        "Number of periodic images to compute the displacement gradient field"
    );
    parser.parse(cliParser::VERBOSE);
    parser.add_option(cliParser::OPTIONAL, 
        cliParser::BOOL, &pbc[0], 3, "-pbc", "--pbc_flags", 
        "Periodic boundary conditions along the 3 directions"
    );
    parser.add_option(cliParser::OPTIONAL, 
        cliParser::BOOL, &reg_conv, 1, "-reg", "--regularize_convergence", 
        "Apply regularization of the conditional convergence"
    );
    parser.parse(cliParser::VERBOSE);
    
    SerialDisNet* net = read_paradis(datafile.c_str());
    if (!pbc[0]) net->cell.xpbc = FREE_BOUND;
    if (!pbc[1]) net->cell.ypbc = FREE_BOUND;
    if (!pbc[2]) net->cell.zpbc = FREE_BOUND;
    
    DisNetManager* net_mngr = make_network_manager(net);
    DeviceDisNet* d_net = net_mngr->get_device_network();
    
    Kokkos::Timer timer;
    Kokkos::fence(); timer.reset();
    
    std::vector<int> Ngrid = {N, N, N};
    if (Nxyz[0]*Nxyz[1]*Nxyz[2] > 0)
        Ngrid = {Nxyz[0], Nxyz[1], Nxyz[2]};
    if (Ngrid[0]*Ngrid[1]*Ngrid[2] <= 0)
        ExaDiS_fatal("Error: the grid resolution must be positive\n");
        
    std::vector<int> Nimg = {Nim,Nim,Nim};
    
    printf("Compute field on Ngrid = [%d x %d x %d] (Nimg = %d, reg = %d)...\n",
    Ngrid[0], Ngrid[1], Ngrid[2], Nim, reg_conv);
    
    DispGradIso::Params params(NU, a);
    std::vector<Mat33> dispgradfield = dispgrad_field_grid(d_net, params, Ngrid, Nimg, reg_conv);
    
    Kokkos::fence();
    printf(" Compute time: %e sec\n", timer.seconds());
    
    write_output(net->cell, Ngrid, dispgradfield.data(), "dispgrad", outname);
    
    exadis_delete(net_mngr);
    
    return 0;
}
