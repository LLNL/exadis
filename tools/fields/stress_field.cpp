/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "stress_field.h"
#include "functions.h"
#include "cli_parser.h"

using namespace ExaDiS;
using namespace ExaDiS::tools;

/*---------------------------------------------------------------------------
 *
 *    Function:     stress_field_grid
 *                  Computes the dislocation stress field on a regular grid
 *
 *-------------------------------------------------------------------------*/
template<class N>
std::vector<Mat33> stress_field_grid(N* net, StressIso::Params params,
                                     std::vector<int> Ndim, std::vector<int> Nimg={1,1,1})
{
    StressFieldGrid<N> stress_field(net, params, Ndim, Nimg);
    
    std::vector<Mat33> stressgrid(Ndim[0]*Ndim[1]*Ndim[2]);
    for (int kx = 0; kx < Ndim[0]; kx++) {
        for (int ky = 0; ky < Ndim[1]; ky++) {
            for (int kz = 0; kz < Ndim[2]; kz++) {
                int kind = kz+Ndim[2]*ky+Ndim[1]*Ndim[2]*kx;
                stressgrid[kind] = stress_field.gridval(kx, ky, kz);
            }
        }
    }
    return stressgrid;
}

template<class N>
std::vector<Mat33> stress_field_grid(N* net, StressIso::Params params, int Ngrid)
{
    std::vector<int> Ndim = {Ngrid, Ngrid, Ngrid};
    return stress_field_grid(net, params, Ndim);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     stress_field_points
 *                  Computes the dislocation stress field at a
 *                  list of points
 *
 *-------------------------------------------------------------------------*/
template<class N>
std::vector<Mat33> stress_field_points(N* net, StressIso::Params params, std::vector<Vec3>& p)
{
    FieldPoints<StressIso, N> stress_field(net, params, p);
    
    std::vector<Mat33> stresspoints(p.size());
    for (int i = 0; i < p.size(); i++) {
        stresspoints[i] = stress_field.pointval(i);
    }
    return stresspoints;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     stress_field_point
 *                  Computes the dislocation stress field at a point
 *
 *-------------------------------------------------------------------------*/
template<class N>
Mat33 stress_field_point(N* net, StressIso::Params params, Vec3& p)
{
    return field_point<StressIso, N>(net, params, p);
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
    fprintf(fp, "# x y z Sxx Syy Szz Syz Sxz Sxy\n");
    
    for (int kx = 0; kx < N[0]; kx++) {
        for (int ky = 0; ky < N[1]; ky++) {
            for (int kz = 0; kz < N[2]; kz++) {
                int kind = kz+N[2]*ky+N[1]*N[2]*kx;
                Mat33* S = &gridval[kind];
                
                Vec3 s(1.0*(kx+0.5)/N[0], 1.0*(ky+0.5)/N[1], 1.0*(kz+0.5)/N[2]);
                Vec3 p = cell.real_position(s);
                
                fprintf(fp, "%e %e %e %e %e %e %e %e %e\n", p.x, p.y, p.z,
                S->xx(), S->yy(), S->zz(), S->yz(), S->xz(), S->xy());
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
	printf("ExaDiS compute stress field\n");
	printf("----------------------------------------------------\n");
    
    std::string datafile;
    int N = 32;
    int Nxyz[3] = {0,0,0};
    double MU = 50e9;
    double NU = 0.3;
    double a = 1.0;
    int Nim = 1;
    std::string outname = "stress.dat";
    
    cliParser parser(argc, argv);
    parser.add_argument(
        cliParser::STRING, &datafile, 1, "<dataFile>",
        "File containing the dislocation network (.data)"
    );
    parser.add_option(cliParser::REQUIRED, 
        cliParser::DOUBLE, &MU, 1, "-MU", "--shear_modulus", 
        "Shear modulus (Pa)"
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
        "Grid resolution [N,N,N] to compute the stress field"
    );
    parser.add_option(cliParser::OPTIONAL, 
        cliParser::INT, &Nxyz, 3, "-Nxyz", "--ngridxyz", 
        "Grid resolution [Nx,Ny,Nz] to compute the stress field"
    );
    parser.add_option(cliParser::OPTIONAL, 
        cliParser::STRING, &outname, 1, "-o", "--output", 
        "Output file name"
    );
    parser.add_option(cliParser::OPTIONAL, 
        cliParser::INT, &Nim, 1, "-Nimg", "--nimages", 
        "Number of periodic images to compute the stress field"
    );
    parser.parse(cliParser::VERBOSE);
    
    SerialDisNet* net = read_paradis(datafile.c_str());
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
    
    printf("Compute field on Ngrid = [%d x %d x %d] (Nimg = [%d x %d x %d])...\n",
    Ngrid[0], Ngrid[1], Ngrid[2], Nimg[0], Nimg[1], Nimg[2]);
    
    StressIso::Params params(MU, NU, a);
    std::vector<Mat33> stressgrid = stress_field_grid(d_net, params, Ngrid, Nimg);
    
    Kokkos::fence();
    printf(" Compute time: %e sec\n", timer.seconds());
    
    write_output(net->cell, Ngrid, stressgrid.data(), "stress", outname);
    
    exadis_delete(net_mngr);
    
    return 0;
}
