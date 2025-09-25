/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  ForceFFT module
 *  This module implements the calculation of long-range forces using the
 *  DDD-FFT approach. Stresses are computed on a grid from the continuum
 *  Nye's tensor reprensation of the dislocation network.
 *  Ref: N. Bertin, International Journal of Plasticity 122, 268-284 (2019)
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_FORCE_FFT_H
#define EXADIS_FORCE_FFT_H

#include "force.h"

#ifdef EXADIS_FFT

// FFT libraries
#if defined(KOKKOS_ENABLE_CUDA)
// CUDA
#include <cufft.h>
#include <cuComplex.h>
#define CUFFTWRAPPER( call ) { \
    cufftResult_t err; \
    if ((err = (call)) != CUFFT_SUCCESS) { \
        printf("cuFFT error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
}

#elif defined(KOKKOS_ENABLE_HIP)
// HIP
#include <hipfft/hipfft.h>

#else
// FFTW
#ifdef FFTW
#include <fftw3.h>
#else
#error FFTW must be enabled to use ForceFFT
#endif
#endif

#endif // EXADIS_FFT

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Functions:    FFT utility functions and wrappers for the 
 *                  Kokkos::DefaultExecutionSpace excecution space
 *
 *-------------------------------------------------------------------------*/
enum fft_sign {FFT_FORWARD, FFT_BACKWARD};

struct FFTPlanBase {
    int Nx, Ny, Nz;
};

template <class ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct FFTPlan : FFTPlanBase {
    void initialize(int _Nx, int _Ny, int _Nz) {
        Nx = _Nx; Ny = _Ny; Nz = _Nz;
    }
    void finalize() {}
};

template <class ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct FFT3DTransform
{
    template <typename ViewType>
    FFT3DTransform(FFTPlan<ExecutionSpace>& plan, 
                   const ViewType& in, const ViewType& out,
                   int sign)
    {
#ifdef FFTW
        int FFT_DIR = (sign == FFT_FORWARD) ? FFTW_FORWARD : FFTW_BACKWARD;
        fftw_plan p = fftw_plan_dft_3d(
            plan.Nx, plan.Ny, plan.Nz, 
            reinterpret_cast<fftw_complex*>(in.data()),
            reinterpret_cast<fftw_complex*>(out.data()),
            FFT_DIR, FFTW_ESTIMATE
        );
        fftw_execute(p);
        fftw_destroy_plan(p);
#else
        ExaDiS_fatal("Error: FFTW must be enabled to use ForceFFT in host space\n");
#endif
    }
};

/*---------------------------------------------------------------------------
 *
 *    Functions:    FFT utility functions and wrappers for the 
 *                  Kokkos::Cuda excecution space
 *
 *-------------------------------------------------------------------------*/
#if defined(EXADIS_FFT) && defined(KOKKOS_ENABLE_CUDA)
template<>
struct FFTPlan<Kokkos::Cuda> : FFTPlanBase {
    bool plan_created = 0;
    cufftHandle* plan = nullptr;
    KOKKOS_INLINE_FUNCTION FFTPlan() {}
    KOKKOS_INLINE_FUNCTION FFTPlan(const FFTPlan& p) : plan(p.plan) { plan_created = 0; }
    void initialize(int _Nx, int _Ny, int _Nz) {
        Nx = _Nx; Ny = _Ny; Nz = _Nz;
        if (plan) delete plan;
        plan = new cufftHandle();
        CUFFTWRAPPER(cufftPlan3d(plan, Nx, Ny, Nz, CUFFT_Z2Z));
        plan_created = 1;
    }
    void finalize() {
        if (plan_created && plan) cufftDestroy(*plan);
    }
};

template<>
struct FFT3DTransform<Kokkos::Cuda>
{
    template <typename ViewType>
    FFT3DTransform(FFTPlan<Kokkos::Cuda>& plan, 
                   const ViewType& in, const ViewType& out, 
                   int sign)
    {
        int FFT_DIR = (sign == FFT_FORWARD) ? CUFFT_FORWARD : CUFFT_INVERSE;
        CUFFTWRAPPER(cufftExecZ2Z( //double precision
            *plan.plan,
            reinterpret_cast<cufftDoubleComplex*>(in.data()),
            reinterpret_cast<cufftDoubleComplex*>(out.data()),
            FFT_DIR
        ));
    }
};
#endif

/*---------------------------------------------------------------------------
 *
 *    Functions:    FFT utility functions and wrappers for the 
 *                  Kokkos::HIP excecution space
 *
 *-------------------------------------------------------------------------*/
#if defined(EXADIS_FFT) && defined(KOKKOS_ENABLE_HIP)
template<>
struct FFTPlan<Kokkos::HIP> : FFTPlanBase {
    bool plan_created = 0;
    hipfftHandle* plan = nullptr;
    KOKKOS_INLINE_FUNCTION FFTPlan() {}
    KOKKOS_INLINE_FUNCTION FFTPlan(const FFTPlan& p) : plan(p.plan) { plan_created = 0; }
    void initialize(int _Nx, int _Ny, int _Nz) {
        Nx = _Nx; Ny = _Ny; Nz = _Nz;
        if (plan) delete plan;
        plan = new hipfftHandle();
        hipfftPlan3d(plan, Nx, Ny, Nz, HIPFFT_Z2Z);
        plan_created = 1;
    }
    void finalize() {
        if (plan_created && plan) hipfftDestroy(*plan);
    }
};

template<>
struct FFT3DTransform<Kokkos::HIP>
{
    template <typename ViewType>
    FFT3DTransform(FFTPlan<Kokkos::HIP>& plan, 
                   const ViewType& in, const ViewType& out, 
                   int sign)
    {
        int FFT_DIR = (sign == FFT_FORWARD) ? HIPFFT_FORWARD : HIPFFT_BACKWARD;
        hipfftExecZ2Z( //double precision
            *plan.plan,
            reinterpret_cast<hipfftDoubleComplex*>(in.data()),
            reinterpret_cast<hipfftDoubleComplex*>(out.data()),
            FFT_DIR
        );
    }
};
#endif

/*---------------------------------------------------------------------------
 *
 *    Functions:    Complex type definition and utility functions
 *
 *-------------------------------------------------------------------------*/
typedef Kokkos::complex<double> complex;

KOKKOS_INLINE_FUNCTION
complex conj(const complex &a) {
    return complex(a.real(), -a.imag());
}

KOKKOS_INLINE_FUNCTION
void InverseComplex33(complex M[3][3])
{
    complex a, b, c, d, e, f, g, h, i, com;
    a = M[0][0]; b = M[0][1]; c = M[0][2];
    d = M[1][0]; e = M[1][1]; f = M[1][2];
    g = M[2][0]; h = M[2][1]; i = M[2][2];

    com = 1.0/(a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g);

    M[0][0] = com*(e*i - f*h);
    M[0][1] = com*(c*h - b*i);
    M[0][2] = com*(b*f - c*e);
    M[1][0] = com*(f*g - d*i);
    M[1][1] = com*(a*i - c*g);
    M[1][2] = com*(c*d - a*f);
    M[2][0] = com*(d*h - e*g);
    M[2][1] = com*(b*g - a*h);
    M[2][2] = com*(a*e - b*d);
}

/*---------------------------------------------------------------------------
 *
 *    Class:        ForceFFT
 *                  Long-range force contribution computed from Nye's tensor
 *                  using the DDD-FFT spectral method.
 *                  See Bertin, IJP (2019); Bertin et al., MSMSE (2015)
 *
 *-------------------------------------------------------------------------*/
class ForceFFT : public Force {
private:
    FFTPlan<> plan;
    int Ngrid[3], Ngrid3;
    double rcgrid, rcnei;
    
    typedef Kokkos::View<complex***, Kokkos::LayoutRight, T_memory_space> T_grid;
    T_grid gridval[9];
    T_grid w;
    double wsum;
    
#if !EXADIS_FULL_UNIFIED_MEMORY
    T_grid::HostMirror h_gridval[6];
    bool host_synced;
#endif
    
    enum gridcomps {
        GRID_XX, GRID_XY, GRID_XZ,
        GRID_YX, GRID_YY, GRID_YZ,
        GRID_ZX, GRID_ZY, GRID_ZZ,
    };
    int stress_comps[6] = {
        GRID_XX, GRID_XY, GRID_XZ, GRID_YY, GRID_YZ, GRID_ZZ
    };
    
    int GREEN_OP = 0; // Modified Green operator
    
    double stress_fact = 1.0;//1e-9;
    double C[3][3][3][3];
    
    DeviceDisNet* d_net;
    Mat33 H0, Hk, Hkinv;
    Cell cell;
    Vec3 Hs;
    double Vs;
    
    int TIMER_FFTPRECOMP, TIMER_FFTALPHA, TIMER_FFTSTRESS, TIMER_FFTTRANSFORMS;

public:
    struct Params {
        int Ngrid[3];
        Params() { Ngrid[0] = Ngrid[1] = Ngrid[2] = -1; }
        Params(int _Ngrid) { Ngrid[0] = Ngrid[1] = Ngrid[2] = _Ngrid; }
        Params(int Nx, int Ny, int Nz) { Ngrid[0] = Nx; Ngrid[1] = Ny; Ngrid[2] = Nz; }
    };
    
    ForceFFT(System* system, Params params)
    {
#ifndef EXADIS_FFT
        ExaDiS_fatal("Error: EXADIS_FFT must be enabled to use ForceFFT\n");
#endif
        initialize(system, params);
        
        TIMER_FFTPRECOMP = system->add_timer("ForceFFT precompute");
        TIMER_FFTALPHA = system->add_timer("ForceFFT precompute alpha");
        TIMER_FFTSTRESS = system->add_timer("ForceFFT precompute stress");
        TIMER_FFTTRANSFORMS = system->add_timer("ForceFFT transforms");
    }
    
    void initialize(System* system, Params params)
    {
        for (int i = 0; i < 3; i++) {
            Ngrid[i] = params.Ngrid[i];
            if (Ngrid[i] <= 0)
                ExaDiS_fatal("Error: undefined grid size in ForceFFT\n");
        }
        Ngrid3 = Ngrid[0]*Ngrid[1]*Ngrid[2];
        plan.initialize(Ngrid[0], Ngrid[1], Ngrid[2]);
        
        for (int i = 0; i < 9; i++)
            Kokkos::resize(gridval[i], Ngrid[0], Ngrid[1], Ngrid[2]);
            
#if !EXADIS_FULL_UNIFIED_MEMORY
        host_synced = false;
#endif
        
        double MU = system->params.MU * stress_fact;
        double NU = system->params.NU;
        double LA = 2.0*MU*NU/(1.0-2.0*NU);

        Mat33 delta = Mat33().eye();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    for (int l = 0; l < 3; l++)
                        C[i][j][k][l] = LA*(delta[i][j]*delta[k][l]) +
                        MU*(delta[i][k]*delta[j][l]+delta[i][l]*delta[j][k]);
            
        SerialDisNet* network = system->get_serial_network();
        if (network->cell.xpbc != PBC_BOUND ||
            network->cell.ypbc != PBC_BOUND ||
            network->cell.zpbc != PBC_BOUND)
            ExaDiS_fatal("Error: ForceFFT requires full periodic boundary conditions\n");
        
        initialize_spectral_core(system, network);
    }
    
    struct TagComputeW {};
    struct TagNormalizeW {};
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagComputeW, const int& kx, const int& ky, const int& kz, double &psum) const {
        int kxmax = Ngrid[0]/2 + Ngrid[0] % 2;
        int kymax = Ngrid[1]/2 + Ngrid[1] % 2;
        int kzmax = Ngrid[2]/2 + Ngrid[2] % 2;
        int kxx = kx; if (kxx >= kxmax) kxx -= Ngrid[0];
        int kyy = ky; if (kyy >= kymax) kyy -= Ngrid[1];
        int kzz = kz; if (kzz >= kzmax) kzz -= Ngrid[2];
        
        Vec3 r = Hk * Vec3(kxx, kyy, kzz);
        double r2 = r.norm2();
        double aw2 = rcgrid*rcgrid;
        double wr = 15.0*aw2*aw2/8.0/M_PI/pow(r2+aw2, 3.5);
        w(kx, ky, kz) = complex(wr, 0.0);
        
        psum += wr;
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagNormalizeW, const int& kx, const int& ky, const int& kz) const {
        w(kx, ky, kz) /= wsum;
    }
    
    template<class N>
    void initialize_spectral_core(System* system, N* net)
    {
        // Non-singular kernel
        H0 = net->cell.H;
        Hk = Mat33().set_columns(
            1.0/Ngrid[0] * H0.colx(),
            1.0/Ngrid[1] * H0.coly(),
            1.0/Ngrid[2] * H0.colz()
        );
        Hkinv = Hk.inverse();
        
        Vec3 H;
        H.x = H0.xx() / Ngrid[0];
        H.y = H0.yy() / Ngrid[1];
        H.z = H0.zz() / Ngrid[2];
        double Hmax = fmax(fmax(H.x, H.y), H.z);
        
        // Check voxel aspect ratio...
        if (H.y/H.x < 0.5 || H.y/H.x > 1.5 || 
            H.z/H.y < 0.5 || H.z/H.y > 1.5 || 
            H.x/H.z < 0.5 || H.x/H.z > 1.5)
            ExaDiS_log("Warning: high aspect ratio voxels in ForceFFT\n"
            " Consider adjusting the simulation parameters\n");
        
        // Grid core radius
        rcgrid = 2.0*Hmax;
        
        // Neighbor cutoff
        if (system->params.a >= rcgrid) {
            rcgrid = system->params.a;
            rcnei = 0.0;
        } else {
            rcnei = 4.0*rcgrid;
        }
        
        // Register neighbor cutoff required for the grid
        system->register_neighbor_cutoff(fmax(0.5*Hmax, system->params.maxseg));
        
        Kokkos::resize(w, Ngrid[0], Ngrid[1], Ngrid[2]);
        wsum = 0.0;
        
        Kokkos::parallel_reduce("ForceFFT::ComputeW",
            Kokkos::MDRangePolicy<TagComputeW,Kokkos::Rank<3>>(
                {0, 0, 0}, {Ngrid[0], Ngrid[1], Ngrid[2]}
            ), *this, wsum
        );
        Kokkos::fence();
        
        Kokkos::parallel_for("ForceFFT::NormalizeW",
            Kokkos::MDRangePolicy<TagNormalizeW,Kokkos::Rank<3>>(
                {0, 0, 0}, {Ngrid[0], Ngrid[1], Ngrid[2]}
            ), *this
        );
        Kokkos::fence();
        
        FFT3DTransform(plan, w, w, FFT_FORWARD);
        Kokkos::fence();
    }
    
    double get_neighbor_cutoff() { return rcnei; }
    double get_rcgrid() { return rcgrid; }
    
    KOKKOS_INLINE_FUNCTION
    double alpha_box_segment(const Vec3 &p1, const Vec3 &t, const double &L,
                             const Vec3 &bc, const Vec3 &H) const
    {
        double eps = 1e-10;
        Vec3 tinv(1.0/(t.x+eps), 1.0/(t.y+eps), 1.0/(t.z+eps));
        
        Vec3 t1 = bc - H - p1;
        t1.x *= tinv.x; t1.y *= tinv.y; t1.z *= tinv.z;
        Vec3 t2 = bc + H - p1;
        t2.x *= tinv.x; t2.y *= tinv.y; t2.z *= tinv.z;
        
        Vec3 tmin(fmin(t1.x, t2.x), fmin(t1.y, t2.y), fmin(t1.z, t2.z));
        Vec3 tmax(fmax(t1.x, t2.x), fmax(t1.y, t2.y), fmax(t1.z, t2.z));
        
        double cmin = fmax(fmax(tmin.x, tmin.y), tmin.z);
        double cmax = fmin(fmin(tmax.x, tmax.y), tmax.z);
        
        cmin = fmax(cmin, 0.0);
        cmax = fmin(cmax, L);
        
        Vec3 x1 = p1 + cmin * t;
        Vec3 x2 = p1 + cmax * t;
        double lx = (x2-x1).norm();
        
        double W = 0.0;
        if (cmin <= cmax && lx >= eps) {
            // Parametrize segment
            Vec3 R = bc-x1;
            double dr = dot(R, t);
            Vec3 drt = dr * t;
            Vec3 d = R-drt;
            Vec3 x0 = x1+drt;
            double s1 = dot(x1-x0, t);
            double s2 = dot(x2-x0, t);
            
            Vec3 s;
            s.x = (fabs(t.x) < eps) ? s2+1 : d.x/t.x;
            s.y = (fabs(t.y) < eps) ? s2+1 : d.y/t.y;
            s.z = (fabs(t.z) < eps) ? s2+1 : d.z/t.z;
            
            Vec3 sk;
            sk.x = fmin(fmax(s.x, s1), s2);
            sk.y = fmin(fmax(s.y, s1), s2);
            sk.z = fmin(fmax(s.z, s1), s2);
            
            // First term
            W = s2-s1;
            
            // Second term Ai
            for (int k = 0; k < 3; k++) {
                W -= 1.0/H[k]*fabs(0.5*(sk[k]-s1)*(2.0*d[k]-t[k]*(sk[k]+s1)));
                W -= 1.0/H[k]*fabs(0.5*(s2-sk[k])*(2.0*d[k]-t[k]*(s2+sk[k])));
            }
    
            // Third term Bij
            for (int k = 0; k < 3; k++) {
                int i1 = k;
                int i2 = (k+1) % 3;
                double sm1 = fmin(sk[i1], sk[i2]);
                double sm2 = fmax(sk[i1], sk[i2]);
                double ss[4] = {s1, sm1, sm2, s2};
                double B = 0.0;
                for (int l = 0; l < 3; l++) {
                    double ss1 = ss[l];
                    double ss2 = ss[l+1];
                    B += fabs(d[i1]*d[i2]*(ss2-ss1)
                    -0.5*(d[i1]*t[i2]+d[i2]*t[i1])*(ss2*ss2-ss1*ss1)
                    +1.0/3.0*t[i1]*t[i2]*(ss2*ss2*ss2-ss1*ss1*ss1));
                }
                W += 1.0/H[i1]/H[i2]*B;
            }
            
            // Forth term Cijk
            double sm1 = fmin(sk[0], sk[1]);
            double sm2 = fmax(sk[0], sk[1]);
            double sn1 = fmin(sm1, sk[2]);
            double si2 = fmax(sm1, sk[2]);
            double sn2 = fmin(sm2, si2);
            double sn3 = fmax(sm2, si2);
            double ss[5] = {s1, sn1, sn2, sn3, s2};
            double C = 0.0;
            for (int l = 0; l < 4; l++) {
                double ss1 = ss[l];
                double ss2 = ss[l+1];
                C += fabs(d[0]*d[1]*d[2]*(ss2-ss1)
                -0.5*(t[0]*d[1]*d[2]+d[0]*t[1]*d[2]+d[0]*d[1]*t[2])*(ss2*ss2-ss1*ss1)
                +1.0/3.0*(d[0]*t[1]*t[2]+t[0]*d[1]*t[2]+t[0]*t[1]*d[2])*(ss2*ss2*ss2-ss1*ss1*ss1)
                -0.25*(t[0]*t[1]*t[2])*(ss2*ss2*ss2*ss2-ss1*ss1*ss1*ss1));
            }
            W -= 1.0/H[0]/H[1]/H[2]*C;
        }
        
        return W;
    }
    
    struct TagComputeAlpha {};
    struct TagStressFromAlpha {};
    struct TagNormalizeStress {};
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagComputeAlpha, const int& i) const {
        auto nodes = d_net->get_nodes();
        auto segs = d_net->get_segs();
        
        int n1 = segs[i].n1;
        int n2 = segs[i].n2;
        Vec3 b = segs[i].burg;
        Vec3 r1 = nodes[n1].pos;
        Vec3 r2 = cell.pbc_position(r1, nodes[n2].pos);
        
        Vec3 t = r2-r1;
        double L = t.norm();
        if (L > 1e-10) {
            
            r1 = cell.scaled_position(r1);
            r2 = cell.scaled_position(r2);
            double Ls = (r2-r1).norm();
            double scale = L / Ls;
            
            t = t.normalized();
            
            double xmin = fmin(r1.x, r2.x) - 0.5*Hs.x;
            double xmax = fmax(r1.x, r2.x) - 0.5*Hs.x;
            double ymin = fmin(r1.y, r2.y) - 0.5*Hs.y;
            double ymax = fmax(r1.y, r2.y) - 0.5*Hs.y;
            double zmin = fmin(r1.z, r2.z) - 0.5*Hs.z;
            double zmax = fmax(r1.z, r2.z) - 0.5*Hs.z;

            int imin = floor(xmin/Hs.x);
            int imax = floor(xmax/Hs.x) + 1;
            int jmin = floor(ymin/Hs.y);
            int jmax = floor(ymax/Hs.y) + 1;
            int kmin = floor(zmin/Hs.z);
            int kmax = floor(zmax/Hs.z) + 1;
            
            for (int ib = imin; ib <= imax; ib++) {
                for (int jb = jmin; jb <= jmax; jb++) {
                    for (int kb = kmin; kb <= kmax; kb++) {
                        
                        Vec3 bc;
                        bc.x = (ib+0.5)*Hs.x;
                        bc.y = (jb+0.5)*Hs.y;
                        bc.z = (kb+0.5)*Hs.z;

                        double W = scale * alpha_box_segment(r1, t, Ls, bc, Hs);
                        
                        // Box index
                        int kx = ib % Ngrid[0]; if (kx < 0) kx += Ngrid[0];
                        int ky = jb % Ngrid[1]; if (ky < 0) ky += Ngrid[1];
                        int kz = kb % Ngrid[2]; if (kz < 0) kz += Ngrid[2];
                        
                        Kokkos::atomic_add(&gridval[0](kx, ky, kz), W/Vs*b.x*t.x);
                        Kokkos::atomic_add(&gridval[1](kx, ky, kz), W/Vs*b.x*t.y);
                        Kokkos::atomic_add(&gridval[2](kx, ky, kz), W/Vs*b.x*t.z);
                        Kokkos::atomic_add(&gridval[3](kx, ky, kz), W/Vs*b.y*t.x);
                        Kokkos::atomic_add(&gridval[4](kx, ky, kz), W/Vs*b.y*t.y);
                        Kokkos::atomic_add(&gridval[5](kx, ky, kz), W/Vs*b.y*t.z);
                        Kokkos::atomic_add(&gridval[6](kx, ky, kz), W/Vs*b.z*t.x);
                        Kokkos::atomic_add(&gridval[7](kx, ky, kz), W/Vs*b.z*t.y);
                        Kokkos::atomic_add(&gridval[8](kx, ky, kz), W/Vs*b.z*t.z);
                    }
                }
            }
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagStressFromAlpha, const int& kx, const int& ky, const int& kz) const {
        // Voxel id and wave-vector coordinates
        int kxmax = Ngrid[0]/2 + Ngrid[0] % 2;
        int kymax = Ngrid[1]/2 + Ngrid[1] % 2;
        int kzmax = Ngrid[2]/2 + Ngrid[2] % 2;
        int kxx = kx; if (kxx >= kxmax) kxx -= Ngrid[0];
        int kyy = ky; if (kyy >= kymax) kyy -= Ngrid[1];
        int kzz = kz; if (kzz >= kzmax) kzz -= Ngrid[2];
        
        bool zero = false;
        if (kxx == 0 && kyy == 0 && kzz == 0) {
            zero = true;
        }
        
        complex wk = w(kx, ky, kz);
        
        complex T[3][3];
        T[0][0] = wk * gridval[GRID_XX](kx, ky, kz);
        T[0][1] = wk * gridval[GRID_XY](kx, ky, kz);
        T[0][2] = wk * gridval[GRID_XZ](kx, ky, kz);
        T[1][0] = wk * gridval[GRID_YX](kx, ky, kz);
        T[1][1] = wk * gridval[GRID_YY](kx, ky, kz);
        T[1][2] = wk * gridval[GRID_YZ](kx, ky, kz);
        T[2][0] = wk * gridval[GRID_ZX](kx, ky, kz);
        T[2][1] = wk * gridval[GRID_ZY](kx, ky, kz);
        T[2][2] = wk * gridval[GRID_ZZ](kx, ky, kz);
        
        double xk[3];
        xk[0] = 2.0*M_PI*kxx/Ngrid[0];
        xk[1] = 2.0*M_PI*kyy/Ngrid[1];
        xk[2] = 2.0*M_PI*kzz/Ngrid[2];
        
        complex bxk[3];
        // iGreenOp=0: k=i*q
        if (GREEN_OP == 0) {
            bxk[0] = complex(0.0, xk[0]);
            bxk[1] = complex(0.0, xk[1]);
            bxk[2] = complex(0.0, xk[2]);
        // iGreenOp=1: k=i*sin(q) (C)
        } else if (GREEN_OP == 1) {
            bxk[0] = complex(0.0, sin(xk[0]));
            bxk[1] = complex(0.0, sin(xk[1]));
            bxk[2] = complex(0.0, sin(xk[2]));
        // iGreenOp=2: k=e^(i*q)-1 (W)
        } else if (GREEN_OP == 2) {
            bxk[0] = complex(cos(xk[0])-1.0, sin(xk[0]));
            bxk[1] = complex(cos(xk[1])-1.0, sin(xk[1]));
            bxk[2] = complex(cos(xk[2])-1.0, sin(xk[2]));
        // iGreenOp=3: k(R), rotated scheme
        } else if (GREEN_OP == 3) {
            if (kxx == -Ngrid[0]/2 || kyy == -Ngrid[1]/2 || kzz == -Ngrid[2]/2) {
                zero = true;
            }
            complex fact0(1.0+cos(xk[0]), sin(xk[0]));
            complex fact1(1.0+cos(xk[1]), sin(xk[1]));
            complex fact2(1.0+cos(xk[2]), sin(xk[2]));
            complex fact = fact0*fact1*fact2;
            bxk[0] = 0.25*fact*complex(0.0, tan(0.5*xk[0]));
            bxk[1] = 0.25*fact*complex(0.0, tan(0.5*xk[1]));
            bxk[2] = 0.25*fact*complex(0.0, tan(0.5*xk[2]));
        }
        
        complex cxk[3];
        for (int i = 0; i < 3; i++) {
            cxk[i] = complex(0.0, 0.0);
            for (int j = 0; j < 3; j++)
                cxk[i] += Hkinv[j][i] * bxk[j];
        }
        
        double nxk2 = cxk[0].real()*cxk[0].real() + cxk[0].imag()*cxk[0].imag() +
                      cxk[1].real()*cxk[1].real() + cxk[1].imag()*cxk[1].imag() +
                      cxk[2].real()*cxk[2].real() + cxk[2].imag()*cxk[2].imag();
        double nxk2inv = 1.0/nxk2;
        
        complex K[3][3];
        K[0][0] = -nxk2inv*(T[0][1]*cxk[2]-T[0][2]*cxk[1]);
        K[0][1] = -nxk2inv*(T[0][2]*cxk[0]-T[0][0]*cxk[2]);
        K[0][2] = -nxk2inv*(T[0][0]*cxk[1]-T[0][1]*cxk[0]);

        K[1][0] = -nxk2inv*(T[1][1]*cxk[2]-T[1][2]*cxk[1]);
        K[1][1] = -nxk2inv*(T[1][2]*cxk[0]-T[1][0]*cxk[2]);
        K[1][2] = -nxk2inv*(T[1][0]*cxk[1]-T[1][1]*cxk[0]);

        K[2][0] = -nxk2inv*(T[2][1]*cxk[2]-T[2][2]*cxk[1]);
        K[2][1] = -nxk2inv*(T[2][2]*cxk[0]-T[2][0]*cxk[2]);
        K[2][2] = -nxk2inv*(T[2][0]*cxk[1]-T[2][1]*cxk[0]);

        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            T[i][j] = complex(0.0, 0.0);
            for (int k = 0; k < 3; k++) for (int l = 0; l < 3; l++) {
                T[i][j] += C[i][j][k][l]*K[k][l];
            }
        }

        complex A[3][3];
        for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) {
            A[i][k] = complex(0.0, 0.0);
            for (int j = 0; j < 3; j++) for (int l = 0; l < 3; l++) {
                A[i][k] += C[k][j][i][l]*cxk[j]*conj(cxk[l]);
            }
        }
        if (!zero) InverseComplex33(A);
        
        complex G[3][3][3][3];
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) for (int l = 0; l < 3; l++) {
                G[i][j][k][l] = A[i][k]*cxk[j]*conj(cxk[l]);
            }
        }

        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            A[i][j] = complex(0.0, 0.0);
            for (int k = 0; k < 3; k++) for (int l = 0; l < 3; l++) {
                A[i][j] -= G[i][j][k][l]*T[k][l];
            }
        }
        
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            T[i][j] = complex(0.0, 0.0);
            for (int k = 0; k < 3; k++) for (int l = 0; l < 3; l++) {
                T[i][j] += C[i][j][k][l]*(A[k][l] + K[k][l]);
            }
        }
        
        if (!zero) {
            gridval[GRID_XX](kx, ky, kz) = T[0][0];
            gridval[GRID_XY](kx, ky, kz) = T[0][1];
            gridval[GRID_XZ](kx, ky, kz) = T[0][2];
            gridval[GRID_YY](kx, ky, kz) = T[1][1];
            gridval[GRID_YZ](kx, ky, kz) = T[1][2];
            gridval[GRID_ZZ](kx, ky, kz) = T[2][2];
        } else {
            gridval[GRID_XX](kx, ky, kz) = complex(0.0, 0.0);
            gridval[GRID_XY](kx, ky, kz) = complex(0.0, 0.0);
            gridval[GRID_XZ](kx, ky, kz) = complex(0.0, 0.0);
            gridval[GRID_YY](kx, ky, kz) = complex(0.0, 0.0);
            gridval[GRID_YZ](kx, ky, kz) = complex(0.0, 0.0);
            gridval[GRID_ZZ](kx, ky, kz) = complex(0.0, 0.0);
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagNormalizeStress, const int& kx, const int& ky, const int& kz) const {
        for (int i = 0; i < 6; i++)
            gridval[stress_comps[i]](kx, ky, kz) /= (stress_fact*Ngrid3);
    }
    
    void compute_alpha(System* system)
    {
        d_net = system->get_device_network();
        cell = d_net->cell;
        
        if ((cell.H.colx() - H0.colx()).norm2() > 1.0 ||
            (cell.H.coly() - H0.coly()).norm2() > 1.0 ||
            (cell.H.colz() - H0.colz()).norm2() > 1.0) {
            initialize_spectral_core(system, d_net);
        }
        
        Hs.x = 1.0/Ngrid[0];
        Hs.y = 1.0/Ngrid[1];
        Hs.z = 1.0/Ngrid[2];
        Vs = cell.H.det()/Ngrid[0]/Ngrid[1]/Ngrid[2];
        
        Kokkos::fence();
        system->devtimer[TIMER_FFTALPHA].start();
        
        for (int i = 0; i < 9; i++)
            Kokkos::deep_copy(gridval[i], 0.0);
        
#if !EXADIS_FULL_UNIFIED_MEMORY
        host_synced = false;
#endif
        
        Kokkos::parallel_for("ForceFFT::ComputeAlpha",
            Kokkos::RangePolicy<TagComputeAlpha>(0, d_net->Nsegs_local), *this
        );
        Kokkos::fence();
        system->devtimer[TIMER_FFTALPHA].stop();
    }
    
    void compute_stress_from_alpha(System *system)
    {
        Kokkos::fence();
        system->devtimer[TIMER_FFTSTRESS].start();
        
        system->devtimer[TIMER_FFTTRANSFORMS].start();
        for (int i = 0; i < 9; i++)
            FFT3DTransform(plan, gridval[i], gridval[i], FFT_FORWARD);
        Kokkos::fence();
        system->devtimer[TIMER_FFTTRANSFORMS].stop();
        
        Kokkos::parallel_for("ForceFFT::StressFromAlpha",
            Kokkos::MDRangePolicy<TagStressFromAlpha,Kokkos::Rank<3>>(
                {0, 0, 0}, {Ngrid[0], Ngrid[1], Ngrid[2]}
            ), *this
        );
        Kokkos::fence();
        
        system->devtimer[TIMER_FFTTRANSFORMS].start();
        for (int i = 0; i < 6; i++)
            FFT3DTransform(plan, gridval[stress_comps[i]], gridval[stress_comps[i]], FFT_BACKWARD);
        Kokkos::fence();
        system->devtimer[TIMER_FFTTRANSFORMS].stop();
        
        Kokkos::parallel_for("ForceFFT::NormalizeStress",
            Kokkos::MDRangePolicy<TagNormalizeStress,Kokkos::Rank<3>>(
                {0, 0, 0}, {Ngrid[0], Ngrid[1], Ngrid[2]}
            ), *this
        );
        Kokkos::fence();
        system->devtimer[TIMER_FFTSTRESS].stop();
    }
    
    inline void synchronize_stress_gridval()
    {
#if !EXADIS_FULL_UNIFIED_MEMORY
        if (!host_synced) {
            for (int i = 0; i < 6; i++) {
                h_gridval[i] = Kokkos::create_mirror_view(gridval[stress_comps[i]]);
                Kokkos::deep_copy(h_gridval[i], gridval[stress_comps[i]]);
            }
            host_synced = true;
        }
#endif
    }
    
    std::vector<Mat33> export_stress_gridval()
    {
        synchronize_stress_gridval();
        std::vector<Mat33> stress_gridval(Ngrid3);
        int ind = 0;
        for (int kx = 0; kx < Ngrid[0]; kx++) {
            for (int ky = 0; ky < Ngrid[1]; ky++) {
                for (int kz = 0; kz < Ngrid[2]; kz++) {
                    double S[6];
                    for (int i = 0; i < 6; i++)
                        S[i] = get_stress_gridval<SerialDisNet>(i, kx, ky, kz);
                    // S: GRID_XX, GRID_XY, GRID_XZ, GRID_YY, GRID_YZ, GRID_ZZ
                    // Mat33: xx, yy, zz, xy, xz, yz
                    stress_gridval[ind++] = Mat33().symmetric(S[0], S[3], S[5], S[1], S[2], S[4]);
                }
            }
        }
        return stress_gridval;
    }
    
    std::vector<Mat33> interpolate_stress_array(std::vector<Vec3>& p)
    {
        synchronize_stress_gridval();
        std::vector<Mat33> stress(p.size());
        for (size_t i = 0; i < p.size(); i++)
            stress[i] = interpolate_stress<SerialDisNet>(p[i]);
        return stress;
    }
    
#if EXADIS_FULL_UNIFIED_MEMORY
    template<class N>
    KOKKOS_INLINE_FUNCTION
    double get_stress_gridval(int i, int kx, int ky, int kz) {
        return gridval[stress_comps[i]](kx, ky, kz).real();
    }
#else
    template<class N>
    KOKKOS_INLINE_FUNCTION
    double get_stress_gridval(int i, int kx, int ky, int kz) {
        if constexpr (std::is_same<N, SerialDisNet>::value) {
            return h_gridval[i](kx, ky, kz).real();
        } else {
            return gridval[stress_comps[i]](kx, ky, kz).real();
        }
    }
#endif
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    Mat33 interpolate_stress(const Vec3& p)
    {
        Vec3 s = cell.scaled_position(p);
        
        double q[3];
        q[0] = s.x * Ngrid[0] - 0.5;
        q[1] = s.y * Ngrid[1] - 0.5;
        q[2] = s.z * Ngrid[2] - 0.5;
        
        int g[3];
        g[0] = (int)floor(q[0]);
        g[1] = (int)floor(q[1]);
        g[2] = (int)floor(q[2]);

        double xi[3];
        xi[0] = 2.0*(q[0]-g[0]) - 1.0;
        xi[1] = 2.0*(q[1]-g[1]) - 1.0;
        xi[2] = 2.0*(q[2]-g[2]) - 1.0;

        // Determine elements for interpolation and apply PBC
        int ind1d[3][2];
        for (int i = 0; i < 2; i++) {
            ind1d[0][i] = (g[0]+i)%Ngrid[0];
            if (ind1d[0][i] < 0) ind1d[0][i] += Ngrid[0];
            ind1d[1][i] = (g[1]+i)%Ngrid[1];
            if (ind1d[1][i] < 0) ind1d[1][i] += Ngrid[1];
            ind1d[2][i] = (g[2]+i)%Ngrid[2];
            if (ind1d[2][i] < 0) ind1d[2][i] += Ngrid[2];
        }

        // 1d shape functions
        double phi1d[3][2];
        for (int i = 0; i < 3; i++) {
            phi1d[i][0] = 0.5*(1.0-xi[i]);
            phi1d[i][1] = 0.5*(1.0+xi[i]);
        }

        // 3d shape functions and indices
        double S[6];
        for (int l = 0; l < 6; l++) {
            S[l] = 0.0;
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 2; k++) {
                        double phi = phi1d[0][i]*phi1d[1][j]*phi1d[2][k];
                        S[l] += phi * get_stress_gridval<N>(l, ind1d[0][i], ind1d[1][j], ind1d[2][k]);
                    }
                }
            }
        }
        
        // S: GRID_XX, GRID_XY, GRID_XZ, GRID_YY, GRID_YZ, GRID_ZZ
        // Mat33: xx, yy, zz, xy, xz, yz
        return Mat33().symmetric(S[0], S[3], S[5], S[1], S[2], S[4]);
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    SegForce segment_force(System* system, N* net, int i)
    {
        auto nodes = net->get_nodes();
        auto segs = net->get_segs();
        auto cell = net->cell;
        
        int n1 = segs[i].n1;
        int n2 = segs[i].n2;
        Vec3 b = segs[i].burg;
        Vec3 r1 = nodes[n1].pos;
        Vec3 r2 = cell.pbc_position(r1, nodes[n2].pos);
        Vec3 dr = r2-r1;
        double L = dr.norm();
        
        Vec3 f1(0.0), f2(0.0);
        if (L > 1.e-10) {
            
            int intPoints = 3;
            double positions[3], weights[3];
            positions[0] = -0.774596669241483;
            positions[1] = 0.0;
            positions[2] = -positions[0];
            weights[0] = 0.5*5.0/9.0;
            weights[1] = 0.5*8.0/9.0;
            weights[2] = weights[0];
            
            Vec3 pmid = 0.5*(r1+r2);
            Vec3 pspan = 0.5*dr;
            
            for (int j = 0; j < intPoints; j++) {

                double pos = positions[j];
                Vec3 p = pmid + pos*pspan;

                Mat33 S = interpolate_stress<N>(p);
                Vec3 sigb = S * b;
                Vec3 fLinv = cross(sigb, pspan);

                double temp = weights[j]*positions[j];
                double mult = weights[j]+temp;
                f2 += mult * fLinv;

                mult = weights[j]-temp;
                f1 += mult * fLinv;
            }
        }
        
        return SegForce(f1, f2);
    }
    
#if defined(EXADIS_USE_COMPUTE_MAPS)
    const bool use_map = true;
#else
    const bool use_map = false;
#endif
    Kokkos::View<SegForce*> fmap;
    
    struct TagComputeForce {};
    struct TagMapForce {};
    
    template<class N>
    struct AddSegmentForce {
        System* system;
        ForceFFT* force;
        N* net;
        AddSegmentForce(System* _system, ForceFFT* _force, N* _net) : 
        system(_system), force(_force), net(_net) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator()(TagComputeForce, const int& i) const {
            auto nodes = net->get_nodes();
            auto segs = net->get_segs();
            int n1 = segs[i].n1;
            int n2 = segs[i].n2;
            
            SegForce fseg = force->segment_force(system, net, i);
            
            if (force->use_map) {
                force->fmap(i) = fseg;
            } else {
                Kokkos::atomic_add(&nodes[n1].f, fseg.f1);
                Kokkos::atomic_add(&nodes[n2].f, fseg.f2);
            }
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator()(TagMapForce, const int& i) const {
            auto nodes = net->get_nodes();
            auto conn = net->get_conn();
            
            Vec3 fn(0.0);
            for (int j = 0; j < conn[i].num; j++) {
                int k = conn[i].seg[j];
                SegForce fseg = force->fmap(k);
                fn += ((conn[i].order[j] == 1) ? fseg.f1 : fseg.f2);
            }
            nodes[i].f += fn;
        }
    };
    
    void pre_compute(System *system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_FORCE].start();
        system->devtimer[TIMER_FFTPRECOMP].start();
        
        compute_alpha(system);
        compute_stress_from_alpha(system);
        
        Kokkos::fence();
        system->timer[system->TIMER_FORCE].stop();
        system->devtimer[TIMER_FFTPRECOMP].stop();
    }

    void compute(System *system, bool zero=true)
    {
        Kokkos::fence();
        system->timer[system->TIMER_FORCE].start();
        
        DeviceDisNet *net = system->get_device_network();
        if (zero) zero_force(net);
        
        if (use_map) {
            resize_view(fmap, net->Nsegs_local);
        }
        
        using policy = Kokkos::RangePolicy<TagComputeForce,Kokkos::LaunchBounds<64,1>>;
        Kokkos::parallel_for(policy(0, net->Nsegs_local), AddSegmentForce<DeviceDisNet>(system, this, net));
        
        if (use_map) {
            Kokkos::fence();
            using policy = Kokkos::RangePolicy<TagMapForce,Kokkos::LaunchBounds<64,1>>;
            Kokkos::parallel_for(policy(0, net->Nnodes_local), AddSegmentForce<DeviceDisNet>(system, this, net));
        }
        
        Kokkos::fence();
        system->timer[system->TIMER_FORCE].stop();
    }
    
    Vec3 node_force(System *system, const int &i)
    {
        SerialDisNet *net = system->get_serial_network();
        synchronize_stress_gridval();
        
        auto conn = net->get_conn();
        
        Vec3 f(0.0);
        for (int j = 0; j < conn[i].num; j++) {
            int k = conn[i].seg[j];
            SegForce fs = segment_force(system, net, k);
            f += ((conn[i].order[j] == 1) ? fs.f1 : fs.f2);
        }
        
        return f;
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    Vec3 node_force(System* system, N* net, const int& i, const team_handle& team)
    {
        auto nodes = net->get_nodes();
        auto conn = net->get_conn();
            
        Vec3 f(0.0);
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, conn[i].num), [&] (const int& j, Vec3& fsum) {
            int k = conn[i].seg[j];
            SegForce fs = segment_force(system, net, k);
            fsum += ((conn[i].order[j] == 1) ? fs.f1 : fs.f2);
        }, f);
        team.team_barrier();
        
        return f;
    }
    
    ~ForceFFT() {
        plan.finalize();
    }
    
    const char* name() { return "ForceFFT"; }
};

namespace ForceType {
    typedef ForceLongShort<ForceFFT,ForceSegSegList<SegSegIsoFFT>> LONG_FFT_SHORT_ISO;
    typedef ForceCollection2<CORE_SELF_PKEXT,LONG_FFT_SHORT_ISO> DDD_FFT_MODEL;
}

} // namespace ExaDiS

#endif
