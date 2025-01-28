#include <iostream>
#include "system.h"
#include "force.h"
#include "functions.h"

#define STR(str) #str
#define STRING(str) STR(str)
std::string exadis_root_dir = STRING(EXADIS_ROOT_DIR);

using namespace ExaDiS;


class TestFFT {
public:
    FFTPlan<> plan;
    int Ngrid, Ngrid3;
    double rcgrid, rcnei;
    
    typedef Kokkos::View<complex***, Kokkos::LayoutRight, Kokkos::SharedSpace> T_grid;
    //typedef Kokkos::View<complex***, Kokkos::LayoutRight> T_grid;
    
    T_grid gridval[9];
    T_grid w;
    double wsum;
    
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
    
    DeviceDisNet *d_net;
    Vec3 Lbox, H, bmin;
    double V;
    
    double t0,t1,t2,t3,t4;

    TestFFT(DeviceDisNet* net, int _Ngrid) {
        Ngrid = _Ngrid;
        if (Ngrid <= 0)
            ExaDiS_fatal("Error: undefined grid size in ForceFFT\n");
        
        Ngrid3 = Ngrid*Ngrid*Ngrid;
        plan.initialize(Ngrid, Ngrid, Ngrid);
        
        for (int i = 0; i < 9; i++)
            Kokkos::resize(gridval[i], Ngrid, Ngrid, Ngrid);
        
        double MU = 50e9 * stress_fact;
        double NU = 0.3;
        double LA = 2.0*MU*NU/(1.0-2.0*NU);

        Mat33 delta = Mat33().eye();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    for (int l = 0; l < 3; l++)
                        C[i][j][k][l] = LA*(delta[i][j]*delta[k][l]) +
                        MU*(delta[i][k]*delta[j][l]+delta[i][l]*delta[j][k]);
            
        initialize_spectral_core(net);
    }
    
    struct TagComputeW {};
    struct TagNormalizeW {};
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagComputeW, const int &ind, double &psum) const {
        int kx = ind / (Ngrid * Ngrid);
        int ky = (ind - kx * Ngrid * Ngrid) / Ngrid;
        int kz = ind - kx * Ngrid * Ngrid - ky * Ngrid;
        
        int kmax = Ngrid/2 + Ngrid % 2;
        int kxx = kx; if (kxx >= kmax) kxx -= Ngrid;
        int kyy = ky; if (kyy >= kmax) kyy -= Ngrid;
        int kzz = kz; if (kzz >= kmax) kzz -= Ngrid;
        
        double x = kxx*Lbox.x/Ngrid;
        double y = kyy*Lbox.y/Ngrid;
        double z = kzz*Lbox.z/Ngrid;
        double r2 = x*x + y*y + z*z;
        double aw2 = rcgrid*rcgrid;
        double wr = 15.0*aw2*aw2/8.0/M_PI/pow(r2+aw2, 3.5);
        w(kx, ky, kz) = complex(wr, 0.0);
        
        psum += wr;
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagNormalizeW, const int &ind) const {
        int kx = ind / (Ngrid * Ngrid);
        int ky = (ind - kx * Ngrid * Ngrid) / Ngrid;
        int kz = ind - kx * Ngrid * Ngrid - ky * Ngrid;
        w(kx, ky, kz) /= wsum;
    }
    
    void initialize_spectral_core(DeviceDisNet* net)
    {
        // Non-singular kernel
        Lbox.x = net->cell.H.xx();
        Lbox.y = net->cell.H.yy();
        Lbox.z = net->cell.H.zz();
        
        H = 1.0/Ngrid * Lbox;
        double Hmax = fmax(fmax(H.x, H.y), H.z);
        
        // Grid core radius
        rcgrid = 2.0*Hmax;
        rcnei = 4.0*rcgrid;
        
        Kokkos::resize(w, Ngrid, Ngrid, Ngrid);
        Kokkos::deep_copy(w, 0.0);
        wsum = 0.0;
        
        Kokkos::parallel_reduce("ForceFFT::ComputeW",
            Kokkos::RangePolicy<TagComputeW>(0, Ngrid3), *this, wsum
        );
        Kokkos::fence();
        
        Kokkos::parallel_for("ForceFFT::NormalizeW",
            Kokkos::RangePolicy<TagNormalizeW>(0, Ngrid3), *this
        );
        Kokkos::fence();
        
        FFT3DTransform(plan, w, w, FFT_FORWARD);
        Kokkos::fence();
    }
    
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
    void operator() (TagComputeAlpha, const int &i) const {
        auto nodes = d_net->get_nodes();
        auto segs = d_net->get_segs();
        auto cell = d_net->cell;
        
        int n1 = segs[i].n1;
        int n2 = segs[i].n2;
        Vec3 b = segs[i].burg;
        Vec3 r1 = nodes[n1].pos;
        Vec3 r2 = cell.pbc_position(r1, nodes[n2].pos);
        
        Vec3 t = r2-r1;
        double L = t.norm();
        if (L > 1e-10) {
            
            t = t.normalized();
            
            double xmin = fmin(r1.x, r2.x) - 0.5*H.x - bmin.x;
            double xmax = fmax(r1.x, r2.x) - 0.5*H.x - bmin.x;
            double ymin = fmin(r1.y, r2.y) - 0.5*H.y - bmin.y;
            double ymax = fmax(r1.y, r2.y) - 0.5*H.y - bmin.y;
            double zmin = fmin(r1.z, r2.z) - 0.5*H.z - bmin.z;
            double zmax = fmax(r1.z, r2.z) - 0.5*H.z - bmin.z;

            int imin = floor(xmin/H.x);
            int imax = floor(xmax/H.x) + 1;
            int jmin = floor(ymin/H.y);
            int jmax = floor(ymax/H.y) + 1;
            int kmin = floor(zmin/H.z);
            int kmax = floor(zmax/H.z) + 1;
            
            double Wtot = 0.0;
            
            for (int ib = imin; ib <= imax; ib++) {
                for (int jb = jmin; jb <= jmax; jb++) {
                    for (int kb = kmin; kb <= kmax; kb++) {
                        
                        Vec3 bc;
                        bc.x = (ib+0.5)*H.x + bmin.x;
                        bc.y = (jb+0.5)*H.y + bmin.y;
                        bc.z = (kb+0.5)*H.z + bmin.z;

                        double W = alpha_box_segment(r1, t, L, bc, H);
                        Wtot += W;
                        
                        // Box index
                        int kx = ib % Ngrid; if (kx < 0) kx += Ngrid;
                        int ky = jb % Ngrid; if (ky < 0) ky += Ngrid;
                        int kz = kb % Ngrid; if (kz < 0) kz += Ngrid;
                        
                        Kokkos::atomic_add(&gridval[0](kx, ky, kz), W/V*b.x*t.x);
                        Kokkos::atomic_add(&gridval[1](kx, ky, kz), W/V*b.x*t.y);
                        Kokkos::atomic_add(&gridval[2](kx, ky, kz), W/V*b.x*t.z);
                        Kokkos::atomic_add(&gridval[3](kx, ky, kz), W/V*b.y*t.x);
                        Kokkos::atomic_add(&gridval[4](kx, ky, kz), W/V*b.y*t.y);
                        Kokkos::atomic_add(&gridval[5](kx, ky, kz), W/V*b.y*t.z);
                        Kokkos::atomic_add(&gridval[6](kx, ky, kz), W/V*b.z*t.x);
                        Kokkos::atomic_add(&gridval[7](kx, ky, kz), W/V*b.z*t.y);
                        Kokkos::atomic_add(&gridval[8](kx, ky, kz), W/V*b.z*t.z);
                    }
                }
            }
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagStressFromAlpha, const int &ind) const {
        // Voxel id and coordinates
        int kx = ind / (Ngrid * Ngrid);
        int ky = (ind - kx * Ngrid * Ngrid) / Ngrid;
        int kz = ind - kx * Ngrid * Ngrid - ky * Ngrid;
        
        int kmax = Ngrid/2 + Ngrid % 2;
        int kxx = kx; if (kxx >= kmax) kxx -= Ngrid;
        int kyy = ky; if (kyy >= kmax) kyy -= Ngrid;
        int kzz = kz; if (kzz >= kmax) kzz -= Ngrid;
        
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
        xk[0] = 2.0*M_PI*kxx/Ngrid;
        xk[1] = 2.0*M_PI*kyy/Ngrid;
        xk[2] = 2.0*M_PI*kzz/Ngrid;
        
        complex cxk[3];
        // iGreenOp=0: k=i*q
        if (GREEN_OP == 0) {
            cxk[0] = 1.0/H[0]*complex(0.0, xk[0]);
            cxk[1] = 1.0/H[1]*complex(0.0, xk[1]);
            cxk[2] = 1.0/H[2]*complex(0.0, xk[2]);
        // iGreenOp=1: k=i*sin(q) (C)
        } else if (GREEN_OP == 1) {
            cxk[0] = 1.0/H[0]*complex(0.0, sin(xk[0]));
            cxk[1] = 1.0/H[1]*complex(0.0, sin(xk[1]));
            cxk[2] = 1.0/H[2]*complex(0.0, sin(xk[2]));
        // iGreenOp=2: k=e^(i*q)-1 (W)
        } else if (GREEN_OP == 2) {
            cxk[0] = 1.0/H[0]*complex(cos(xk[0])-1.0, sin(xk[0]));
            cxk[1] = 1.0/H[1]*complex(cos(xk[1])-1.0, sin(xk[1]));
            cxk[2] = 1.0/H[2]*complex(cos(xk[2])-1.0, sin(xk[2]));
        // iGreenOp=3: k(R), rotated scheme
        } else if (GREEN_OP == 3) {
            if (kxx == -Ngrid/2 || kyy == -Ngrid/2 || kzz == -Ngrid/2) {
                zero = true;
            }
            complex fact0(1.0+cos(xk[0]), sin(xk[0]));
            complex fact1(1.0+cos(xk[1]), sin(xk[1]));
            complex fact2(1.0+cos(xk[2]), sin(xk[2]));
            complex fact = fact0*fact1*fact2;
            cxk[0] = 0.25/H[0]*fact*complex(0.0, tan(0.5*xk[0]));
            cxk[1] = 0.25/H[1]*fact*complex(0.0, tan(0.5*xk[1]));
            cxk[2] = 0.25/H[2]*fact*complex(0.0, tan(0.5*xk[2]));
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
    void operator() (TagNormalizeStress, const int &ind) const {
        int kx = ind / (Ngrid * Ngrid);
        int ky = (ind - kx * Ngrid * Ngrid) / Ngrid;
        int kz = ind - kx * Ngrid * Ngrid - ky * Ngrid;
        
        for (int i = 0; i < 6; i++)
            gridval[stress_comps[i]](kx, ky, kz) /= (stress_fact*Ngrid3);
    }
    
    void compute_alpha(DeviceDisNet* net)
    {
        d_net = net;
        
        // Only for orthorombic boxes as for now
        if (d_net->cell.is_triclinic())
            ExaDiS_fatal("Error: ForceFFT only implemented for orthorombic cells\n");
        
        Lbox.x = d_net->cell.H.xx();
        Lbox.y = d_net->cell.H.yy();
        Lbox.z = d_net->cell.H.zz();
        H = 1.0/Ngrid * Lbox;
        V = H.x * H.y * H.z;
        bmin = d_net->cell.origin;
        
        
        
        for (int i = 0; i < 9; i++)
            Kokkos::deep_copy(gridval[i], 0.0);
            
        Kokkos::Timer timer;
        Kokkos::fence(); timer.reset();
        Kokkos::parallel_for("ForceFFT::ComputeAlpha",
            Kokkos::RangePolicy<TagComputeAlpha>(0, d_net->Nsegs_local), *this
        );
        Kokkos::fence();
        t0 = timer.seconds();
    }
    
    void compute_stress()
    {
        Kokkos::Timer timer;
        
        Kokkos::fence(); timer.reset();
        for (int i = 0; i < 9; i++)
            FFT3DTransform(plan, gridval[i], gridval[i], FFT_FORWARD);
        Kokkos::fence();
        t1 = timer.seconds();
        
        Kokkos::fence(); timer.reset();
        Kokkos::parallel_for("ForceFFT::StressFromAlpha",
            Kokkos::RangePolicy<TagStressFromAlpha>(0, Ngrid3), *this
        );
        Kokkos::fence();
        t2 = timer.seconds();
        
        Kokkos::fence(); timer.reset();
        for (int i = 0; i < 6; i++)
            FFT3DTransform(plan, gridval[stress_comps[i]], gridval[stress_comps[i]], FFT_BACKWARD);
        Kokkos::fence();
        t3 = timer.seconds();
        
        Kokkos::fence(); timer.reset();
        Kokkos::parallel_for("ForceFFT::NormalizeStress",
            Kokkos::RangePolicy<TagNormalizeStress>(0, Ngrid3), *this
        );
        Kokkos::fence();
        t4 = timer.seconds();
        
    }
    
    void print() {
        double ttot = t0+t1+t2+t3+t4;
        printf("%s time: %f sec (%.2f %%)\n", "alpha    ", t0, t0/ttot*100.0);
        printf("%s time: %f sec (%.2f %%)\n", "transform", t1, t1/ttot*100.0);
        printf("%s time: %f sec (%.2f %%)\n", "stress   ", t2, t2/ttot*100.0);
        printf("%s time: %f sec (%.2f %%)\n", "transform", t3, t3/ttot*100.0);
        printf("%s time: %f sec (%.2f %%)\n", "normalize", t4, t4/ttot*100.0);
        printf("%s time: %f sec (%.2f %%)\n", "total    ", ttot, 100.0);
    }
    
    ~TestFFT() {
        plan.finalize();
    }
};

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard guard(argc, argv);
    Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout);

    int Ngrid = 64;
    if (argc == 2) Ngrid = atoi(argv[1]);
    printf("test_fft: Ngrid = %d\n", Ngrid);
    
    double Lbox = 300.0;
    double maxseg = 15.0;
    SerialDisNet* config = generate_prismatic_config(Crystal(BCC_CRYSTAL), Lbox, 200, 0.2*Lbox, maxseg);
    printf("Nnodes = %d\n", config->number_of_nodes());
    
    Kokkos::Timer timer;

    for (int i = 0; i < 2; i++) {
    
        Kokkos::fence(); timer.reset();
        config->generate_connectivity();
        DisNetManager* net_mngr = exadis_new<DisNetManager>(config);
        DeviceDisNet* net = net_mngr->get_device_network();
        double t0 = timer.seconds();
        
        Kokkos::fence(); timer.reset();
        TestFFT fft(net, Ngrid);
        double t1 = timer.seconds();
        
        printf("--------------\n");
        printf("%s time: %f sec\n", "net_mngr ", t0);
        printf("%s time: %f sec\n", "init fft ", t1);
        printf("--------------\n");
        
        fft.compute_alpha(net);
        fft.compute_stress();
        fft.print();
        
        net_mngr->set_network((SerialDisNet*)nullptr);
        exadis_delete(net_mngr);

    }

    return 0;
}
