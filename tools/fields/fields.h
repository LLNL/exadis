/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_FIELDS_H
#define EXADIS_FIELDS_H

#include "system.h"

namespace ExaDiS { namespace tools {

/*---------------------------------------------------------------------------
 *
 *    Struct:       FieldParams
 *                  Parameters to compute field values
 *
 *-------------------------------------------------------------------------*/
template<class F>
struct FieldParams {
    typedef typename F::T_val T_val;
    F field;
    int Nimg[3];
    Vec3 d1, d2, d3;
};

/*---------------------------------------------------------------------------
 *
 *    Function:     field_seg_value
 *                  Computes the field contribution of segment i (and its
 *                  periodic replica) at field point p
 *
 *-------------------------------------------------------------------------*/
template<class F, class N>
KOKKOS_INLINE_FUNCTION
typename F::T_val field_seg_value(const F& f, N* net, int i, const Vec3& p)
{
    auto nodes = net->get_nodes();
    auto segs = net->get_segs();
    auto cell = net->cell;
    
    typedef typename F::T_val T_val;
    T_val val = T_val().zero();
    
    int n1 = segs[i].n1;
    int n2 = segs[i].n2;
    Vec3 b = segs[i].burg;
    Vec3 r1 = nodes[n1].pos;
    Vec3 r2 = cell.pbc_position(r1, nodes[n2].pos);
    
    for (int ii = -f.Nimg[0]; ii < f.Nimg[0]+1; ii++) {
        for (int jj = -f.Nimg[1]; jj < f.Nimg[1]+1; jj++) {
            for (int kk = -f.Nimg[2]; kk < f.Nimg[2]+1; kk++) {
                Vec3 pp = p + ii*f.d1 + jj*f.d2 + kk*f.d3;
                val += f.field.field_seg_value(r1, r2, b, pp);
            }
        }
    }
    
    return val;
}

/*---------------------------------------------------------------------------
 *
 *    Struct:       FieldGrid
 *                  Base class to compute a field on a regular grid arising
 *                  from the superposition of individual dislocation
 *                  contributions.
 *                  The struct must be instantied with a field kernel <F>
 *                  that implements function field_value().
 *
 *-------------------------------------------------------------------------*/
template <class F, class N>
struct FieldGrid {
    typedef typename F::T_val T_val;
    Kokkos::View<T_val***, Kokkos::LayoutRight, Kokkos::SharedSpace> gridval;
    int Ng[3];
    bool reg_conv;
    
    typedef typename F::Params Params;
    FieldParams<F> params;
    Cell cell;
    
    FieldGrid(N* net, Params p, std::vector<int> _Ng, std::vector<int> Nimg={1,1,1}, bool _reg_conv=true)
    {
        if (_Ng.size() != 3)
            ExaDiS_fatal("Error: number of grid points must be a list of 3 integers\n");
        if (Nimg.size() != 3)
            ExaDiS_fatal("Error: number of images must be a list of 3 integers\n");
        
        for (int i = 0; i < 3; i++)
            Ng[i] = _Ng[i];
            
        Kokkos::resize(gridval, Ng[0], Ng[1], Ng[2]);
        
        reg_conv = _reg_conv;
        
        params.Nimg[0] = (net->cell.xpbc == PBC_BOUND) ? Nimg[0] : 0;
        params.Nimg[1] = (net->cell.ypbc == PBC_BOUND) ? Nimg[1] : 0;
        params.Nimg[2] = (net->cell.zpbc == PBC_BOUND) ? Nimg[2] : 0;
        
        params.d1 = net->cell.H.colx();
        params.d2 = net->cell.H.coly();
        params.d3 = net->cell.H.colz();
        
        params.field = p;
        
        compute(net);
    }
    
    struct ComputeGrid {
        FieldGrid f;
        N* net;
        ComputeGrid(FieldGrid _f, N* _net) : f(_f), net(_net) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const int& kx, const int& ky, const int& kz) const {
            Vec3 s((kx+0.5)/f.Ng[0], (ky+0.5)/f.Ng[1], (kz+0.5)/f.Ng[2]);
            Vec3 p = net->cell.real_position(s);
            
            T_val val = T_val().zero();
            for (int i = 0; i < net->Nsegs_local; i++)
                val += field_seg_value(f.params, net, i, p);
            
            f.gridval(kx, ky, kz) = val;
        }
    };
    
    void compute(N* net)
    {
        // Make sure the accessors are properly set up (for SerialDisNet)
        cell = net->cell;
        net->update_ptr();
        using policy = Kokkos::MDRangePolicy<typename N::ExecutionSpace,Kokkos::Rank<3>>;
        Kokkos::parallel_for(policy({0, 0, 0}, {Ng[0], Ng[1], Ng[2]}), ComputeGrid(*this, net));
        Kokkos::fence();
        
        if (reg_conv)
            regularize_convergence(net);
    }
    
    struct RegularizeConvergence {
        FieldGrid f;
        N* net;
        Vec3 p0, px, py, pz;
        Kokkos::View<T_val*> V;
        T_val Vavg;
        
        RegularizeConvergence(FieldGrid _f, N* _net, Kokkos::View<T_val*>& _V, T_val _Vavg=T_val()) :
                              f(_f), net(_net), V(_V), Vavg(_Vavg) {
            p0 = net->cell.origin;
            px = p0 + f.params.d1;
            py = p0 + f.params.d2;
            pz = p0 + f.params.d3;
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const int& i) const {
            
            T_val V0 = field_seg_value(f.params, net, i, p0);
            Kokkos::atomic_add(&V(0), V0); // V0 value
            
            if (net->cell.xpbc == PBC_BOUND) {
                T_val Vx = field_seg_value(f.params, net, i, px);
                Kokkos::atomic_add(&V(1), Vx); // Vx value
            }
            if (net->cell.ypbc == PBC_BOUND) {
                T_val Vy = field_seg_value(f.params, net, i, py);
                Kokkos::atomic_add(&V(2), Vy); // Vy value
            }
            if (net->cell.zpbc == PBC_BOUND) {
                T_val Vz = field_seg_value(f.params, net, i, pz);
                Kokkos::atomic_add(&V(3), Vz); // Vz value
            }
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const int& kx, const int& ky, const int& kz, T_val& Vsum) const {
            T_val val = f.gridval(kx, ky, kz);
            Vsum += val;
            
            Vec3 s((kx+0.5)/f.Ng[0], (ky+0.5)/f.Ng[1], (kz+0.5)/f.Ng[2]);
            val -= (s.x*(V(1)-V(0)) + s.y*(V(2)-V(0)) + s.z*(V(3)-V(0)));
            f.gridval(kx, ky, kz) = val;
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const int& kx, const int& ky, const int& kz) const {
            T_val Vreg = 1.0/(f.Ng[0]*f.Ng[1]*f.Ng[2])*Vavg - 0.5*(V(1)-V(0) + V(2)-V(0) + V(3)-V(0));
            f.gridval(kx, ky, kz) -= Vreg;
        }
    };
    
    void regularize_convergence(N* net, bool average=true)
    {
        if (net->cell.xpbc == FREE_BOUND &&
            net->cell.ypbc == FREE_BOUND &&
            net->cell.zpbc == FREE_BOUND) return;
        
        // Conditional convergence regularization
        Kokkos::View<T_val*> V("Vfield", 4);
        Kokkos::deep_copy(V, 0.0);
        
        // Cell corner values
        Kokkos::parallel_for(net->Nsegs_local, RegularizeConvergence(*this, net, V));
        Kokkos::fence();
        
        // Linear term
        T_val Vavg;
        Kokkos::parallel_reduce(Kokkos::MDRangePolicy<typename N::ExecutionSpace,
            Kokkos::Rank<3>>({0, 0, 0}, {Ng[0], Ng[1], Ng[2]}),
            RegularizeConvergence(*this, net, V), Vavg
        );
        Kokkos::fence();
        
        // Average term
        if (average) {
            Kokkos::parallel_for(Kokkos::MDRangePolicy<typename N::ExecutionSpace,
                Kokkos::Rank<3>>({0, 0, 0}, {Ng[0], Ng[1], Ng[2]}),
                RegularizeConvergence(*this, net, V, Vavg)
            );
            Kokkos::fence();
        }
    }
    
    T_val interpolate(const Vec3& p)
    {
        Vec3 s = cell.scaled_position(p);
        
        double q[3];
        q[0] = s.x * Ng[0] - 0.5;
        q[1] = s.y * Ng[1] - 0.5;
        q[2] = s.z * Ng[2] - 0.5;
        
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
            ind1d[0][i] = (g[0]+i)%Ng[0];
            if (ind1d[0][i] < 0) ind1d[0][i] += Ng[0];
            ind1d[1][i] = (g[1]+i)%Ng[1];
            if (ind1d[1][i] < 0) ind1d[1][i] += Ng[1];
            ind1d[2][i] = (g[2]+i)%Ng[2];
            if (ind1d[2][i] < 0) ind1d[2][i] += Ng[2];
        }

        // 1d shape functions
        double phi1d[3][2];
        for (int i = 0; i < 3; i++) {
            phi1d[i][0] = 0.5*(1.0-xi[i]);
            phi1d[i][1] = 0.5*(1.0+xi[i]);
        }

        // 3d shape functions and indices
        T_val val = T_val().zero();
        for (int k = 0; k < 2; k++) {
            for (int j = 0; j < 2; j++) {
                for (int i = 0; i < 2; i++) {
                    double phi = phi1d[0][i]*phi1d[1][j]*phi1d[2][k];
                    val += phi * gridval(ind1d[0][i], ind1d[1][j], ind1d[2][k]);
                }
            }
        }
        
        return val;
    }
};

/*---------------------------------------------------------------------------
 *
 *    Struct:       FieldPoints
 *                  Base class to compute a field at a list of positions 
 *                  arising from the superposition of individual
 *                  dislocation contributions.
 *                  The struct must be instantied with a field kernel <F>
 *                  that implements function field_value().
 *
 *-------------------------------------------------------------------------*/
template <class F, class N>
struct FieldPoints {
    typedef typename F::T_val T_val;
    Kokkos::View<T_val*, Kokkos::SharedSpace> pointval;
    int Npoints;
    Kokkos::View<Vec3*, Kokkos::SharedSpace> points;
    
    typedef typename F::Params Params;
    FieldParams<F> params;
    
    FieldPoints(N* net, Params p, std::vector<Vec3>& _points, std::vector<int> Nimg={1,1,1})
    {
        Npoints = _points.size();
        Kokkos::resize(points, Npoints);
        for (int i = 0; i < Npoints; i++)
            points(i) = _points[i];
            
        if (Nimg.size() != 3)
            ExaDiS_fatal("Error: number of images must be a list of 3 integers\n");
        
        Kokkos::resize(pointval, Npoints);
        
        params.Nimg[0] = (net->cell.xpbc == PBC_BOUND) ? Nimg[0] : 0;
        params.Nimg[1] = (net->cell.ypbc == PBC_BOUND) ? Nimg[1] : 0;
        params.Nimg[2] = (net->cell.zpbc == PBC_BOUND) ? Nimg[2] : 0;
        
        params.d1 = net->cell.H.colx();
        params.d2 = net->cell.H.coly();
        params.d3 = net->cell.H.colz();
        
        params.field = p;
        
        compute(net);
    }
    
    struct ComputePoints {
        FieldPoints f;
        N* net;
        ComputePoints(FieldPoints _f, N* _net) : f(_f), net(_net) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const int& i) const {
            Vec3 p = f.points(i);
            
            T_val val = T_val().zero();
            for (int i = 0; i < net->Nsegs_local; i++)
                val += field_seg_value(f.params, net, i, p);
            
            f.pointval(i) = val;
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const team_handle& team) const {
            int tid = team.team_rank(); // returns a number between 0 and TEAM_SIZE
            int lid = team.league_rank(); // returns a number between 0 and N
            
            int i = lid; // point id
            Vec3 p;
            if (tid == 0) p = f.points(i);
            team.team_broadcast(p, 0);
            
            T_val val;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, net->Nsegs_local), [&](const int& j, T_val& vsum) {
                vsum += field_seg_value(f.params, net, j, p);
            }, val);
            
            Kokkos::single(Kokkos::PerTeam(team), [=]() {
                f.pointval(i) = val;
            });
        }
    };
    
    void compute(N* net)
    {
        // Make sure the accessors are properly set up (for SerialDisNet)
        net->update_ptr();
        
        if constexpr (std::is_same<typename N::ExecutionSpace, Kokkos::Serial>::value) {
            using policy = Kokkos::RangePolicy<typename N::ExecutionSpace>;
            Kokkos::parallel_for(policy(0, Npoints), ComputePoints(*this, net));
        } else {
            using team_policy = Kokkos::TeamPolicy<typename N::ExecutionSpace>;
            Kokkos::parallel_for(team_policy(Npoints, Kokkos::AUTO), ComputePoints(*this, net));
        }
        Kokkos::fence();
    }
};

/*---------------------------------------------------------------------------
 *
 *    Function:     field_point
 *                  Base function to compute a field at a given position
 *                  arising from the superposition of individual
 *                  dislocation contributions.
 *                  The function must be instantied with a field kernel <F>
 *                  that implements function field_value().
 *
 *-------------------------------------------------------------------------*/
template <class F, class N>
typename F::T_val field_point(N* net, typename F::Params params, const Vec3& p, std::vector<int> Nimg={1,1,1})
{
    if (Nimg.size() != 3)
        ExaDiS_fatal("Error: number of images must be a list of 3 integers\n");
    
    FieldParams<F> fparams;
    
    fparams.Nimg[0] = (net->cell.xpbc == PBC_BOUND) ? Nimg[0] : 0;
    fparams.Nimg[1] = (net->cell.ypbc == PBC_BOUND) ? Nimg[1] : 0;
    fparams.Nimg[2] = (net->cell.zpbc == PBC_BOUND) ? Nimg[2] : 0;
    
    fparams.d1 = net->cell.H.colx();
    fparams.d2 = net->cell.H.coly();
    fparams.d3 = net->cell.H.colz();
    
    fparams.field = params;
    
    // Make sure the accessors are properly set up (for SerialDisNet)
    net->update_ptr();
    
    typedef typename F::T_val T_val;
    Kokkos::View<T_val*, Kokkos::SharedSpace> val("val", 1);
    
    TeamSize ts = get_team_sizes(net->Nsegs_local);
    Kokkos::parallel_for(Kokkos::TeamPolicy<>(ts.num_teams, ts.team_size), KOKKOS_LAMBDA(const team_handle& team) {
        int ts = team.team_size();
        int lid = team.league_rank();
        
        T_val vteam;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, ts), [=](int& t, T_val& vsum) {
            int j = lid*ts + t; // segment id
            if (j < net->Nsegs_local)
                vsum += field_seg_value(fparams, net, j, p);
        }, vteam);
        
        Kokkos::single(Kokkos::PerTeam(team), [=]() {
            Kokkos::atomic_add(&val(0), vteam);
        });
    });
    Kokkos::fence();
    
    return val(0);
}

} } // namespace ExaDiS::tools

#endif
