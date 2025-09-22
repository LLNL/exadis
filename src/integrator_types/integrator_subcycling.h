/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_INTEGRATOR_SUBCYCLING_H
#define EXADIS_INTEGRATOR_SUBCYCLING_H

#include "integrator.h"
#include "force.h"

namespace ExaDiS {

#define MAXGROUPS 5

/*---------------------------------------------------------------------------
 *
 *    Class:        ForceSubcycling
 *                  Force class to be used with the IntegratorSubcycling.
 *                  It contains member properties specific to the subcycling
 *                  scheme.
 *                  
 *                  NOTE: when option drift=1, this is a modified version
 *                  of the original subcycling scheme in which nodal forces 
 *                  are stored for groups > 0 and summed up when group 0 is
 *                  selected. This allows to integrate nodes under the total
 *                  force during each subcycle, and makes it compatible with
 *                  arbitrary (non-linear) mobility laws. The original
 *                  subcycling scheme is used when drift=0 (default).
 *                  
 *                  Eventually this class could be templated to be
 *                  more generic and offer more flexibility on the type of
 *                  force contributions that can be used with subcycling.
 *
 *-------------------------------------------------------------------------*/
class ForceSubcycling : public Force {
public:
    typedef ForceType::CORE_SELF_PKEXT FSeg;
    typedef ForceFFT FLong;
    typedef ForceSegSegList<SegSegIsoFFT> FSegSeg;
    
    FSeg* fseg;
    FLong* flong;
    FSegSeg* fsegseg;
    int Ngroups;
    int group;
    bool drift;
    bool flong_group0;
    
    static const int Ngmax = MAXGROUPS;
    Kokkos::View<Vec3*> fgroup[Ngmax-1];
    
public:
    struct Params {
        FSeg::Params FSegParams;
        FLong::Params FLongParams;
        bool drift = false;
        bool flong_group0 = true;
        Params(int Ngrid) { FLongParams = FLong::Params(Ngrid); }
        Params(int Nx, int Ny, int Nz) { FLongParams = FLong::Params(Nx, Ny, Nz); }
        Params(int Ngrid, bool _drift, bool _flong_group0) {
            FLongParams = FLong::Params(Ngrid); drift = _drift; flong_group0 = _flong_group0;
        }
        Params(int Nx, int Ny, int Nz, bool _drift, bool _flong_group0) {
            FLongParams = FLong::Params(Nx, Ny, Nz); drift = _drift; flong_group0 = _flong_group0;
        }
        Params(FSeg::Params _FSegParams, FLong::Params _FLongParams) :
        FSegParams(_FSegParams), FLongParams(_FLongParams) {}
    };
    
    ForceSubcycling(System* system, Params params) {
        // Segment force (core, pk, etc...)
        fseg = exadis_new<FSeg>(system, params.FSegParams);
        // Long-range
        flong = exadis_new<FLong>(system, params.FLongParams);
        // Short-range segseg forces
        fsegseg = exadis_new<FSegSeg>(system, flong);
        // Number of groups
        Ngroups = 0;
        // Assign no group
        group = -1;
        // Options
        drift = params.drift;
        flong_group0 = params.flong_group0;
    }
    
    void init_subforce(System* system, int _Ngroups) {
        if (_Ngroups != Ngroups)
            ExaDiS_fatal("Error: inconsistent number of groups in ForceSubcycling\n");
        for (int i = 0; i < Ngroups-1; i++)
            Kokkos::resize(fgroup[i], system->Nnodes_local());
    }
    
    void save_subforce(DeviceDisNet* net, int group) {
        Kokkos::View<Vec3*>& f = fgroup[group-1];
        Kokkos::parallel_for(net->Nnodes_local, KOKKOS_LAMBDA(const int& i) {
            auto nodes = net->get_nodes();
            f(i) = nodes[i].f;
        });
        Kokkos::fence();
    }
    
    struct AddSubForces {
        DeviceDisNet* net;
        ForceSubcycling* force;
        
        AddSubForces(DeviceDisNet* _net, ForceSubcycling* _force) : 
        net(_net), force(_force) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const int& i) const {
            auto nodes = net->get_nodes();
            Vec3 f(0.0);
            for (int j = 0; j < force->Ngroups-1; j++)
                f += force->fgroup[j](i);
            nodes[i].f += f;
        }
    };
    
    void pre_compute(System* system) {
        fseg->pre_compute(system);
        flong->pre_compute(system);
        // Skip fsegseg pre-compute. We'll build all
        // subcycling groups during the integration.
    }
    
    void compute(System* system, bool zero=true) {
        if (Ngroups <= 0)
            ExaDiS_fatal("Error: undefined number of groups in ForceSubcycling\n");
        
        DeviceDisNet *net = system->get_device_network();
        if (zero) zero_force(net);
        
        if (group == 0) {
            // This is the group containing the segment forces
            fseg->compute(system, false);
            if (flong_group0)
                flong->compute(system, false);
            fsegseg->compute(system, false);
            // In the drift scheme we integrate under all forces
            // so add all other group forces
            if (drift) {
                Kokkos::parallel_for(net->Nnodes_local, AddSubForces(net, this));
                Kokkos::fence();
            }
        } else if (group > 0) {
            // These are the groups containing the seg/seg forces
            if (!flong_group0 && group == Ngroups-1)
                flong->compute(system, false);
            fsegseg->compute(system, false);
        }
        // Do not compute forces if group = -1
    }
    
    Vec3 node_force(System *system, const int &i) {
        Vec3 f(0.0);
        f += fseg->node_force(system, i);
        f += flong->node_force(system, i);
        f += fsegseg->node_force(system, i);
        return f;
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    Vec3 node_force(System* system, N* net, const int& i, const team_handle& team)
    {
        Vec3 f(0.0);
        f += fseg->node_force(system, net, i, team);
        f += flong->node_force(system, net, i, team);
        f += fsegseg->node_force(system, net, i, team);
        return f;
    }
    
    ~ForceSubcycling() {
        exadis_delete(fseg);
        exadis_delete(flong);
        exadis_delete(fsegseg);
    }
    
    const char* name() { return "ForceSubcycling"; }
};

namespace ForceType {
    typedef ForceSubcycling SUBCYCLING_MODEL;
}

/*---------------------------------------------------------------------------
 *
 *    Struct:       SegSegGroups
 *                  Data structure to hold the lists of segment pairs
 *                  associated with each subgroup.
 *
 *-------------------------------------------------------------------------*/
struct SegSegGroups
{
    static const int Ngmax = MAXGROUPS;
    int Ngroups;
    double r2groups[Ngmax-1];
    int Nsegseg[Ngmax];
    Kokkos::View<int*> gcount;
    Kokkos::DualView<SegSeg*> segseglist[Ngmax];
    int Nsegseg_tot;
    double gfrac[Ngmax];
    
    SegSegList* ssl;
    Kokkos::View<double*> gdist2;
    double rg9s;
    Kokkos::View<int*> countmove;
    Kokkos::View<int*> nflag;
    
    Kokkos::DualView<SegSegForce*> fsegseglist[Ngmax];
    SegSegList::NodeComputeMap<DeviceDisNet> compute_map[Ngmax];
    
    SegSegGroups(System* system, std::vector<double> rgroups, double cutoff) {
        if (rgroups.size() == 0) {
            // No group radii were provided, let's try to select some appropriate values
            Ngroups = MAXGROUPS;
            double rmax = fmin(system->params.maxseg, cutoff);
            double rmin = fmin(fmax(0.3*system->params.minseg, 3.0*system->params.rann), rmax);
            rgroups.resize(Ngroups-1);
            rgroups[0] = 0.0;
            for (int i = 1; i < Ngroups-1; i++) {
                double f = (1.0*(i-1)/(Ngroups-2)-0.5)*M_PI;
                f = 0.5*(sin(f)+1.0);
                rgroups[i] = rmin + f*(rmax-rmin);
            }
        } else {
            Ngroups = (int)rgroups.size()+1;
        }
        
        if (Ngroups < 2)
            ExaDiS_fatal("Error: there must be at least 2 groups in SegSegGroups\n");
        if (Ngroups > MAXGROUPS)
            ExaDiS_fatal("Error: there must be at most %d groups in SegSegGroups\n", MAXGROUPS);
        
        for (int i = 0; i < Ngroups-1; i++) {
            r2groups[i] = rgroups[i] * rgroups[i];
            if (i > 0) {
                if (rgroups[i-1] > rgroups[i]) {
                    for (int j = 0; j < rgroups.size(); j++)
                        printf("Group %d: r = %e\n", j, rgroups[j]);
                    ExaDiS_fatal("Error: subcycling group radii must be of increasing size\n");
                }
            }
        }
        Kokkos::resize(gcount, Ngroups);
        
        rg9s = fmax(3.0*rgroups[Ngroups-2], 0.5*(rgroups[Ngroups-2]+cutoff));
        rg9s = rg9s * rg9s;
        Kokkos::resize(countmove, 1);
    }
    
    void init_groups(SegSegList* _ssl) {
        Nsegseg_tot = 0;
        auto h_gcount = Kokkos::create_mirror_view(gcount);
        Kokkos::deep_copy(h_gcount, gcount);
        ssl = _ssl;
        for (int i = 0; i < Ngroups; i++) {
            Nsegseg[i] = h_gcount(i);
            resize_view(segseglist[i], Nsegseg[i]);
            if (ssl->use_compute_map)
                resize_view(fsegseglist[i], Nsegseg[i]);
            Nsegseg_tot += Nsegseg[i];
        }
        resize_view(ssl->segsegflag, Nsegseg[Ngroups-1]);
        Kokkos::deep_copy(ssl->segsegflag, 1);
        resize_view(gdist2, Nsegseg[Ngroups-1]);
        Kokkos::deep_copy(countmove, 0);
        for (int i = 0; i < Ngroups; i++)
            gfrac[i] = (Nsegseg_tot > 0) ? 1.0*Nsegseg[i]/Nsegseg_tot : 0;
    }
    
    KOKKOS_INLINE_FUNCTION
    int find_group(double dist2) {
        for (int i = 0; i < Ngroups-1; i++)
            if (dist2 < r2groups[i]) return i;
        return Ngroups-1;
    }
    
    struct MoveInteractions {
        DeviceDisNet* net;
        SegSegGroups* ssg;
        double rgs;
        
        MoveInteractions(DeviceDisNet* _net, SegSegGroups* _ssg, double _rgs) :
        net(_net), ssg(_ssg), rgs(_rgs) {}
        
        MoveInteractions(SegSegGroups* _ssg) : ssg(_ssg) {}
        
        struct TagFlag {};
        struct TagMove {};
        
        KOKKOS_INLINE_FUNCTION
        void operator() (TagFlag, const int& i) const {
            auto segs = net->get_segs();
            SegSeg s = ssg->segseglist[ssg->Ngroups-1].d_view(i);
            int n1 = segs[s.s1].n1;
            int n2 = segs[s.s1].n2;
            int n3 = segs[s.s2].n1;
            int n4 = segs[s.s2].n2;
            // Flag the interaction to be moved to the lower group
            // if any of its nodes is flagged 2
            if ((ssg->nflag(n1)-2)*(ssg->nflag(n2)-2)*(ssg->nflag(n3)-2)*(ssg->nflag(n4)-2) == 0) {
                if (ssg->gdist2(i) <= rgs && ssg->ssl->segsegflag(i) == 1) {
                    ssg->ssl->segsegflag(i) = 0;
                    Kokkos::atomic_inc(&ssg->countmove(0));
                }
            }
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (TagMove, const int& i) const {
            if (!ssg->ssl->segsegflag(i)) {
                int idx = Kokkos::atomic_fetch_add(&ssg->gcount(ssg->Ngroups-2), 1);
                ssg->segseglist[ssg->Ngroups-2].d_view(idx) = ssg->segseglist[ssg->Ngroups-1].d_view(i);
            }
        }
    };
    
    void move_interactions(DeviceDisNet* net, int iTry) {
        double rgs = rg9s * (iTry+1) * (iTry+1);
        using policy = Kokkos::RangePolicy<typename MoveInteractions::TagFlag>;
        Kokkos::parallel_for(policy(0, Nsegseg[Ngroups-1]), MoveInteractions(net, this, rgs));
        Kokkos::fence();
        //printf("move_interactions iTry = %d: countmove = %d\n",iTry,countmove(0));
    }
    
    void update_interactions(DeviceDisNet* net) {
        auto h_countmove = Kokkos::create_mirror_view(countmove);
        Kokkos::deep_copy(h_countmove, countmove);
        if (h_countmove(0) > 0) {
            Nsegseg[Ngroups-2] += h_countmove(0);
            resize_view(segseglist[Ngroups-2], Nsegseg[Ngroups-2]);
            using policy = Kokkos::RangePolicy<typename MoveInteractions::TagMove>;
            Kokkos::parallel_for(policy(0, Nsegseg[Ngroups-1]), MoveInteractions(this));
            Kokkos::fence();
            if (ssl->use_compute_map) {
                resize_view(fsegseglist[Ngroups-2], Nsegseg[Ngroups-2]);
                SegSegList::build_compute_map<DeviceDisNet>(ssl, &compute_map[Ngroups-2], net);
            }
        }
    }
};

/*---------------------------------------------------------------------------
 *
 *    Struct:       BuildSegSegGroups
 *                  Struct to assign segment pairs to individual subgroups
 *
 *-------------------------------------------------------------------------*/
template<class N, class G>
struct BuildSegSegGroups {
    N* net;
    G* groups;
    NeighborList* neilist;
    double cutoff2;
    bool hinge;
    bool count_only;
    
    BuildSegSegGroups(N* _net, G* _groups, NeighborList* _neilist, 
                      double _cutoff, bool _hinge, bool _count_only) : 
    net(_net), groups(_groups), neilist(_neilist), hinge(_hinge), count_only(_count_only) {
        cutoff2 = _cutoff * _cutoff;
        Kokkos::deep_copy(groups->gcount, 0); 
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator()(const int& i) const {
        if (cutoff2 <= 0.0) return;
        auto count = neilist->get_count();
        auto nei = neilist->get_nei();
        
        int Nnei = count[i];
        for (int l = 0; l < Nnei; l++) {
            int j = nei(i,l); // neighbor seg
            if (i < j) { // avoid double-counting
                // Compute distance
                double dist2 = get_min_dist2_segseg(net, i, j, hinge);
                if (dist2 >= 0.0 && dist2 < cutoff2) {
                    int groupid = groups->find_group(dist2);
                    if (count_only) {
                        Kokkos::atomic_inc(&groups->gcount(groupid));
                    } else {
                        int idx = Kokkos::atomic_fetch_add(&groups->gcount(groupid), 1);
                        groups->segseglist[groupid].d_view(idx) = SegSeg(i, j);
                        if (groupid == groups->Ngroups-1) groups->gdist2(idx) = dist2;
                    }
                }
            }
        }
    }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        IntegratorSubcyclingBase
 *                  Base class of a subcycling-enabled base integrator
 *
 *-------------------------------------------------------------------------*/
class IntegratorSubcyclingBase {
protected:
    bool subcycling;
    static const int Ngmax = MAXGROUPS;
    SegSegGroups* subgroups;

public:
    double nextdtsub[Ngmax], realdtsub[Ngmax];
    int group = -1;
    
    IntegratorSubcyclingBase(System* system) : subcycling(false), group(0) {}
    
    IntegratorSubcyclingBase(System* system, SegSegGroups* _subgroups) :
    subcycling(true) {
        subgroups = _subgroups;
        for (int i = 0; i < subgroups->Ngroups; i++)
            nextdtsub[i] = system->params.nextdt;
    }
    
    virtual void init_subcycling_step(System* system) {}
    virtual void finish_subcycling_step(System* system) {}
    
    template<class I>
    void integrate(System* system, I* integrator) {
        double realdt = system->realdt;
        integrator->nextdt = nextdtsub[group];
        if (group < subgroups->Ngroups-1)
            integrator->nextdt = fmin(realdt, integrator->nextdt);
        
        integrator->I::Base::integrate(system);
        
        realdtsub[group] = system->realdt;
        nextdtsub[group] = integrator->nextdt;
        if (group < subgroups->Ngroups-1) system->realdt = realdt;
    }
    
    void flag_oscillating_nodes(DeviceDisNet* net) {
        if (group != subgroups->Ngroups-1) {
            Kokkos::View<int*>& nflag = subgroups->nflag;
            Kokkos::parallel_for(net->Nnodes_local, KOKKOS_LAMBDA(const int& i) {
                auto nodes = net->get_nodes();
                if (nflag(i) <= 0) nodes[i].v = Vec3(0.0);
            });
            Kokkos::fence();
        }
    }
    
    void forward_progress_check(DeviceDisNet* net, T_v& vold) {
        Kokkos::View<int*>& nflag = subgroups->nflag;
        Kokkos::parallel_for(net->Nnodes_local, KOKKOS_LAMBDA(const int& i) {
            auto nodes = net->get_nodes();
            if (nflag(i) <= 0) return;
            if (dot(vold(i), nodes[i].v) < 0.0) nflag(i) = 0;
        });
        Kokkos::fence();
    }
    
    void write_restart_sub(FILE* fp) { 
        for (int i = 0; i < subgroups->Ngroups; i++)
            fprintf(fp, "nextdtsub%d %.17g\n", i, nextdtsub[i]);
    }
    void read_restart_sub(FILE* fp) {
        int j;
        for (int i = 0; i < subgroups->Ngroups; i++)
            fscanf(fp, "nextdtsub%d %lf\n", &j, &nextdtsub[i]);
    }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        IntegratorRKFSubcycling
 *                  Modified IntegratorRKF class to handle the bookkeeping
 *                  required during integration of subgroups.
 *
 *-------------------------------------------------------------------------*/
class IntegratorRKFSubcycling : public IntegratorRKF,
                                public IntegratorSubcyclingBase {
private:
    int nTry = 3;
    
public:
    typedef IntegratorRKF Base;
    typedef IntegratorSubcyclingBase Subcycl;
    
    struct Params {
        double rtolth, rtolrel;
        Params() { rtolth = 1.0; rtolrel = 0.1; }
        Params(double _rtolth, double _rtolrel) : rtolth(_rtolth), rtolrel(_rtolrel) {}
    };
    
    IntegratorRKFSubcycling(System* system, Force* _force, Mobility* _mobility,
                            SegSegGroups* _subgroups, Params params) : 
    IntegratorRKF(system, _force, _mobility), Subcycl(system, _subgroups) {
        rtolth = params.rtolth;
        rtolrel = params.rtolrel;
    }
    
    struct ErrorFlagNodes {
        System* s;
        DeviceDisNet* net;
        IntegratorRKFSubcycling* itgr;
        bool flagnodes;
        
        ErrorFlagNodes(System* _s, DeviceDisNet* _net, 
                       IntegratorRKFSubcycling* _itgr, bool _flagnodes) :
        s(_s), net(_net), itgr(_itgr), flagnodes(_flagnodes) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const int& i, double& emax0, double& emax1, int& errnans) const {
            auto nodes = net->get_nodes();
            auto cell = net->cell;
            
            itgr->rkf[5](i) = nodes[i].v;
            
            Vec3 err(0.0);
            for (int j = 0; j < 6; j++)
                err += itgr->er[j] * itgr->rkf[j](i);
            err = itgr->newdt * err;
            double errnet = err.norm();
            if (errnet > emax0) emax0 = errnet;
            
            Vec3 xold = s->xold(i);
            xold = cell.pbc_position(nodes[i].pos, xold);
            Vec3 dr = nodes[i].pos - xold;
            double drn = dr.norm();
            double relerr = 0.0;
            if (errnet > itgr->rtolth) {
                if (drn > itgr->rtolth/itgr->rtolrel) {
                    relerr = errnet/drn;
                } else {
                    relerr = 2*itgr->rtolrel;
                }
            }
            if (relerr > emax1) emax1 = relerr;
            if (std::isnan(drn)) errnans++;
            
            if (flagnodes) {
                if (itgr->subgroups->nflag(i) > 0) {
                    if (errnet < itgr->rtol && (errnet < itgr->rtolth || errnet/drn < itgr->rtolrel)) {
                        itgr->subgroups->nflag(i) = 1; // unflag node
                    } else {
                        itgr->subgroups->nflag(i) = 2; // flag node
                    }
                }
            }
        }
    };
    
    inline void rkf_step(int i)
    {
        Base::rkf_step(i);
        
        if (i < 5) {
            // Zero-out velocity of oscillating nodes
            Subcycl::flag_oscillating_nodes(network);
        }
    }
    
    inline void compute_error()
    {
        int errnans = 0;
        Kokkos::parallel_reduce("IntegratorRKFSubcycling::ErrorFlagNodes", network->Nnodes_local,
            ErrorFlagNodes(s, network, this, (group == subgroups->Ngroups-1 && iTry < nTry)),
            Kokkos::Max<double>(errmax[0]), Kokkos::Max<double>(errmax[1]), errnans
        );
        Kokkos::fence();
        
        if (errnans > 0)
            ExaDiS_fatal("Error: %d NaNs found during integration\n", errnans);
    }
    
    inline void non_convergent()
    {
        if (iTry < nTry && group == subgroups->Ngroups-1)
            subgroups->move_interactions(network, iTry);
        
        Base::non_convergent();
    }
    
    void integrate(System* system) {
        Subcycl::integrate(system, this);
    }
    
    void forward_progress_check(DeviceDisNet* net) {
        Subcycl::forward_progress_check(net, vcurr);
    }
    
    void write_restart(FILE* fp) {
        Subcycl::write_restart_sub(fp);
    }
    void read_restart(FILE* fp) {
        Subcycl::read_restart_sub(fp);
    } 
};

/*---------------------------------------------------------------------------
 *
 *    Class:        IntegratorSubcyclingDriver
 *                  Subcycling integrator that drives the integration of
 *                  subgroups in turn and does all the bookkeeping.
 *                  It must be initialized by passing the list of group
 *                  radii as an argument. Right now the number of groups
 *                  is set to 5.
 *                  It is a modified version of the original subcycling
 *                  integrator when drift=1 in ForceSubcycling. In this
 *                  case, for groups > 0, partial nodal forces are just
 *                  stored and the nodes are actually only integrated in
 *                  time when group = 0 is selected, in which case the force
 *                  contributions from all other groups are fetched and
 *                  summed. Since nodes are integrated under the total force,
 *                  this scheme allows for the use of non-linear mobilities.
 *
 *-------------------------------------------------------------------------*/
template<class I>
class IntegratorSubcyclingDriver : public Integrator {
private:
    ForceSubcycling* force;
    Mobility* mobility;
    
    T_x xold;
    
    static const int Ngmax = MAXGROUPS;
    SegSegGroups* subgroups;
    I* integrator;
    SegSegList* segseglist;
    NeighborList* neilist;
    int numsubcyc[Ngmax];
    int totsubcyc;
    
    int TIMER_BUILDGROUPS, TIMER_GROUPN, TIMER_SUBCYCLING;
    
    bool stats = false;
    std::string* fstats;
    
public:
    struct Params {
        typedef typename I::Params IP;
        std::vector<double> rgroups;
        IP Iparams;
        std::string fstats = "";
        Params() { Iparams = IP(); }
        Params(std::vector<double> _rgroups) : rgroups(_rgroups) { Iparams = IP(); }
        Params(std::vector<double> _rgroups, IP _Iparams) : rgroups(_rgroups), Iparams(_Iparams) {}
    };
    
    IntegratorSubcyclingDriver(System* system, Force* _force, Mobility* _mobility, Params params=Params()) : 
    mobility(_mobility)
    {
        force = dynamic_cast<ForceSubcycling*>(_force);
        if (force == nullptr)
            ExaDiS_fatal("Error: must use ForceSubcycling with IntegratorSubcycling\n");
        
        // Need to use subcycling mode drift=1 for non-linear mobility laws
        if (mobility->non_linear && !force->drift) {
            ExaDiS_log("WARNING: using non-linear mobility law with subcycling integrator\n"
            " Switching to force option drift=1 to ensure correct time-integration\n");
            force->drift = 1;
        }
        
        double cutoff = force->fsegseg->get_cutoff();
        subgroups = exadis_new<SegSegGroups>(system, params.rgroups, cutoff);
        force->Ngroups = subgroups->Ngroups;
        neilist = exadis_new<NeighborList>();
        
        // Make sure we have hinges in group 0 for subcycling drift model
        if (force->drift)
            subgroups->r2groups[0] = fmax(subgroups->r2groups[0], 1.0);
        
        integrator = exadis_new<I>(system, force, mobility, subgroups, params.Iparams);
        
        TIMER_BUILDGROUPS = system->add_timer("IntegratorSubcycling build groups");
        TIMER_GROUPN = system->add_timer("IntegratorSubcycling integrate group N");
        TIMER_SUBCYCLING = system->add_timer("IntegratorSubcycling integrate subcycles");
        
        if (!params.fstats.empty()) {
            stats = true;
            fstats = new std::string(params.fstats);
        }
    }
    
    IntegratorSubcyclingDriver(const IntegratorSubcyclingDriver&) = delete;
    
    SegSegGroups* get_subgroups() { return subgroups; }
    
    void write_stats() {
        if (0) {
            printf("Subcycling\n");
            for (int i = 0; i < subgroups->Ngroups; i++)
                printf("  Group %d: cutoff = %e, Nsegseg = %d, fraction = %f, numsubcyc = %d\n", i, 
                (i == subgroups->Ngroups-1) ? force->fsegseg->get_cutoff() : sqrt(subgroups->r2groups[i]), 
                subgroups->Nsegseg[i], subgroups->gfrac[i], numsubcyc[i]);
        }
        if (stats) {
            FILE* fp = fopen(fstats->c_str(), "a");
            fprintf(fp, "%d ", totsubcyc);
            for (int i = 0; i < subgroups->Ngroups; i++) fprintf(fp, "%d ", numsubcyc[i]);
            for (int i = 0; i < subgroups->Ngroups; i++) fprintf(fp, "%d ", subgroups->Nsegseg[i]);
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
    
    // Use a functor so that we can safely call the destructor
    // to free Kokkos allocations at the end of the run
    struct SavePositions {
        DeviceDisNet* net;
        T_x xold;
        SavePositions(DeviceDisNet* _net, T_x& _xold) : net(_net), xold(_xold) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const int& i) const {
            auto nodes = net->get_nodes();
            xold(i) = nodes[i].pos;
        }
    };
    
    struct RestorePositions {
        DeviceDisNet* net;
        T_x xold;
        RestorePositions(DeviceDisNet* _net, T_x& _xold) : net(_net), xold(_xold) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const int& i) const {
            auto nodes = net->get_nodes();
            nodes[i].pos = xold(i);
        }
    };
    
    void build_groups_lists(System* system, bool hinge=true)
    {
        system->devtimer[TIMER_BUILDGROUPS].start();
        
        DeviceDisNet* net = system->get_device_network();
        double cutoff = force->fsegseg->get_cutoff();
        generate_neighbor_list(system, net, neilist, cutoff, Neighbor::NeiSeg);
        
        Kokkos::parallel_for("IntegratorSubcycling::BuildSegSegGroups", net->Nsegs_local, 
            BuildSegSegGroups(net, subgroups, neilist, cutoff, hinge, true)
        );
        Kokkos::fence();
        
        subgroups->init_groups(segseglist);
        /*
        printf("SegSegGroups: cutoff = %e, Nsegseg = %d\n", cutoff, subgroups->Nsegseg_tot);
        for (int i = 0; i < subgroups->Ngroups; i++)
            printf("  Group %d: cutoff = %e, Nsegseg = %d, fraction = %f\n", i,
            (i == subgroups->Ngroups-1) ? cutoff : sqrt(subgroups->r2groups[i]), subgroups->Nsegseg[i], subgroups->gfrac[i]);
        */
        
        Kokkos::parallel_for("IntegratorSubcycling::BuildSegSegGroups", net->Nsegs_local, 
            BuildSegSegGroups(net, subgroups, neilist, cutoff, hinge, false)
        );
        Kokkos::fence();
        
        if (segseglist->use_compute_map) {
            for (int group = 0; group < subgroups->Ngroups; group++) {
                set_group(group);
                SegSegList::build_compute_map<DeviceDisNet>(segseglist, &subgroups->compute_map[group], net);
            }
        }
        
        system->devtimer[TIMER_BUILDGROUPS].stop();
    }
    
    void reset_nodes_flag() {
        Kokkos::deep_copy(subgroups->nflag, 1);
    }
    
    void set_group(int group) {
        force->group = group;
        segseglist->Nsegseg = subgroups->Nsegseg[group];
        segseglist->segseglist = subgroups->segseglist[group];
        if (segseglist->use_compute_map) {
            segseglist->fsegseglist = subgroups->fsegseglist[group];
            segseglist->d_compute_map = subgroups->compute_map[group];
        }
        if (group == subgroups->Ngroups-1) {
            segseglist->use_flag = 1;
        } else {
            segseglist->use_flag = 0;
        }
        integrator->group = group;
    }
    
    void integrate(System* system)
    {
        // Save initial positions
        DeviceDisNet* network = system->get_device_network();
        Kokkos::resize(xold, network->Nnodes_local);
        Kokkos::parallel_for(network->Nnodes_local, SavePositions(network, xold));
        Kokkos::fence();
        
        // Build subcycling groups. These groups contain segment
        // and segment pair lists used for force calculations.
        segseglist = force->fsegseg->get_segseglist();
        bool hinge = !(force->drift);
        build_groups_lists(system, hinge);
        if (force->drift) {
            // Resize subforce arrays
            force->init_subforce(system, subgroups->Ngroups);
        }
        
        // Reset nodes flag
        Kokkos::resize(subgroups->nflag, network->Nnodes_local);
        reset_nodes_flag();
        
        // Time integrate highest group forces to set global time step
        system->devtimer[TIMER_GROUPN].start();
        int group = subgroups->Ngroups-1;
        set_group(group);
        force->compute(system);
        if (force->drift) {
            // Save group forces
            force->save_subforce(network, group);
        }
        mobility->compute(system);
        // Integrate
        integrator->integrate(system);
        if (force->drift) {
            // Restore nodal positions
            Kokkos::parallel_for(network->Nnodes_local, RestorePositions(network, system->xold));
            Kokkos::fence();
        }
        system->devtimer[TIMER_GROUPN].stop();
        
        system->devtimer[TIMER_SUBCYCLING].start();
        // We may need to update groups as some segment pairs may
        // have been moved during integration of the highest group
        set_group(subgroups->Ngroups-2);
        subgroups->update_interactions(network);
        
        
        // Time integrate groups 0,...,N-1 interactions (subcycle)
        // Initialize the time for each group based on whether it has any forces in it
        double subtime[subgroups->Ngroups-1];
        for (int i = 0; i < subgroups->Ngroups-1; i++) {
            subtime[i] = (i == 0 || subgroups->Nsegseg[i] > 0) ? 0.0 : system->realdt;
            numsubcyc[i] = 0;
        }
        numsubcyc[subgroups->Ngroups-1] = 1;
        // Initialize some other stuff
        int oldgroup = -1;
        int nsubcyc;
        totsubcyc = 0;

        // Subcycle until the subcycle group times (subtime[i]) catch up
        // with the highest group time (realdt). Note that nodal forces
        // will reset to zero when subcycling is performed
        bool subcycle = false;
        for (int i = 0; i < subgroups->Ngroups-1; i++)
            subcycle |= (subtime[i] < system->realdt);
        
        while (subcycle) {
            bool cutdt = 0;

            // The group that is furthest behind goes first
            group = subgroups->Ngroups-2;
            for (int i = subgroups->Ngroups-2; i >= 0; i--)
                if (subtime[i] < subtime[group]) group = i;

            // If we switched groups, reset subcycle count
            if (group != oldgroup) {
                nsubcyc = 0;
                reset_nodes_flag();
                set_group(group);
                if (force->drift && group > 0) {
                    force->compute(system);
                    force->save_subforce(network, group);
                }
            }
            oldgroup = group;

            // Make sure we don't pass the highest group in time
            double nextdtsub = integrator->nextdtsub[group];
            double olddtsub;
            if (subtime[group] + nextdtsub > system->realdt) {
                olddtsub  = nextdtsub;
                nextdtsub = system->realdt - subtime[group];
                cutdt = 1;
                integrator->nextdtsub[group] = nextdtsub;
            }

            // Time integrate the chosen group for one subcycle
            if (!force->drift || group == 0)
                force->compute(system);
            mobility->compute(system);
            integrator->integrate(system);
            if (force->drift && group > 0) {
                // Restore nodal positions if group > 0
                Kokkos::parallel_for(network->Nnodes_local, RestorePositions(network, system->xold));
                Kokkos::fence();
            }
            
            // Flag oscillating nodes for subsequent cycles
            if (nsubcyc > 3) integrator->forward_progress_check(network);
            nsubcyc++;

            // Do bookkeeping on the time step and number of subcycles
            if (cutdt && integrator->realdtsub[group] == nextdtsub)
                integrator->nextdtsub[group] = olddtsub;
            subtime[group] += integrator->realdtsub[group];
            
            if (subtime[group] >= system->realdt ||
                (system->realdt - subtime[group]) < 1e-20) {
                subtime[group] = system->realdt;
            }
            
            numsubcyc[group]++;
            totsubcyc++;
            
            // Check if we should continue to subcycle
            subcycle = false;
            for (int i = 0; i < subgroups->Ngroups-1; i++)
                subcycle |= (subtime[i] < system->realdt);
            if (subtime[0] >= system->realdt) subcycle = false;
        }
        system->devtimer[TIMER_SUBCYCLING].stop();
        
        write_stats();
        
        // Set the force group to -1 to skip unnecessary computations
        force->group = -1;
        
        // Restore initial positions as old positions
        // for plastic strain calculation and collisions
        Kokkos::deep_copy(system->xold, xold);
    }
    
    void write_restart(FILE* fp) { integrator->write_restart(fp); }
    void read_restart(FILE* fp) { integrator->read_restart(fp); }
    
    KOKKOS_FUNCTION ~IntegratorSubcyclingDriver() {
        KOKKOS_IF_ON_HOST((
            exadis_delete(integrator);
            exadis_delete(subgroups);
            exadis_delete(neilist);
            if (stats) delete fstats;
        ))
    }
    
    const char* name() { return "IntegratorSubcycling"; }
};

typedef IntegratorSubcyclingDriver<IntegratorRKFSubcycling> IntegratorSubcycling;

} // namespace ExaDiS

#endif
