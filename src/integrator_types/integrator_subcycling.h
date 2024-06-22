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

#define NGROUPS 5

/*---------------------------------------------------------------------------
 *
 *    Class:        ForceSubcycling
 *                  Force class to be used with the IntegratorSubcycling.
 *                  It contains member properties specific to the subcycling
 *                  scheme.
 *                  
 *                  WARNING: by default (drift=1), this is a modified version
 *                  of the original subcycling scheme in which nodal forces 
 *                  are stored for groups > 0 and summed up when group 0 is
 *                  selected. This allows to integrate nodes under the total
 *                  force during each subcycle, and makes it compatible with
 *                  arbitrary mobility laws. The original subcycling scheme
 *                  is used when drift=0.
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
    typedef ForceSegSegList<SegSegIsoFFT,false> FSegSeg;
    
    bool drift;
    FSeg* fseg;
    FLong* flong;
    FSegSeg* fsegseg;
    int group;
    
    // Should template this eventually...
    static const int Ngroups = NGROUPS;
    Kokkos::View<Vec3*> fgroup[Ngroups-1];
    
public:
    struct Params {
        FSeg::Params FSegParams;
        FLong::Params FLongParams;
        bool drift = true;
        Params(int Ngrid) { FLongParams.Ngrid = Ngrid; }
        Params(int Ngrid, bool _drift) { FLongParams.Ngrid = Ngrid; drift = _drift; }
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
        // Assign no group
        group = -1;
        // Drift scheme
        drift = params.drift;
    }
    
    void init_subforce(System* system, int _Ngroups) {
        if (_Ngroups != Ngroups)
            ExaDiS_fatal("Error: inconsistent number of groups in ForceSubcyclingDrift\n");
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
        DeviceDisNet *net = system->get_device_network();
        if (zero) zero_force(net);
        
        if (group == 0) {
            // This is the group containing the segment forces
            fseg->compute(system, false);
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
    
    virtual const char* name() { return "ForceSubcycling"; }
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
template<int Ng>
struct SegSegGroups
{
    static const int Ngroups = Ng;
    double r2groups[Ng-1];
    int Nsegseg[Ng];
    Kokkos::View<int*> gcount;
    Kokkos::DualView<SegSeg*> segseglist[Ng];
    int Nsegseg_tot;
    double gfrac[Ng];
    
    SegSegList* ssl;
    Kokkos::View<double*> gdist2;
    double rg9s;
    Kokkos::View<int*> countmove;
    Kokkos::View<int*> nflag;
    
    SegSegGroups(System* system, std::vector<double> rgroups, double cutoff) {
        if (Ngroups < 2)
            ExaDiS_fatal("Error: there must be at least 2 groups in SegSegGroups\n");
            
        if (rgroups.size() == 0) {
            // No group radii were provided, let's try to select some appropriate values
            double rmin = fmax(0.3*system->params.minseg, 3.0*system->params.rann);
            double rmax = fmin(system->params.maxseg, cutoff);
            rgroups.resize(Ngroups-1);
            rgroups[0] = 0.0;
            for (int i = 1; i < Ngroups-1; i++) {
                double f = (1.0*(i-1)/(Ngroups-2)-0.5)*M_PI;
                f = 0.5*(sin(f)+1.0);
                rgroups[i] = rmin + f*(rmax-rmin);
            }
        }
        
        if (rgroups.size() != Ngroups-1)
            ExaDiS_fatal("Error: subcycling rgroups array must be of length Ngroups-1\n");
        
        for (int i = 0; i < Ngroups-1; i++) {
            r2groups[i] = rgroups[i] * rgroups[i];
            if (i > 0) {
                if (rgroups[i-1] > rgroups[i])
                    ExaDiS_fatal("Error: subcycling group radii must be of increasing size\n");
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
        for (int i = 0; i < Ngroups; i++) {
            Nsegseg[i] = h_gcount(i);
            Kokkos::resize(segseglist[i], Nsegseg[i]);
            Nsegseg_tot += Nsegseg[i];
        }
        ssl = _ssl;
        Kokkos::resize(ssl->segsegflag, Nsegseg[Ngroups-1]);
        Kokkos::deep_copy(ssl->segsegflag, 1);
        Kokkos::resize(gdist2, Nsegseg[Ngroups-1]);
        Kokkos::deep_copy(countmove, 0);
        for (int i = 0; i < Ngroups; i++)
            gfrac[i] = 1.0*Nsegseg[i]/Nsegseg_tot;
    }
    
    KOKKOS_INLINE_FUNCTION
    int find_group(double dist2) {
        for (int i = 0; i < Ngroups-1; i++)
            if (dist2 < r2groups[i]) return i;
        return Ngroups-1;
    }
    
    struct MoveInteractions {
        DeviceDisNet* net;
        SegSegGroups<Ng>* ssg;
        double rgs;
        
        MoveInteractions(DeviceDisNet* _net, SegSegGroups<Ng>* _ssg, double _rgs) :
        net(_net), ssg(_ssg), rgs(_rgs) {}
        
        MoveInteractions(SegSegGroups<Ng>* _ssg) : ssg(_ssg) {}
        
        struct TagFlag {};
        struct TagMove {};
        
        KOKKOS_INLINE_FUNCTION
        void operator() (TagFlag, const int& i) const {
            auto segs = net->get_segs();
            SegSeg s = ssg->segseglist[Ngroups-1].d_view(i);
            int n1 = segs[s.s1].n1;
            int n2 = segs[s.s1].n2;
            int n3 = segs[s.s2].n1;
            int n4 = segs[s.s2].n2;
            // Flag the interaction to be moved to the lower group
            // if any of its nodes is flagged 2
            if ((ssg->nflag[n1]-2)*(ssg->nflag[n2]-2)*(ssg->nflag[n3]-2)*(ssg->nflag[n4]-2) == 0) {
                if (ssg->gdist2[i] <= rgs && ssg->ssl->segsegflag[i] == 1) {
                    ssg->ssl->segsegflag[i] = 0;
                    Kokkos::atomic_increment(&ssg->countmove(0));
                }
            }
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (TagMove, const int& i) const {
            if (!ssg->ssl->segsegflag[i]) {
                int idx = Kokkos::atomic_fetch_add(&ssg->gcount(Ngroups-2), 1);
                ssg->segseglist[Ngroups-2].d_view(idx) = ssg->segseglist[Ngroups-1].d_view(i);
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
    
    void update_interactions() {
        auto h_countmove = Kokkos::create_mirror_view(countmove);
        Kokkos::deep_copy(h_countmove, countmove);
        Nsegseg[Ngroups-2] += h_countmove(0);
        Kokkos::resize(segseglist[Ngroups-2], Nsegseg[Ngroups-2]);
        using policy = Kokkos::RangePolicy<typename MoveInteractions::TagMove>;
        Kokkos::parallel_for(policy(0, Nsegseg[Ngroups-1]), MoveInteractions(this));
        Kokkos::fence();
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
    bool count_only;
    
    BuildSegSegGroups(N* _net, G* _groups, NeighborList* _neilist, 
                      double _cutoff, bool _count_only) : 
    net(_net), groups(_groups), neilist(_neilist), count_only(_count_only) {
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
                double dist2 = get_min_dist2_segseg(net, i, j, 1);
                if (dist2 < cutoff2) {
                    int groupid = groups->find_group(dist2);
                    if (count_only) {
                        Kokkos::atomic_increment(&groups->gcount(groupid));
                    } else {
                        int idx = Kokkos::atomic_fetch_add(&groups->gcount(groupid), 1);
                        groups->segseglist[groupid].d_view(idx) = SegSeg(i, j);
                        if (groupid == groups->Ngroups-1) groups->gdist2[idx] = dist2;
                    }
                }
            }
        }
    }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        IntegratorRKFSubcycling
 *                  Modified IntegratorRKF class to handle the bookkeeping
 *                  required during integration of subgroups.
 *
 *-------------------------------------------------------------------------*/
template<int Ng>
class IntegratorRKFSubcycling : public IntegratorRKF {
private:
    int Ngroups = Ng;
    SegSegGroups<Ng>* subgroups;
    int nTry = 3;
    
public:
    double nextdtsub[Ng-1], realdtsub[Ng-1];
    int group = -1;
    
    struct Params {
        double rtolth, rtolrel;
        int Ngroups;
        SegSegGroups<Ng>* subgroups;
        Params(double _rtolth, double _rtolrel, SegSegGroups<Ng>* _subgroups) : 
        rtolth(_rtolth), rtolrel(_rtolrel), subgroups(_subgroups) {}
    };
    
    IntegratorRKFSubcycling(System* system, Force* _force, Mobility* _mobility, Params params) : 
    IntegratorRKF(system, _force, _mobility) {
        rtolth = params.rtolth;
        rtolrel = params.rtolrel;
        subgroups = params.subgroups;
        Ngroups = subgroups->Ngroups;
        for (int i = 0; i < Ng-1; i++)
            nextdtsub[i] = nextdt;
    }
    
    struct ErrorFlagNodes {
        System* s;
        DeviceDisNet* net;
        IntegratorRKFSubcycling<Ng>* itgr;
        bool flagnodes;
        
        ErrorFlagNodes(System* _s, DeviceDisNet* _net, 
                       IntegratorRKFSubcycling<Ng>* _itgr, bool _flagnodes) :
        s(_s), net(_net), itgr(_itgr), flagnodes(_flagnodes) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const int& i) const {
            auto nodes = net->get_nodes();
            auto cell = net->cell;
            
            itgr->rkf[5](i) = nodes[i].v;
            
            Vec3 err(0.0);
            for (int j = 0; j < 6; j++)
                err += itgr->er[j] * itgr->rkf[j](i);
            err = itgr->newdt * err;
            double errnet = err.norm();
            Kokkos::atomic_max(&itgr->errmax(0), errnet);
            
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
            Kokkos::atomic_max(&itgr->errmax(1), relerr);
            
            if (flagnodes) {
                if (errnet < itgr->rtol && (errnet < itgr->rtolth || errnet/drn < itgr->rtolrel)) {
                    itgr->subgroups->nflag[i] = 1; // unflag node
                } else {
                    itgr->subgroups->nflag[i] = 2; // flag node
                }
            }
        }
    };
    
    void integrate(System* system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_INTEGRATION].start();
        
        if (group == Ngroups-1) newdt = fmin(maxdt, nextdt);
        else newdt = fmin(maxdt, nextdtsub[group]);
        
        if (newdt <= 0.0) {
            if (group == Ngroups-1) newdt = maxdt;
            else newdt = system->realdt;
        }
        
        s = system;
        network = system->get_device_network();
        
        for (int i = 0; i < 6; i++) {
            Kokkos::resize(rkf[i], network->Nnodes_local);
            Kokkos::deep_copy(rkf[i], 0.0);
        }
        
        // Save nodal data
        Kokkos::resize(system->xold, network->Nnodes_local);
        Kokkos::resize(vcurr, network->Nnodes_local);
        Kokkos::parallel_for("IntegratorRKFSubcycling::PreserveData",
            Kokkos::RangePolicy<TagPreserveData>(0, network->Nnodes_local), *this
        );
        Kokkos::fence();

        int convergent = 0;
        int incrDelta = 1;
        int iTry = -1;
        double errormax = 0.0;
        double relerrormax = 0.0;
        
        if (group > 0 && subgroups->Nsegseg[group] == 0)
            convergent = 1;
        
        while (!convergent) {
            iTry++;
            errormax = 0.0;
            relerrormax = 0.0;
            
            // Apply the Runge-Kutta-Fehlberg integrator one step at a time
            for (int i = 0; i < 5; i++) {
                rkf_step(i);
                force->compute(system);
                mobility->compute(system);
                
                // Zero-out velocity of oscillating nodes
                if (group != Ngroups-1) {
                    DeviceDisNet* net = network;
                    Kokkos::View<int*>& nflag = subgroups->nflag;
                    Kokkos::parallel_for(net->Nnodes_local, KOKKOS_LAMBDA(const int& i) {
                        auto nodes = net->get_nodes();
                        if (nflag[i] == 0) nodes[i].v = Vec3(0.0);
                    });
                    Kokkos::fence();
                }
            }
            
            // Calculate the error
            errmax = Kokkos::View<double*>("IntegratorRKFSubcycling:errmax", 2);
            Kokkos::parallel_for("IntegratorRKFSubcycling::ErrorFlagNodes", network->Nnodes_local,
                ErrorFlagNodes(system, network, this, (group == Ngroups-1 && iTry < nTry))
            );
            Kokkos::fence();
            auto h_errmax = Kokkos::create_mirror_view(errmax);
            Kokkos::deep_copy(h_errmax, errmax);
            errormax = h_errmax(0);
            relerrormax = h_errmax(1);
            
            // If the error is within the tolerance, we've reached
            // convergence so we can accept this dt. Otherwise
            // reposition the nodes and try again.
            if (errormax < rtol && relerrormax < rtolrel) {
                // Calculate final positions
                rkf_step(5);
                convergent = 1;
            }
            
            if (!convergent) {
                // We may want to move interactions between groups
                // to maximize subcycling performance.
                if (iTry < nTry && group == Ngroups-1)
                    subgroups->move_interactions(network, iTry);
                
                // We need to start from the old velocities. So, first,
                // substitute them with the old ones.
                Kokkos::parallel_for("IntegratorRKFSubcycling::RestoreVels",
                    Kokkos::RangePolicy<TagRestoreVels>(0, network->Nnodes_local), *this
                );
                Kokkos::fence();
                
                incrDelta = 0;
                newdt *= dtDecrementFact;
                
                if ((newdt < 1.0e-20) /*&& (system->proc_rank == 0)*/)
                    ExaDiS_fatal("IntegratorRKFSubcycling(): Timestep has dropped below\n"
                                 "minimal threshold to %e. Aborting!\n", newdt);
            }
            
        } // while (!convergent)
        
        if (group == Ngroups-1) system->realdt = newdt;
        else realdtsub[group] = newdt;
        
        if (incrDelta) {
            if (dtVariableAdjustment) {
                double tmp1, tmp2, tmp3, tmp4, factor;
                tmp1 = pow(dtIncrementFact, dtExponent);
                tmp2 = errormax/rtol;
                tmp3 = 1.0 / dtExponent;
                tmp4 = pow(1.0/(1.0+(tmp1-1.0)*tmp2), tmp3);
                factor = dtIncrementFact * tmp4;
                newdt = fmin(maxdt, newdt*factor);
            } else {
                newdt = fmin(maxdt, newdt*dtIncrementFact);
            }
        }
        
        if (group == Ngroups-1) nextdt = newdt;
        else nextdtsub[group] = newdt;
        
        Kokkos::fence();
        system->timer[system->TIMER_INTEGRATION].stop();
    }
    
    void forward_progress_check(DeviceDisNet* net) {
        T_v& vold = vcurr;
        Kokkos::View<int*>& nflag = subgroups->nflag;
        
        Kokkos::parallel_for(net->Nnodes_local, KOKKOS_LAMBDA(const int& i) {
            auto nodes = net->get_nodes();
            if (dot(vold(i), nodes[i].v) < 0.0) nflag[i] = 0;
        });
        Kokkos::fence();
    }
    
    void write_restart(FILE* fp) { 
        fprintf(fp, "nextdt %.17g\n", nextdt);
        for (int i = 0; i < Ng-1; i++)
            fprintf(fp, "nextdtsub%d %.17g\n", i, nextdtsub[i]);
    }
    void read_restart(FILE* fp) {
        fscanf(fp, "nextdt %lf\n", &nextdt);
        int j;
        for (int i = 0; i < Ng-1; i++)
            fscanf(fp, "nextdtsub%d %lf\n", &j, &nextdtsub[i]);
    }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        IntegratorSubcycling
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
class IntegratorSubcycling : public Integrator {
private:
    ForceSubcycling* force;
    Mobility* mobility;
    
    T_x xold;
    
    static const int Ngroups = NGROUPS;
    typedef SegSegGroups<Ngroups> G;
    G* subgroups;
    IntegratorRKFSubcycling<Ngroups>* integrator;
    SegSegList* segseglist;
    
    int TIMER_BUILDGROUPS, TIMER_GROUPN, TIMER_SUBCYCLING;
    
public:
    struct Params {
        std::vector<double> rgroups;
        double rtolth, rtolrel;
        Params() { rtolth = 1.0; rtolrel = 0.1; }
        Params(std::vector<double> _rgroups) : rgroups(_rgroups) { 
            rtolth = 1.0; rtolrel = 0.1;
        }
        Params(std::vector<double> _rgroups, double _rtolth, double _rtolrel) : 
        rgroups(_rgroups), rtolth(_rtolth), rtolrel(_rtolrel) {}
    };
    
    IntegratorSubcycling(System* system, Force* _force, Mobility* _mobility, Params params=Params()) : 
    mobility(_mobility)
    {
        force = dynamic_cast<ForceSubcycling*>(_force);
        if (force == nullptr)
            ExaDiS_fatal("Error: must use ForceSubcycling with IntegratorSubcycling\n");
        
        double cutoff = force->fsegseg->get_cutoff();
        subgroups = exadis_new<G>(system, params.rgroups, cutoff);
        
        IntegratorRKFSubcycling<Ngroups>::Params IParams(params.rtolth, params.rtolrel, subgroups);
        integrator = exadis_new<IntegratorRKFSubcycling<Ngroups> >(system, force, mobility, IParams);
        
        TIMER_BUILDGROUPS = system->add_timer("IntegratorSubcycling build groups");
        TIMER_GROUPN = system->add_timer("IntegratorSubcycling integrate group N");
        TIMER_SUBCYCLING = system->add_timer("IntegratorSubcycling integrate subcycles");
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
    
    void build_groups_lists(System* system)
    {
        system->devtimer[TIMER_BUILDGROUPS].start();
        
        DeviceDisNet* net = system->get_device_network();
        double cutoff = force->fsegseg->get_cutoff();
        NeighborList* neilist = generate_neighbor_list(system, net, cutoff, Neighbor::NeiSeg);
        
        Kokkos::parallel_for("IntegratorSubcycling::BuildSegSegGroups", net->Nsegs_local, 
            BuildSegSegGroups(net, subgroups, neilist, cutoff, true)
        );
        Kokkos::fence();
        
        subgroups->init_groups(segseglist);
        /*
        printf("SegSegGroups: cutoff = %e, Nsegseg = %d\n", cutoff, subgroups->Nsegseg_tot);
        for (int i = 0; i < Ngroups; i++)
            printf("  Group %d: cutoff = %e, Nsegseg = %d, fraction = %f\n", i,
            (i == Ngroups-1) ? cutoff : sqrt(subgroups->r2groups[i]), subgroups->Nsegseg[i], subgroups->gfrac[i]);
        */
        
        Kokkos::parallel_for("IntegratorSubcycling::BuildSegSegGroups", net->Nsegs_local, 
            BuildSegSegGroups(net, subgroups, neilist, cutoff, false)
        );
        Kokkos::fence();
        
        exadis_delete(neilist);
        
        system->devtimer[TIMER_BUILDGROUPS].stop();
    }
    
    void reset_nodes_flag() {
        Kokkos::deep_copy(subgroups->nflag, 1);
    }
    
    void set_group(int group) {
        force->group = group;
        segseglist->Nsegseg = subgroups->Nsegseg[group];
        segseglist->segseglist = subgroups->segseglist[group];
        if (group == Ngroups-1) {
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
        build_groups_lists(system);
        if (force->drift) {
            // Resize subforce arrays
            force->init_subforce(system, Ngroups);
        }
        
        // Reset nodes flag
        Kokkos::resize(subgroups->nflag, network->Nnodes_local);
        reset_nodes_flag();
        
        // Time integrate highest group forces to set global time step
        system->devtimer[TIMER_GROUPN].start();
        int group = Ngroups-1;
        set_group(group);
        force->compute(system);
        mobility->compute(system);
        if (force->drift) {
            // Save group forces
            force->save_subforce(network, group);
        }
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
        subgroups->update_interactions();
        
        
        // Time integrate group 1, 2, 3 and 4 interactions (subcycle)
        // Initialize the time for each group based on whether it has any forces in it
        double subtime[Ngroups-1];
        int numsubcyc[Ngroups-1];
        for (int i = 0; i < Ngroups-1; i++) {
            subtime[i] = (i == 0 || subgroups->Nsegseg[i] > 0) ? 0.0 : system->realdt;
            numsubcyc[i] = 0;
        }
        // Initialize some other stuff
        int oldgroup = -1;
        int nsubcyc;
        int totsubcyc = 0;

        // Subcycle until the subcycle group times (subtime[i]) catch up
        // with the highest group time (realdt). Note that nodal forces
        // will reset to zero when subcycling is performed
        bool subcycle = false;
        for (int i = 0; i < Ngroups-1; i++)
            subcycle |= (subtime[i] < system->realdt);
        
        while (subcycle) {
            bool cutdt = 0;

            // The group that is furthest behind goes first
            group = Ngroups-2;
            for (int i = Ngroups-2; i >= 0; i--)
                if (subtime[i] < subtime[group]) group = i;

            // If we switched groups, reset subcycle count
            if (group != oldgroup) {
                nsubcyc = 0;
                reset_nodes_flag();
                set_group(group);
                force->compute(system);
                mobility->compute(system);
                if (force->drift && group > 0)
                    force->save_subforce(network, group);
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
            
            numsubcyc[group]++;
            totsubcyc++;
            
            // Check if we should continue to subcycle
            subcycle = false;
            for (int i = 0; i < Ngroups-1; i++)
                subcycle |= (subtime[i] < system->realdt);
        }
        system->devtimer[TIMER_SUBCYCLING].stop();
        
        /*
        printf("Subcycling\n");
        for (int i = 0; i < Ngroups; i++)
            printf("  Group %d: cutoff = %e, Nsegseg = %d, fraction = %f, numsubcyc = %d\n", i, 
            (i == Ngroups-1) ? force->fsegseg->get_cutoff() : sqrt(subgroups->r2groups[i]), 
            subgroups->Nsegseg[i], subgroups->gfrac[i], (i == Ngroups-1) ? 1 : numsubcyc[i]);
        */
        
        // Set the force group to -1 to skip unnecessary computations
        force->group = -1;
        
        // Restore initial positions as old positions
        // for plastic strain calculation and collisions
        Kokkos::deep_copy(system->xold, xold);
    }
    
    void write_restart(FILE* fp) { integrator->write_restart(fp); }
    void read_restart(FILE* fp) { integrator->read_restart(fp); }
    
    ~IntegratorSubcycling() {
        exadis_delete(integrator);
        exadis_delete(subgroups);
    }
    
    const char* name() { return "IntegratorSubcycling"; }
};

} // namespace ExaDiS

#endif
