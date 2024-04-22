/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "system.h"

namespace ExaDiS {

FILE* flog = nullptr;

/*---------------------------------------------------------------------------
 *
 *    Function:     System::System()
 *
 *-------------------------------------------------------------------------*/
System::System()
{
#ifdef MPI
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
#else
    num_ranks = 1;
    proc_rank = 0;
#endif

    net_mngr = nullptr;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::initialize()
 *
 *-------------------------------------------------------------------------*/
void System::initialize(Params _params, Crystal _crystal, SerialDisNet *network)
{
    //ExaDiS_log("Initialize system\n");
    
    params = _params;
    crystal = _crystal;
    
#ifdef MPI
    // handle communications
    ExaDiS_fatal("System::initialize not implemented for MPI\n");
#endif
    
    // Allocate network manager on unified host/device space
    if (!network)
        ExaDiS_fatal("Error: undefined initial dislocation configuration\n");
    network->generate_connectivity();
    if (crystal.use_glide_planes) {
        network->update_ptr();
        for (int i = 0; i < network->number_of_segs(); i++) {
            Vec3 p = crystal.find_seg_glide_plane(network, i);
            if (p.norm2() > 1e-5) network->segs[i].plane = p;
        }
    }
    net_mngr = exadis_new<DisNetManager>(network);
    neighbor_cutoff = 0.0;
    
    // Initialize variables 
    extstress.zero();
    dEp.zero();
    dWp.zero();
    realdt = 0.0;
    density = network->dislocation_density(params.burgmag);
    
    // Initialize timers
    for (int i = 0; i < TIMER_END; i++)
        timer[i].accumtime = 0.0;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::~System()
 *
 *-------------------------------------------------------------------------*/
System::~System()
{
    exadis_delete(net_mngr);
    if (flog) fclose(flog);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::print_timers()
 *
 *-------------------------------------------------------------------------*/
void System::print_timers()
{
    ExaDiS_log("------------------------------------------\n");
    ExaDiS_log("Force time: %f sec\n", timer[TIMER_FORCE].accumtime);
    ExaDiS_log("Mobility time: %f sec\n", timer[TIMER_MOBILITY].accumtime);
    ExaDiS_log("Integration time: %f sec\n", timer[TIMER_INTEGRATION].accumtime);
    ExaDiS_log("Collision time: %f sec\n", timer[TIMER_COLLISION].accumtime);
    ExaDiS_log("Topology time: %f sec\n", timer[TIMER_TOPOLOGY].accumtime);
    ExaDiS_log("Remesh time: %f sec\n", timer[TIMER_REMESH].accumtime);
    ExaDiS_log("Output time: %f sec\n", timer[TIMER_OUTPUT].accumtime);
    ExaDiS_log("------------------------------------------\n");
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::plastic_strain()
 *                  Compute the plastic strain by summing the area swept
 *                  by all dislocations. This is done in blocks of segments
 *                  to avoid race conditions on devices.
 *
 *-------------------------------------------------------------------------*/
template <class N>
class PlasticStrain {
private:
    System* system;
    N* net;
    double vol, bmag;
public:
    PlasticStrain(System* _system, N* _net) : system(_system), net(_net)
    {
        vol = net->cell.volume();
        bmag = system->params.burgmag;
        system->dEp.zero();
        system->dWp.zero();
        system->density = 0.0;
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const team_handle& team) const
    {
        int ts = team.team_size();
        int lid = team.league_rank();
        
        auto nodes = net->get_nodes();
        auto segs = net->get_segs();
        auto cell = net->cell;
        
        Mat33 E, W;
        double rho;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, ts), 
        [=](int& t, Mat33& Esum, Mat33& Wsum, double& rhosum) {
            int i = lid*ts + t; // segment id
            if (i < net->Nsegs_local) {
                int n1 = segs[i].n1;
                int n2 = segs[i].n2;
                Vec3 b = segs[i].burg;
                
                Vec3 r1 = nodes[n1].pos;
                Vec3 r2 = cell.pbc_position(r1, nodes[n2].pos);
                Vec3 r3 = cell.pbc_position(r1, system->xold(n1));
                Vec3 r4 = cell.pbc_position(r3, system->xold(n2));
                Vec3 n = 0.5*cross(r2-r3, r1-r4);
                
                Mat33 P = 1.0/vol * outer(n, b);
                Esum += 0.5 * (P + P.transpose());
                Wsum += 0.5 * (P - P.transpose());
                rhosum += 1.0/vol/bmag/bmag * (r2-r1).norm(); // 1/m^2
            }
        }, E, W, rho);
        
        Kokkos::single(Kokkos::PerTeam(team), [=]() {
            Kokkos::atomic_add(&system->dEp, E);
            Kokkos::atomic_add(&system->dWp, W);
            Kokkos::atomic_add(&system->density, rho);
        });
    }
};

void System::plastic_strain()
{
    DeviceDisNet* net = get_device_network();
    TeamSize ts = get_team_sizes(net->Nsegs_local);
    Kokkos::parallel_for(Kokkos::TeamPolicy<>(ts.num_teams, ts.team_size), 
        PlasticStrain<DeviceDisNet>(this, net)
    );
    Kokkos::fence();
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::reset_glide_planes()
 *                  Reset glide planes when needed. Some glide plane may
 *                  initially not be identifiable when segments are created
 *                  but may become identifiable after some motion.
 *                  Also enforce the glide planes by projecting back nodes
 *                  along glide constraints using previous positions.
 *
 *-------------------------------------------------------------------------*/
void System::reset_glide_planes()
{
    if (!crystal.use_glide_planes) return;
    
    DeviceDisNet* net = get_device_network();
    Crystal* cryst = &crystal;
    
    // Fix glide plane violations
    T_x& xprev = xold;
    Kokkos::parallel_for(net->Nnodes_local, KOKKOS_LAMBDA(const int& i) {
        auto nodes = net->get_nodes();
        auto segs = net->get_segs();
        auto conn = net->get_conn();
        auto cell = net->cell;
        
        // Determine the number of unique glide planes.
        int numplanes = 0;
        Vec3 planes[MAX_CONN];
        for (int j = 0; j < conn[i].num; j++) {
            int s = conn[i].seg[j];
            Vec3 p = segs[s].plane;
            if (j == 0) {
                planes[0] = p;
                numplanes = 1;
            } else {
                for (int k = 0; k < numplanes; k++)
                    p = p.orthogonalize(planes[k]);
                if (p.norm2() > 1e-5)
                    planes[numplanes++] = p.normalized();
            }
        }
        
        Vec3 p = cell.pbc_position(xprev(i), nodes[i].pos);
        if (numplanes == 1) {
            double eqn = dot(planes[0], p) - dot(planes[0], xprev(i));
            p -= eqn * planes[0];
        } else if (numplanes == 2) {
            Vec3 l = cross(planes[0], planes[1]).normalized();
            Vec3 dr = p - xprev(i);
            p = xprev(i) + dot(l, dr) * l;
        } else {
            p = xprev(i);
        }
        nodes[i].pos = cell.pbc_fold(p);
    });
    Kokkos::fence();
    
    // Now reset the glide planes
    Kokkos::parallel_for(net->Nsegs_local, KOKKOS_LAMBDA(const int& i) {
        auto segs = net->get_segs();
        Vec3 p = cryst->find_seg_glide_plane(net, i);
        if (p.norm2() > 1e-5) segs[i].plane = p;
    });
    Kokkos::fence();
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::write_config()
 *
 *-------------------------------------------------------------------------*/
void System::write_config(std::string filename)
{
#ifdef MPI
    // handle communications
    ExaDiS_fatal("System::write_config not implemented for MPI\n");
#endif

    Kokkos::fence();
    timer[TIMER_OUTPUT].start();

    int active_net = net_mngr->get_active();
    SerialDisNet *network = get_serial_network();
    network->write_data(filename);
    
    // We did not make any changes to the network, so
    // let's avoid making unnecessary memory copies
    net_mngr->set_active(active_net);

    Kokkos::fence();
    timer[TIMER_OUTPUT].stop();
}
    
} // namespace ExaDiS
