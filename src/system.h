/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "types.h"
#include "params.h"
#include "crystal.h"

#pragma once
#ifndef EXADIS_SYSTEM_H
#define EXADIS_SYSTEM_H

namespace ExaDiS {

extern FILE* flog;

class System {
public:
    DisNetManager* net_mngr;
    double neighbor_cutoff;
    
    inline SerialDisNet *get_serial_network() { return net_mngr->get_serial_network(); }
    inline DeviceDisNet *get_device_network() { return net_mngr->get_device_network(); }
    
    inline int Nnodes_local() { return net_mngr->Nnodes_local(); }
    inline int Nsegs_local() { return net_mngr->Nsegs_local(); }
    
    inline int Nnodes_total() { return Nnodes_local(); }
    inline int Nsegs_total() { return Nsegs_local(); }
    
    T_x xold;
    
    Mat33 extstress;
    double realdt;
    Mat33 dEp, dWp;
    double density;
    
    Params params;
    Crystal crystal;
    
    System();
    ~System();
    void initialize(Params _params, Crystal _crystal, SerialDisNet *network);
    void plastic_strain();
    void reset_glide_planes();
    void write_config(std::string filename);
    
    int num_ranks;
    int proc_rank;
    
    struct SystemTimer {
        Kokkos::Timer timer;
        double accumtime;
        std::string label;
        SystemTimer() { accumtime = 0.0; }
        SystemTimer(std::string _label) : label(_label) { accumtime = 0.0; }
        void start() { timer.reset(); }
        void stop() { accumtime += timer.seconds(); }
    };
    enum timers {TIMER_FORCE, TIMER_MOBILITY, TIMER_INTEGRATION, TIMER_COLLISION, 
                 TIMER_TOPOLOGY, TIMER_REMESH, TIMER_OUTPUT, TIMER_END};
    SystemTimer timer[TIMER_END];
    
    static const int MAX_DEV_TIMERS = 10;
    int numdevtimer = 0;
    SystemTimer devtimer[MAX_DEV_TIMERS];
    int add_timer(std::string label) {
        if (numdevtimer == MAX_DEV_TIMERS)
            ExaDiS_fatal("Error: MAX_DEV_TIMERS = %d limit reached\n", MAX_DEV_TIMERS);
        devtimer[numdevtimer++].label = label;
        return numdevtimer-1;
    }
    void print_timers();
};

} // namespace ExaDiS

#endif
