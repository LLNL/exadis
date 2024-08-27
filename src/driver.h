/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_DRIVER_H
#define EXADIS_DRIVER_H

#include "exadis.h"
#include <iostream>

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        ExaDiSApp
 *                  Example of a driver to peform a DDD simulation
 *
 *-------------------------------------------------------------------------*/
class ExaDiSApp {
public:
    System* system = nullptr;
    Force* force = nullptr;
    Mobility* mobility = nullptr;
    Integrator* integrator = nullptr;
    Collision* collision = nullptr;
    Topology* topology = nullptr;
    Remesh* remesh = nullptr;
    std::string outputdir = "";
    
    bool dealloc = true;
    bool setup = false;
    bool init = false;
    bool log = true;
    bool restart = false;
    
    int istep;
    Mat33 Etot;
    double stress, strain, pstrain;
    double tottime;
    Vec3 edir;
    
    Kokkos::Timer timer;
    bool timeronefile = true;
    
    struct Stepper {
        enum Types {NUM_STEPS, MAX_STEPS, MAX_STRAIN, MAX_TIME, MAX_WALLTIME};
        int type = NUM_STEPS;
        int maxsteps = 0;
        double stopval = 0.0;
        Stepper(int _type, int _maxsteps) : type(_type), maxsteps(_maxsteps) {}
        Stepper(int _type, double _stopval) : type(_type), stopval(_stopval) {}
        Stepper& operator=(int nsteps) { type = NUM_STEPS; maxsteps = nsteps; return *this; }
        bool iterate(ExaDiSApp* exadis);
    };
    static Stepper NUM_STEPS(int nsteps) { return Stepper(Stepper::NUM_STEPS, nsteps); }
    static Stepper MAX_STEPS(int maxsteps) { return Stepper(Stepper::MAX_STEPS, maxsteps); }
    static Stepper MAX_STRAIN(double maxstrain) { return Stepper(Stepper::MAX_STRAIN, maxstrain); }
    static Stepper MAX_TIME(double maxtime) { return Stepper(Stepper::MAX_TIME, maxtime); }
    static Stepper MAX_WALLTIME(double maxtime) { return Stepper(Stepper::MAX_WALLTIME, maxtime); }
    
    struct Prop {
        enum fields {STEP, STRAIN, STRESS, DENSITY, NNODES, NSEGS, DT, TIME, WALLTIME, EDIR, ALLSTRESS};
    };
    
    enum Loadings {STRESS_CONTROL, STRAIN_RATE_CONTROL};
    struct Control {
        Stepper nsteps = NUM_STEPS(100);
        int loading = STRESS_CONTROL;
        double erate = 1e3;
        Vec3 edir = Vec3(0.0, 0.0, 1.0);
        Mat33 appstress = Mat33().zero();
        int rotation = 0;
        int printfreq = 1;
        int propfreq = 10;
        int outfreq = 100;
        std::vector<Prop::fields> props = {Prop::STEP, Prop::STRAIN, Prop::STRESS, Prop::DENSITY};
    };
    
    ExaDiSApp(int argc, char* argv[]);
    ExaDiSApp();
    ~ExaDiSApp();
    
    virtual void set_modules(
        Force* _force,
        Mobility* _mobility,
        Integrator* _integrator,
        Collision* _collision,
        Topology* _topology,
        Remesh* _remesh
    );
    virtual void set_simulation(std::string restartfile="");
    virtual void set_directory();
    
    virtual void initialize(Control& ctrl);
    virtual void step(Control& ctrl);
    virtual void run(Control& ctrl);
    virtual void update_mechanics(Control& ctrl);
    virtual void output(Control& ctrl);
    
    virtual void write_restart(std::string restartfile);
    virtual void read_restart(std::string restartfile);
    
    double von_mises(const Mat33 &T) {
        Mat33 Tdev = T - 1.0/3.0*T.trace()*Mat33().eye();
        return sqrt(3.0/2.0*dot(Tdev, Tdev));
    }
};
    
} // namespace ExaDiS

#endif
