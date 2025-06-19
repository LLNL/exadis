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
    CrossSlip* crossslip = nullptr;
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
    double outfiletime = 0.0;
    int outfilecounter = 0;
    
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
        enum fields {STEP, STRAIN, STRESS, DENSITY, NNODES, NSEGS, DT, TIME, WALLTIME, EDIR, RORIENT, ALLSTRESS};
        static fields get_field(std::string& name) {
            if (name == "Step" || name == "step") return STEP;
            else if (name == "Strain" || name == "strain") return STRAIN;
            else if (name == "Stress" || name == "stress") return STRESS;
            else if (name == "Density" || name == "density") return DENSITY;
            else if (name == "Nnodes") return NNODES;
            else if (name == "Nsegs") return NSEGS;
            else if (name == "DT" || name == "dt") return DT;
            else if (name == "Time" || name == "time") return TIME;
            else if (name == "Walltime" || name == "walltime") return WALLTIME;
            else if (name == "edir") return EDIR;
            else if (name == "Rorient") return RORIENT;
            else if (name == "Allstress" || name == "allstress") return ALLSTRESS;
            else ExaDiS_fatal("Unknown control property name = %s\n", name.c_str());
            return STEP;
        }
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
        double outfreqdt = -1.0;
        int oprecwritefreq = 0;
        int oprecfilefreq = 0;
        int oprecposfreq = 0;
        std::vector<Prop::fields> props = {Prop::STEP, Prop::STRAIN, Prop::STRESS, Prop::DENSITY};
        void set_props(std::vector<std::string> fields) {
            props.clear();
            for (auto f : fields)
                props.push_back(Prop::get_field(f));
        }
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
        Remesh* _remesh,
        CrossSlip* _crossslip = nullptr
    );
    virtual void set_simulation(std::string restartfile="");
    virtual void set_directory();
    
    virtual void initialize(Control& ctrl, bool check_modules=true);
    virtual void step(Control& ctrl);
    virtual void run(Control& ctrl);
    virtual void update_mechanics(Control& ctrl);
    virtual void output(Control& ctrl);
    
    virtual void oprec_save_integration(Control& ctrl);
    virtual void oprec_replay(Control& ctrl, std::string oprec_file);
    
    virtual void write_restart(std::string restartfile);
    virtual void read_restart(std::string restartfile);
    
    double von_mises(const Mat33 &T) {
        Mat33 Tdev = T - 1.0/3.0*T.trace()*Mat33().eye();
        return sqrt(3.0/2.0*dot(Tdev, Tdev));
    }
};
    
} // namespace ExaDiS

#endif
