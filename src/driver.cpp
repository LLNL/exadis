/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "exadis.h"
#include "driver.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp
 *                  Initialization of a DDD simulation
 *
 *-------------------------------------------------------------------------*/
ExaDiSApp::ExaDiSApp(int argc, char* argv[])
{
    // Allocate system on unified host/device space
    system = exadis_new<System>();
    
    istep = 0;
    Etot = Mat33().zero();
    stress = strain = pstrain = 0.0;
    tottime = 0.0;
   
    if (system->proc_rank == 0) {
        printf("----------------------------------------------------\n");
        printf("ExaDiS\n");
        printf("----------------------------------------------------\n");
        printf("Version: %s\n", EXADIS_VERSION);
#ifdef MPI
        printf("MPI build\n");
#else
        printf("Serial build\n");
#endif
        Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout);
    }
}

ExaDiSApp::ExaDiSApp()
{
    istep = 0;
    Etot = Mat33().zero();
    stress = strain = pstrain = 0.0;
    tottime = 0.0;
    dealloc = false;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ~ExaDiSApp
 *                  Termination of a DDD simulation / free memory
 *
 *-------------------------------------------------------------------------*/
ExaDiSApp::~ExaDiSApp()
{
    if (!dealloc) return;
    exadis_delete(system);
    if (force) delete force;
    if (mobility) delete mobility;
    if (integrator) delete integrator;
    if (crossslip) delete crossslip;
    if (collision) delete collision;
    if (topology) delete topology;
    if (remesh) delete remesh;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::set_modules()
 *                  Set the various base modules (force, mobility, etc.)
 *                  required to run a simulation
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::set_modules(
    Force* _force,
    Mobility* _mobility,
    Integrator* _integrator,
    Collision* _collision,
    Topology* _topology,
    Remesh* _remesh,
    CrossSlip* _crossslip)
{
    force = _force;
    mobility = _mobility;
    integrator = _integrator;
    collision = _collision;
    topology = _topology;
    remesh = _remesh;
    crossslip = _crossslip;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::set_simulation()
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::set_simulation(std::string restartfile)
{
    if (!restartfile.empty()) {
        read_restart(restartfile);
        restart = true;
    }
    
    set_directory();
    setup = true;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::set_directory()
 *                  Set the output directory
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::set_directory()
{
    if (system->proc_rank == 0) {
        if (!restart) remove_directory(outputdir);
        create_directory(outputdir);
    }
    
    // Set log file
    std::string logfile = outputdir + "/exadis.log";
    if (log) flog = fopen(logfile.c_str(), "a");
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::write_restart()
 *                  Write file to restart a simulation from a previous state
 *
 *-------------------------------------------------------------------------*/
#define RESTART_VERSION "1.0"
void ExaDiSApp::write_restart(std::string restartfile)
{
    FILE* fp = fopen(restartfile.c_str(), "w");
    if (fp == NULL)
        ExaDiS_fatal("Error: cannot open restart file %s\n", restartfile.c_str());
        
    ExaDiS_log("Writing restart file\n");
    ExaDiS_log(" Restart file: %s\n", restartfile.c_str());
    
    // header
    fprintf(fp, "# ExaDiS restart file\n");
    fprintf(fp, "\n");
    fprintf(fp, "version %s\n", RESTART_VERSION);
    time_t tp; time(&tp);
    char timestr[64];
    asctime_r(localtime(&tp), timestr);
    timestr[strlen(timestr)-1] = 0;
    fprintf(fp, "date_and_time %s\n", timestr);
    fprintf(fp, "\n");
    
    // driver
    fprintf(fp, "step %d\n", istep);
    fprintf(fp, "Etot %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
    Etot.xx(), Etot.xy(), Etot.xz(), Etot.yx(), Etot.yy(), Etot.yz(), Etot.zx(), Etot.zy(), Etot.zz());
    fprintf(fp, "stress %.17g\n", stress);
    fprintf(fp, "strain %.17g\n", strain);
    fprintf(fp, "pstrain %.17g\n", pstrain);
    fprintf(fp, "tottime %.17g\n", tottime);
    fprintf(fp, "edir %.17g %.17g %.17g\n", edir.x, edir.y, edir.z);
    fprintf(fp, "\n");
    
    // system
    fprintf(fp, "extstress %.17g %.17g %.17g %.17g %.17g %.17g\n",
    system->extstress.xx(), system->extstress.yy(), system->extstress.zz(),
    system->extstress.xy(), system->extstress.xz(), system->extstress.yz());
    fprintf(fp, "realdt %.17g\n", system->realdt);
    fprintf(fp, "density %e\n", system->density);
    fprintf(fp, "\n");
    
    // integrator
    fprintf(fp, "integrator %s\n", integrator->name());
    integrator->write_restart(fp);
    fprintf(fp, "\n");
    
    //crystal
    fprintf(fp, "crystal type %d\n", system->crystal.type);
    fprintf(fp, "crystal orientation %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
    system->crystal.R.xx(), system->crystal.R.xy(), system->crystal.R.xz(),
    system->crystal.R.yx(), system->crystal.R.yy(), system->crystal.R.yz(),
    system->crystal.R.zx(), system->crystal.R.zy(), system->crystal.R.zz());
    fprintf(fp, "\n");
    
    // network
    int active_net = system->net_mngr->get_active();
    SerialDisNet* net = system->get_serial_network();
    
    fprintf(fp, "pbc %d %d %d\n", net->cell.xpbc, net->cell.ypbc, net->cell.zpbc);
    fprintf(fp, "H %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
    net->cell.H.xx(), net->cell.H.xy(), net->cell.H.xz(),
    net->cell.H.yx(), net->cell.H.yy(), net->cell.H.yz(),
    net->cell.H.zx(), net->cell.H.zy(), net->cell.H.zz());
    fprintf(fp, "origin %.17g %.17g %.17g\n", net->cell.origin.x, net->cell.origin.y, net->cell.origin.z);
    fprintf(fp, "\n");
    
    fprintf(fp, "Nnodes %d\n", system->Nnodes_total());
    for (int i = 0; i < net->number_of_nodes(); i++)
        fprintf(fp, "%d %.17g %.17g %.17g %d\n", i, 
        net->nodes[i].pos.x, net->nodes[i].pos.y, net->nodes[i].pos.z, 
        net->nodes[i].constraint);
    fprintf(fp, "\n");
        
    fprintf(fp, "Nsegs %d\n", system->Nsegs_total());
    for (int i = 0; i < net->number_of_segs(); i++)
        fprintf(fp, "%d %d %.17g %.17g %.17g %.17g %.17g %.17g\n", net->segs[i].n1, net->segs[i].n2,
        net->segs[i].burg.x, net->segs[i].burg.y, net->segs[i].burg.z,
        net->segs[i].plane.x, net->segs[i].plane.y, net->segs[i].plane.z);
    fprintf(fp, "\n");
    
    system->net_mngr->set_active(active_net);
        
    fclose(fp);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::read_restart()
 *                  Read restart file to continue a simulation
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::read_restart(std::string restartfile)
{
#ifdef MPI
    ExaDiS_fatal("read_restart() not implemented for MPI\n");
#endif
    
    FILE* fp = fopen(restartfile.c_str(), "r");
    if (fp == NULL)
        ExaDiS_fatal("Error: cannot open restart file %s\n", restartfile.c_str());
    
    ExaDiS_log("Reading restart file\n");
    ExaDiS_log(" Restart file: %s\n", restartfile.c_str());
    
    char *line = NULL;
    size_t len = 0;
    
    // Read general information
    double version = 0.0;
    Cell cell;
    
    while (getline(&line, &len, fp) != -1) {
        // header
        if (strncmp(line, "version", 7) == 0) { sscanf(line, "version %lf\n", &version); }
        
        // driver
        else if (strncmp(line, "step", 4) == 0) { sscanf(line, "step %d\n", &istep); }
        else if (strncmp(line, "Etot", 4) == 0) {
            sscanf(line, "Etot %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
            &Etot[0][0], &Etot[0][1], &Etot[0][2],
            &Etot[1][0], &Etot[1][1], &Etot[1][2],
            &Etot[2][0], &Etot[2][1], &Etot[2][2]);
        }
        else if (strncmp(line, "stress", 6) == 0) { sscanf(line, "stress %lf\n", &stress); }
        else if (strncmp(line, "strain", 6) == 0) { sscanf(line, "strain %lf\n", &strain); }
        else if (strncmp(line, "pstrain", 7) == 0) { sscanf(line, "pstrain %lf\n", &pstrain); }
        else if (strncmp(line, "tottime", 7) == 0) { sscanf(line, "tottime %lf\n", &tottime); }
        else if (strncmp(line, "edir", 4) == 0) { sscanf(line, "edir %lf %lf %lf\n", &edir.x, &edir.y, &edir.z); }
        
        // system
        else if (strncmp(line, "extstress", 9) == 0) {
            double sxx, syy, szz, sxy, sxz, syz;
            sscanf(line, "extstress %lf %lf %lf %lf %lf %lf\n",
            &sxx, &syy, &szz, &sxy, &sxz, &syz);
            system->extstress.symmetric(sxx, syy, szz, sxy, sxz, syz);
        }
        else if (strncmp(line, "realdt", 6) == 0) { 
            sscanf(line, "realdt %lf\n", &system->realdt);
            system->params.nextdt = system->realdt;
        }
        
        // integrator
        // need to save/set some integrator class members to ensure reproducibility
        else if (strncmp(line, "integrator", 10) == 0) {
            char intname[100];
            sscanf(line, "integrator %s\n", intname);
            bool set = 0;
            if (integrator) {
                if (strcmp(integrator->name(), intname) == 0) {
                    integrator->read_restart(fp);
                    set = 1;
                }
            }
            if (!set) ExaDiS_log("Warning: skipped resetting the time-integrator\n");
        }
        
        // crystal
        else if (strncmp(line, "crystal type", 12) == 0) {
            int crystal;
            sscanf(line, "crystal type %d\n", &crystal);
            if (crystal != system->crystal.type)
                ExaDiS_fatal("Error: inconsistent crystal type for restart\n");
        }
        else if (strncmp(line, "crystal orientation", 19) == 0) {
            sscanf(line, "crystal orientation %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
            &system->crystal.R[0][0], &system->crystal.R[0][1], &system->crystal.R[0][2],
            &system->crystal.R[1][0], &system->crystal.R[1][1], &system->crystal.R[1][2],
            &system->crystal.R[2][0], &system->crystal.R[2][1], &system->crystal.R[2][2]);
        }
        
        // cell
        else if (strncmp(line, "pbc", 3) == 0) { sscanf(line, "pbc %d %d %d\n", &cell.xpbc, &cell.ypbc, &cell.zpbc); }
        else if (strncmp(line, "H", 1) == 0) {
            sscanf(line, "H %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
            &cell.H[0][0], &cell.H[0][1], &cell.H[0][2],
            &cell.H[1][0], &cell.H[1][1], &cell.H[1][2],
            &cell.H[2][0], &cell.H[2][1], &cell.H[2][2]);
        }
        else if (strncmp(line, "origin", 6) == 0) { sscanf(line, "origin %lf %lf %lf\n", &cell.origin.x, &cell.origin.y, &cell.origin.z); }
        
        // network
        else if (strncmp(line, "Nnodes", 6) == 0) {
            break;
        }
    }
    
    printf(" version = %g\n", version);
    
    // network
    SerialDisNet* net = system->get_serial_network();
    net->cell = cell;
    net->nodes.clear();
    net->segs.clear();
    net->conn.clear();
    
    int Nnodes;
    sscanf(line, "Nnodes %d\n", &Nnodes);
    printf(" nodes = %d\n", Nnodes);
    for (int i = 0; i < Nnodes; i++) {
        int id, constraint;
        Vec3 pos;
        fscanf(fp, "%d %lf %lf %lf %d\n", &id, &pos.x, &pos.y, &pos.z, &constraint);
        net->add_node(pos, constraint);
    }
    
    int Nsegs;
    fscanf(fp, "Nsegs %d\n", &Nsegs);
    printf(" segments = %d\n", Nsegs);
    for (int i = 0; i < Nsegs; i++) {
        int n1, n2;
        Vec3 burg, plane;
        fscanf(fp, "%d %d %lf %lf %lf %lf %lf %lf\n", &n1, &n2,
        &burg.x, &burg.y, &burg.z, &plane.x, &plane.y, &plane.z);
        net->add_seg(n1, n2, burg, plane);
    }
    
    net->generate_connectivity();
    system->density = net->dislocation_density(system->params.burgmag);
    
    free(line);
    fclose(fp);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::update_mechanics()
 *                  Update stress, strain, crystal rotation, etc.
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::update_mechanics(Control& ctrl)
{
    if (ctrl.rotation) {
        // Counter-rotate stress wrt dislocation configuration
        double p1 = - system->dWp.zy();
        double p2 =   system->dWp.zx();
        double p3 = - system->dWp.yx();
        
        #define Ss(a) ((a))
        #define Cs(a) (1.0 - (0.5 * (a)*(a)))
        Mat33 Rspin;
        Rspin[0][0] =  Cs(p3)*Cs(p2);
        Rspin[1][1] =  Cs(p3)*Cs(p1) + Ss(p3)*Ss(p2)*Ss(p1);
        Rspin[2][2] =  Cs(p2)*Cs(p1);
        Rspin[0][1] = -Ss(p3)*Cs(p1) + Cs(p3)*Ss(p1)*Ss(p2);
        Rspin[1][2] = -Cs(p3)*Ss(p1) + Ss(p3)*Ss(p2)*Cs(p1);
        Rspin[2][0] = -Ss(p2);
        Rspin[0][2] =  Ss(p3)*Ss(p1) + Cs(p3)*Cs(p1)*Ss(p2);
        Rspin[1][0] =  Ss(p3)*Cs(p2);
        Rspin[2][1] =  Cs(p2)*Ss(p1);
        
        edir = Rspin * edir;
        edir = edir.normalized();
        
        // Rotate stress
        system->extstress = Rspin * system->extstress * Rspin.transpose();
    }
    
    if (ctrl.loading == STRAIN_RATE_CONTROL) {
        Mat33 A = outer(edir, edir);
        double dpstrain = dot(system->dEp, A);
        double dstrain = ctrl.erate * system->realdt;
        double Eyoung = 2.0 * system->params.MU * (1.0 + system->params.NU);
        double dstress = Eyoung * (dstrain - dpstrain);
        
        system->extstress += dstress * A;
        Etot += dstrain * A;
        strain = dot(Etot, A);
        stress = dot(system->extstress, A);
        pstrain += dpstrain;
    }
    
    if (ctrl.loading == STRESS_CONTROL) {
        strain = 0.0;
        stress = von_mises(system->extstress);
        double dpstrain = von_mises(system->dEp);
        pstrain += dpstrain;
    }
    
    tottime += system->realdt;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::output()
 *                  Output stress, strain, config, etc.
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::output(Control& ctrl)
{
    // Print info in the console
    if (istep%ctrl.printfreq == 0) {
        if (ctrl.loading == STRAIN_RATE_CONTROL) {
            ExaDiS_log("Step = %6d: nodes = %d, dt = %e, strain = %e, elapsed = %.1f sec\n",
            istep, system->Nnodes_total(), system->realdt, strain, timer.seconds());
        } else {
            ExaDiS_log("Step = %6d: nodes = %d, dt = %e, time = %e, elapsed = %.1f sec\n",
            istep, system->Nnodes_total(), system->realdt, tottime, timer.seconds());
        }
    }
    
    // Output properties
    if (istep%ctrl.propfreq == 0 || istep == 1) {
        std::string filename = outputdir+"/stress_strain_dens.dat";
        FILE *fp = fopen(filename.c_str(), "a");
        if (fp == NULL)
            ExaDiS_fatal("Error: cannot open output file %s\n", filename.c_str());
        int init = (istep == 0);
        if (init) fprintf(fp, "#");
        for (auto field : ctrl.props) {
            if (field == Prop::STEP) if (init) { fprintf(fp, " Step"); } else { fprintf(fp, "%d ", istep); }
            else if (field == Prop::STRAIN) if (init) { fprintf(fp, " Strain"); } else { fprintf(fp, "%e ", strain); }
            else if (field == Prop::STRESS) if (init) { fprintf(fp, " Stress"); } else { fprintf(fp, "%e ", stress); }
            else if (field == Prop::DENSITY) if (init) { fprintf(fp, " Density"); } else { fprintf(fp, "%e ", system->density); }
            else if (field == Prop::NNODES) if (init) { fprintf(fp, " Nnodes"); } else { fprintf(fp, "%d ", system->Nnodes_total()); }
            else if (field == Prop::NSEGS) if (init) { fprintf(fp, " Nsegs"); } else { fprintf(fp, "%d ", system->Nsegs_total()); }
            else if (field == Prop::DT) if (init) { fprintf(fp, " dt"); } else { fprintf(fp, "%e ", system->realdt); }
            else if (field == Prop::TIME) if (init) { fprintf(fp, " Time"); } else { fprintf(fp, "%e ", tottime); }
            else if (field == Prop::WALLTIME) if (init) { fprintf(fp, " Walltime"); } else { fprintf(fp, "%e ", timer.seconds()); }
            else if (field == Prop::EDIR) if (init) { fprintf(fp, " edir"); } else { fprintf(fp, "%e %e %e ", edir.x, edir.y, edir.z); }
            else if (field == Prop::ALLSTRESS) {
                if (init) fprintf(fp, " Sxx Syy Szz Sxy Sxz Syz"); 
                else fprintf(fp, "%e %e %e %e %e %e ", system->extstress.xx(), system->extstress.yy(), system->extstress.zz(), 
                                                       system->extstress.xy(), system->extstress.xz(), system->extstress.yz());
            }
        }
        fprintf(fp, "\n");
        fclose(fp);
    }
    
    // Output configuration
    if (istep%ctrl.outfreq == 0) {
        system->write_config(outputdir+"/config."+std::to_string(istep)+".data");
        
        // Restart files
        write_restart(outputdir+"/restart."+std::to_string(istep)+".exadis");
        
        // Timers
        if (istep > 0) {
            std::string filename;
            if (timeronefile) filename = outputdir+"/timer.dat";
            else filename = outputdir+"/timer."+std::to_string(istep)+".dat";
            FILE *fp = fopen(filename.c_str(), "a");
            if (fp == NULL)
                ExaDiS_fatal("Error: cannot open output file %s\n", filename.c_str());
            fprintf(fp, "Step = %d\n", istep);
            fprintf(fp, "\n");
            fprintf(fp, "Force time:        %12.3f sec\n", system->timer[system->TIMER_FORCE].accumtime);
            fprintf(fp, "Mobility time:     %12.3f sec\n", system->timer[system->TIMER_MOBILITY].accumtime);
            fprintf(fp, "Integration time:  %12.3f sec\n", system->timer[system->TIMER_INTEGRATION].accumtime);
            fprintf(fp, "Cross-slip time:   %12.3f sec\n", system->timer[system->TIMER_CROSSSLIP].accumtime);
            fprintf(fp, "Collision time:    %12.3f sec\n", system->timer[system->TIMER_COLLISION].accumtime);
            fprintf(fp, "Topology time:     %12.3f sec\n", system->timer[system->TIMER_TOPOLOGY].accumtime);
            fprintf(fp, "Remesh time:       %12.3f sec\n", system->timer[system->TIMER_REMESH].accumtime);
            fprintf(fp, "Output time:       %12.3f sec\n", system->timer[system->TIMER_OUTPUT].accumtime);
            fprintf(fp, "\n");
            fprintf(fp, "Total time:        %12.3f sec\n", timer.seconds());
            if (system->numdevtimer > 0) {
                fprintf(fp, "\n\n");
                fprintf(fp, "Additional (development) timers\n");
                fprintf(fp, "\n");
                for (int i = 0; i < system->numdevtimer; i++)
                    fprintf(fp, "%s time: %.3f sec\n", system->devtimer[i].label.c_str(), system->devtimer[i].accumtime);
            }
            if (timeronefile)
                fprintf(fp, "\n--------------------------------------------------\n");
            fclose(fp);
        }
    }
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::Stepper::iterate()
 *                  Controls the number of steps in a run
 *
 *-------------------------------------------------------------------------*/
bool ExaDiSApp::Stepper::iterate(ExaDiSApp* exadis)
{    
    if (exadis->init) {
        if (type == NUM_STEPS) maxsteps += exadis->istep;
        if (type == NUM_STEPS || type == MAX_STEPS) ExaDiS_log("Run for %d steps\n", maxsteps - exadis->istep);
        if (type == MAX_STRAIN) ExaDiS_log("Run until reaching strain = %f\n", stopval);
        if (type == MAX_TIME) ExaDiS_log("Run until reaching time = %f\n", stopval);
        if (type == MAX_WALLTIME) ExaDiS_log("Run until reaching wall time = %f sec\n", stopval);
        exadis->init = false;
    }
    
    exadis->istep++;
    if (exadis->system->Nnodes_total() == 0 || exadis->system->Nsegs_total() == 0) {
        ExaDiS_log("No dislocation in the system. Stopping.\n");
        return false;
    }
    if (type == NUM_STEPS || type == MAX_STEPS) return (exadis->istep <= maxsteps);
    if (type == MAX_STRAIN) return (exadis->strain < stopval);
    if (type == MAX_TIME) return (exadis->tottime < stopval);
    if (type == MAX_WALLTIME) return (exadis->timer.seconds() < stopval);
    return false;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::initialize()
 *                  Iniatialize and check that everything is setup properly
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::initialize(Control& ctrl)
{
    // Required modules
    if (system == nullptr) ExaDiS_fatal("Error: undefined system\n");
    if (force == nullptr) ExaDiS_fatal("Error: undefined force\n");
    if (mobility == nullptr) ExaDiS_fatal("Error: undefined mobility\n");
    if (integrator == nullptr) ExaDiS_fatal("Error: undefined integrator\n");
    if (collision == nullptr) ExaDiS_fatal("Error: undefined collision\n");
    if (topology == nullptr) ExaDiS_fatal("Error: undefined topology\n");
    if (remesh == nullptr) ExaDiS_fatal("Error: undefined remesh\n");
    
    system->params.check_params();
    if (!setup) set_simulation();
    
    if (!restart) {
        system->extstress = ctrl.appstress;
        edir = ctrl.edir;
    }
    
    init = true;
    
    // Initial output at step 0
    output(ctrl);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::step()
 *                  Genereric DDD single simulation step
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::step(Control& ctrl)
{
    // Do some force pre-computation for the step if needed
    force->pre_compute(system);
    
    // Nodal force calculation
    force->compute(system);
    
    // Mobility calculation
    mobility->compute(system);
    
    // Time-integration
    integrator->integrate(system);
    
    // Compute plastic strain
    system->plastic_strain();
    
    // Reset glide planes
    system->reset_glide_planes();
    
    // Cross-slip
    if (crossslip)
        crossslip->handle(system);
    
    // Collision
    collision->handle(system);
    
    // Topology
    topology->handle(system);
    
    // Remesh
    remesh->remesh(system);
    
    // Update stress
    update_mechanics(ctrl);
    
    // Output
    output(ctrl);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::run()
 *                  Generic DDD cycle to run a simulation for a number
 *                  of steps under the conditions prescribed in the 
 *                  Control argument.
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::run(Control& ctrl)
{
    // Iniatialize and check that everything is setup properly
    initialize(ctrl);
    
    // Main loop
    timer.reset();
    while (ctrl.nsteps.iterate(this)) {
        step(ctrl);
    }
    
    Kokkos::fence();
    double totaltime = timer.seconds();
    system->print_timers();
    ExaDiS_log("RUN TIME: %f sec\n", totaltime);
}

} // namespace ExaDiS
