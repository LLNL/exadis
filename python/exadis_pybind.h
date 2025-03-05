/*---------------------------------------------------------------------------
 *
 *	ExaDiS python binding module
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_PYBIND_H
#define EXADIS_PYBIND_H

#include <iostream>
#include <exadis.h>
#include <driver.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace ExaDiS;

/*---------------------------------------------------------------------------
 *
 *    Struct:        Vec3
 *
 *-------------------------------------------------------------------------*/
namespace pybind11 { namespace detail {
    
    template <> struct type_caster<Vec3> {
    public:
        /*
         * This macro establishes the name 'Vec3' in
         * function signatures and declares a local variable
         * 'value' of type Vec3
         */
        PYBIND11_TYPE_CASTER(Vec3, _("Vec3"));

        /*
         * Conversion part 1 (Python->C++): convert a PyObject into a Vec3
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool) {
            if (!isinstance<sequence>(src)) return false;
            sequence seq = reinterpret_borrow<sequence>(src);
            if (seq.size() != 3)
                throw value_error("Expected sequence of length 3.");
            value[0] = seq[0].cast<double>();
            value[1] = seq[1].cast<double>();
            value[2] = seq[2].cast<double>();
            return true;
        }

        /*
         * Conversion part 2 (C++ -> Python): convert a Vec3 instance into
         * a Python object. The second and third arguments are used to
         * indicate the return value policy and parent object (for
         * ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        static handle cast(Vec3 src, return_value_policy, handle) {
            return py::make_tuple(src.x, src.y, src.z).release();
        }
    };
    
    template <> struct type_caster<Mat33> {
    public:
        PYBIND11_TYPE_CASTER(Mat33, _("Mat33"));
        bool load(handle src, bool) {
            if (!isinstance<sequence>(src)) return false;
            sequence seq = reinterpret_borrow<sequence>(src);
            if (seq.size() == 3) {
                value[0] = seq[0].cast<Vec3>();
                value[1] = seq[1].cast<Vec3>();
                value[2] = seq[2].cast<Vec3>();
            } else if (seq.size() == 9) {
                value[0] = Vec3(seq[0].cast<double>(), seq[1].cast<double>(), seq[2].cast<double>());
                value[1] = Vec3(seq[3].cast<double>(), seq[4].cast<double>(), seq[5].cast<double>());
                value[2] = Vec3(seq[6].cast<double>(), seq[7].cast<double>(), seq[8].cast<double>());
            } else {
                throw value_error("Expected sequence of length 9 or 3x3.");
            }
            return true;
        }
        static handle cast(Mat33 src, return_value_policy, handle) {
            return py::make_tuple(src.rowx, src.rowy, src.rowz).release();
        }
    };
    
    template <> struct type_caster<NodeTag> {
    public:
        PYBIND11_TYPE_CASTER(NodeTag, _("NodeTag"));
        bool load(handle src, bool) {
            if (!isinstance<sequence>(src)) return false;
            sequence seq = reinterpret_borrow<sequence>(src);
            if (seq.size() != 2)
                throw value_error("Expected sequence of length 2.");
            value.domain = seq[0].cast<int>();
            value.index  = seq[1].cast<int>();
            return true;
        }
        static handle cast(NodeTag src, return_value_policy, handle) {
            return py::make_tuple(src.domain, src.index).release();
        }
    };
    
}} // namespace pybind11::detail


/*---------------------------------------------------------------------------
 *
 *    Utility functions
 *
 *-------------------------------------------------------------------------*/
void initialize(int num_threads=-1, int device_id=0);
void finalize();
std::vector<int> map_node_tags(DeviceDisNet* net, std::vector<NodeTag>& tags);
void set_positions(System* system, std::vector<Vec3>& pos);
void set_forces(System* system, std::vector<Vec3>& forces, std::vector<NodeTag>& tags);
void set_velocities(System* system, std::vector<Vec3>& vels, std::vector<NodeTag>& tags);
std::vector<Vec3> get_forces(System* system);
std::vector<Vec3> get_velocities(System* system);


/*---------------------------------------------------------------------------
 *
 *    ExaDisNet binding
 *    Wrap the dislocation network into an ExaDiS system object.
 *    This allows to save on overhead time when driving a GPU simulation
 *    and prevents unnecessary memory copies between spaces.
 *
 *-------------------------------------------------------------------------*/
struct ExaDisNet {
    System* system = nullptr;
    
    ExaDisNet() {
        system = make_system(new SerialDisNet(), Crystal(), Params());
        system->pyexadis = true;
    }
    
    ExaDisNet(System* _system) : system(_system) {
        system->pyexadis = true;
    }
    
    ExaDisNet(Cell& cell,
              std::vector<std::vector<double> >& nodes_array, 
              std::vector<std::vector<double> >& segs_array)
    {
        SerialDisNet* net = new SerialDisNet(cell);
        net->set_nodes_array(nodes_array);
        net->set_segs_array(segs_array);
        net->sanity_check();
        system = make_system(net, Crystal(), Params());
        system->pyexadis = true;
    }
    
    void import_data(Cell& cell,
                     std::vector<std::vector<double> >& nodes_array, 
                     std::vector<std::vector<double> >& segs_array)
    {
        if (!system)
            ExaDiS_fatal("Error: cannot import data in unitialized ExaDisNet object\n");
        SerialDisNet* net = system->get_serial_network();
        net->cell = cell;
        net->set_nodes_array(nodes_array);
        net->set_segs_array(segs_array);
        net->sanity_check();
        net->generate_connectivity();
        net->update_ptr();
    }
    
    int number_of_nodes() { return system->Nnodes_total(); }
    int number_of_segs() { return system->Nsegs_total(); }
    
    Cell get_cell() { return system->get_serial_network()->cell; }
    std::vector<std::vector<double> > get_nodes_array() { return system->get_serial_network()->get_nodes_array(); }
    std::vector<std::vector<double> > get_segs_array() { return system->get_serial_network()->get_segs_array(); }
    std::vector<Vec3> get_forces() { return ::get_forces(system); }
    std::vector<Vec3> get_velocities() { return ::get_velocities(system); }
    
    void set_positions(std::vector<Vec3>& pos) { ::set_positions(system, pos); }
    void set_forces(std::vector<Vec3>& forces, std::vector<NodeTag>& tags) { ::set_forces(system, forces, tags); }
    void set_velocities(std::vector<Vec3>& vels, std::vector<NodeTag>& tags) { ::set_velocities(system, vels, tags); }
    
    void write_data(std::string filename) { system->get_serial_network()->write_data(filename); }
    
    py::tuple get_plastic_strain() { return py::make_tuple(system->dEp, system->dWp, system->density); }
};

struct SystemBind : ExaDisNet {
    SystemBind(ExaDisNet disnet, Params params) : ExaDisNet()
    {
        SerialDisNet* net = disnet.system->get_serial_network();
        system = make_system(net, Crystal(params.crystal), params);
        system->params.check_params();
    }
    void set_neighbor_cutoff(double cutoff) {
        system->neighbor_cutoff = cutoff;
    }
    void set_applied_stress(std::vector<double> applied_stress) { 
        system->extstress = Mat33().voigt(applied_stress.data()); 
    }
    void print_timers(bool dev) { system->print_timers(dev); }
};


/*---------------------------------------------------------------------------
 *
 *    Force binding
 *
 *-------------------------------------------------------------------------*/
struct ForceBind {
    enum ForceModel {
        LINE_TENSION_MODEL, CUTOFF_MODEL, DDD_FFT_MODEL, 
        SUBCYCLING_MODEL, PYTHON_MODEL
    };
    Force* force = nullptr;
    int model = -1;
    Params params;
    double neighbor_cutoff = 0.0;
    bool pre_computed = false;
    ForceBind() {}
    ForceBind(Force* _force, int _model, Params _params, double cutoff=0.0) : 
    force(_force), model(_model), params(_params), neighbor_cutoff(cutoff) {}
    void pre_compute(SystemBind& sysbind) { force->pre_compute(sysbind.system); }
    void compute(SystemBind& sysbind) { force->compute(sysbind.system); }
};

class ForcePython : public Force {
private:
    py::object pyforce;
    
public:
    ForcePython(py::object _pyforce) : pyforce(_pyforce) {}
    
    void pre_compute(System* system) {
        ExaDisNet disnet(system);
        pyforce.attr("PreCompute")(disnet);
    }
    
    void compute(System* system, bool zero=true) {
        ExaDisNet disnet(system);
        // This can only be called from within ExaDiS modules 
        // (integration, topology, etc.) so we assume that pre_compute()
        // has been called before and there is no need to call it again.
        pyforce.attr("NodeForce")(disnet, false);
    }
    
    Vec3 node_force(System* system, const int& i) {
        SerialDisNet* net = system->get_serial_network();
        NodeTag tag = net->nodes[i].tag;
        ExaDisNet disnet(system);
        return pyforce.attr("OneNodeForce")(disnet, tag).cast<Vec3>();
    }
    
    ~ForcePython() {}
    const char* name() { return "ForcePython"; }
};


/*---------------------------------------------------------------------------
 *
 *    Mobility binding
 *
 *-------------------------------------------------------------------------*/
struct MobilityBind {
    Mobility* mobility = nullptr;
    Params params;
    MobilityBind() {}
    MobilityBind(Mobility* _mobility, Params _params) : 
    mobility(_mobility), params(_params) {}
    void compute(SystemBind& sysbind) { mobility->compute(sysbind.system); }
};

class MobilityPython : public Mobility {
private:
    py::object pymobility;
    
public:
    MobilityPython(py::object _pymobility) : pymobility(_pymobility) {
        non_linear = pymobility.attr("non_linear").cast<bool>(); 
    }
    
    void compute(System* system) {
        ExaDisNet disnet(system);
        pymobility.attr("Mobility")(disnet);
    }
    
    Vec3 node_velocity(System *system, const int& i, const Vec3& fi) {
        SerialDisNet* net = system->get_serial_network();
        NodeTag tag = net->nodes[i].tag;
        ExaDisNet disnet(system);
        return pymobility.attr("OneNodeMobility")(disnet, tag, fi).cast<Vec3>();
    }
    
    ~MobilityPython() {}
    const char* name() { return "MobilityPython"; }
};


/*---------------------------------------------------------------------------
 *
 *    Integrator binding
 *
 *-------------------------------------------------------------------------*/
struct IntegratorBind {
    Integrator* integrator = nullptr;
    Params params;
    IntegratorBind() {}
    IntegratorBind(Integrator* _integrator, Params _params) : 
    integrator(_integrator), params(_params) {}
    double integrate(SystemBind& sysbind) { 
        integrator->integrate(sysbind.system);
        // We also compute plastic strain here
        sysbind.system->plastic_strain();
        // We also reset/update glide planes here
        sysbind.system->reset_glide_planes();
        return sysbind.system->realdt;
    }
};


/*---------------------------------------------------------------------------
 *
 *    Collision binding
 *
 *-------------------------------------------------------------------------*/
struct CollisionBind {
    Collision* collision = nullptr;
    Params params;
    CollisionBind() {}
    CollisionBind(Collision* _collision, Params _params) : 
    collision(_collision), params(_params) {}
    void handle(SystemBind& sysbind) { collision->handle(sysbind.system); }
};


/*---------------------------------------------------------------------------
 *
 *    Topology binding
 *
 *-------------------------------------------------------------------------*/
struct TopologyBind {
    Topology* topology = nullptr;
    Params params;
    double neighbor_cutoff = 0.0;
    TopologyBind() {}
    TopologyBind(Topology* _topology, Params _params, double cutoff) : 
    topology(_topology), params(_params), neighbor_cutoff(cutoff) {}
    void handle(SystemBind& sysbind) { topology->handle(sysbind.system); }
};

typedef typename Topology::Params TParams;

template<class F>
Topology* make_topology_parallel(System* system, Force* force, Mobility* mobility, TParams& topolparams)
{
    Topology* topology;
    if (strcmp(mobility->name(), "MobilityBCC0b") == 0) {
        topology = new TopologyParallel<F,MobilityType::BCC_0B>(system, force, mobility, topolparams);
    } else if (strcmp(mobility->name(), "MobilityFCC0") == 0) {
        topology = new TopologyParallel<F,MobilityType::FCC_0>(system, force, mobility, topolparams);
    } else if (strcmp(mobility->name(), "MobilityFCC0_fric") == 0) {
        topology = new TopologyParallel<F,MobilityType::FCC_0_FRIC>(system, force, mobility, topolparams);
    } else if (strcmp(mobility->name(), "MobilityFCC0b") == 0) {
        topology = new TopologyParallel<F,MobilityType::FCC_0B>(system, force, mobility, topolparams);
    } else if (strcmp(mobility->name(), "MobilityGlide") == 0) {
        topology = new TopologyParallel<F,MobilityType::GLIDE>(system, force, mobility, topolparams);
    } else if (strcmp(mobility->name(), "MobilityBCC_nl") == 0) {
        topology = new TopologyParallel<F,MobilityType::BCC_NL>(system, force, mobility, topolparams);
    } else {
        ExaDiS_fatal("Error: invalid mobility type = %s for TopologyParallel binding\n", mobility->name());
    }
    return topology;
}

#include "topology_parallel_types.h"


/*---------------------------------------------------------------------------
 *
 *    Remesh binding
 *
 *-------------------------------------------------------------------------*/
struct RemeshBind {
    Remesh* remesh_class = nullptr;
    Params params;
    RemeshBind() {}
    RemeshBind(Remesh* _remesh, Params _params) : 
    remesh_class(_remesh), params(_params) {}
    void remesh(SystemBind& sysbind) { remesh_class->remesh(sysbind.system); }
};


/*---------------------------------------------------------------------------
 *
 *    Cross-slip binding
 *
 *-------------------------------------------------------------------------*/
struct CrossSlipBind {
    CrossSlip* crossslip = nullptr;
    Params params;
    double neighbor_cutoff = 0.0;
    CrossSlipBind() {}
    CrossSlipBind(CrossSlip* _crossslip, Params _params, double cutoff) : 
    crossslip(_crossslip), params(_params), neighbor_cutoff(cutoff) {}
    void handle(SystemBind& sysbind) { crossslip->handle(sysbind.system); }
};


#endif
