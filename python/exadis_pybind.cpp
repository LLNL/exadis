/*---------------------------------------------------------------------------
 *
 *	ExaDiS python binding module
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include <iostream>
#include <exadis.h>
#include <driver.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
 *    DisNet binding functions
 *
 *-------------------------------------------------------------------------*/
void SerialDisNet::set_nodes_array(std::vector<std::vector<double> >& nodes_array) {
    nodes.clear();
    for (size_t i = 0; i < nodes_array.size(); i++) {
        if (nodes_array[i].size() == 3)
            add_node(Vec3(&nodes_array[i][0]));
        else if (nodes_array[i].size() == 4)
            add_node(Vec3(&nodes_array[i][0]), (int)nodes_array[i][3]);
        else if (nodes_array[i].size() == 5)
            add_node(NodeTag((int)nodes_array[i][0], (int)nodes_array[i][1]), Vec3(&nodes_array[i][2]));
        else if (nodes_array[i].size() == 6)
            add_node(NodeTag((int)nodes_array[i][0], (int)nodes_array[i][1]), Vec3(&nodes_array[i][2]), (int)nodes_array[i][5]);
        else
            ExaDiS_fatal("Error: node must have 3 (x,y,z), 4 (x,y,z,constraint),\n"
            " 5 (dom,id,x,y,z) or 6 (dom,id,x,y,z,constraint) attributes\n");
    }
}

void SerialDisNet::set_segs_array(std::vector<std::vector<double> >& segs_array) {
    segs.clear();
    for (size_t i = 0; i < segs_array.size(); i++) {
        if (segs_array[i].size() == 5)
            add_seg((int)segs_array[i][0], (int)segs_array[i][1], Vec3(&segs_array[i][2]));
        else if (segs_array[i].size() == 8)
            add_seg((int)segs_array[i][0], (int)segs_array[i][1], Vec3(&segs_array[i][2]), Vec3(&segs_array[i][5]));
        else
            ExaDiS_fatal("Error: segment must have 5 (n1,n2,burg) or 8 (n1,n2,burg,plane) attributes\n");
    }
}

std::vector<std::vector<double> > SerialDisNet::get_nodes_array() {
    std::vector<std::vector<double> > nodes_array(number_of_nodes());
    for (int i = 0; i < number_of_nodes(); i++) {
        std::vector<double> node = {
            (double)nodes[i].tag.domain, (double)nodes[i].tag.index,
            nodes[i].pos[0], nodes[i].pos[1], nodes[i].pos[2],
            (double)nodes[i].constraint
        };
        nodes_array[i] = node;
    }
    return nodes_array;
}

std::vector<std::vector<double> > SerialDisNet::get_segs_array() {
    std::vector<std::vector<double> > segs_array(number_of_segs());
    for (int i = 0; i < number_of_segs(); i++) {
        std::vector<double> seg = {
            (double)segs[i].n1, (double)segs[i].n2,
            segs[i].burg[0], segs[i].burg[1], segs[i].burg[2],
            segs[i].plane[0], segs[i].plane[1], segs[i].plane[2]
        };
        segs_array[i] = seg;
    }
    return segs_array;
}

std::vector<int> Cell::get_pbc() {
    std::vector<int> pbc = {xpbc, ypbc, zpbc};
    return pbc;
}

std::vector<Vec3> Cell::pbc_position_array(std::vector<Vec3>& r0, std::vector<Vec3>& r) {
    if (r0.size() != r.size())
        ExaDiS_fatal("Error: reference and target position arrays must have the same size for closest_image()\n");
    std::vector<Vec3> rpbc = r;
    for (size_t i = 0; i < r.size(); i++)
        rpbc[i] = pbc_position(r0[i], r[i]);
    return rpbc;
}

std::vector<Vec3> Cell::pbc_position_array(Vec3& r0, std::vector<Vec3>& r) {
    std::vector<Vec3> rpbc = r;
    for (size_t i = 0; i < r.size(); i++)
        rpbc[i] = pbc_position(r0, r[i]);
    return rpbc;
}

std::vector<Vec3> Cell::pbc_fold_array(std::vector<Vec3>& r) {
    std::vector<Vec3> rpbc = r;
    for (size_t i = 0; i < r.size(); i++)
        rpbc[i] = pbc_fold(r[i]);
    return rpbc;
}


/*---------------------------------------------------------------------------
 *
 *    Utility functions
 *
 *-------------------------------------------------------------------------*/
void initialize(int num_threads=-1, int device_id=0) {
    Kokkos::InitializationSettings args;
    if (num_threads > 0) args.set_num_threads(num_threads);
    args.set_device_id(device_id);
    Kokkos::initialize(args);
    Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout);
}

void finalize() {
    Kokkos::finalize();
}

std::vector<int> map_node_tags(DeviceDisNet* net, std::vector<NodeTag>& tags) {
    std::vector<int> tagmap(net->Nnodes_local);
    
    // If no tags are provided, return the identity mapping
    if (tags.size() == 0) {
        for (int i = 0; i < net->Nnodes_local; i++)
            tagmap[i] = i;
        return tagmap;
    }
    
    if (tags.size() != net->Nnodes_local)
        ExaDiS_fatal("Error: tags list must have the same size as the number of nodes\n");
    
    std::map<NodeTag, int> map;
    #if EXADIS_FULL_UNIFIED_MEMORY
        for (int i = 0; i < net->Nnodes_local; i++)
            map.emplace(net->nodes(i).tag, i);
    #else
        T_nodes::HostMirror h_nodes = Kokkos::create_mirror_view(net->nodes);
        Kokkos::deep_copy(h_nodes, net->nodes);
        for (int i = 0; i < net->Nnodes_local; i++)
            map.emplace(h_nodes(i).tag, i);
    #endif
    
    for (size_t i = 0; i < tags.size(); i++) {
        auto iter = map.find(tags[i]);
        if (iter == map.end())
            ExaDiS_fatal("Error: cannot find node tag (%d,%d)\n", tags[i].domain, tags[i].index);
        tagmap[i] = iter->second;
    }
    
    return tagmap;
}

void set_positions(System* system, std::vector<Vec3>& pos) {
    auto net = system->get_device_network();
    
    if (pos.size() != net->Nnodes_local)
        ExaDiS_fatal("Error: positions list must have the same size as the number of nodes\n");
    
    #if EXADIS_FULL_UNIFIED_MEMORY
        for (int i = 0; i < net->Nnodes_local; i++)
            net->nodes(i).pos = pos[i];
    #else
        T_nodes::HostMirror h_nodes = Kokkos::create_mirror_view(net->nodes);
        Kokkos::deep_copy(h_nodes, net->nodes);
        for (int i = 0; i < net->Nnodes_local; i++)
            h_nodes(i).pos = pos[i];
        Kokkos::deep_copy(net->nodes, h_nodes);
    #endif
}

void set_forces(System* system, std::vector<Vec3>& forces, std::vector<NodeTag>& tags) {
    auto net = system->get_device_network();
    
    if (forces.size() != net->Nnodes_local)
        ExaDiS_fatal("Error: forces list must have the same size as the number of nodes\n");
    
    // Map forces to nodes using tags if needed
    std::vector<int> tagmap = map_node_tags(net, tags);
    
    #if EXADIS_FULL_UNIFIED_MEMORY
        for (int i = 0; i < net->Nnodes_local; i++)
            net->nodes(tagmap[i]).f = forces[i];
    #else
        T_nodes::HostMirror h_nodes = Kokkos::create_mirror_view(net->nodes);
        Kokkos::deep_copy(h_nodes, net->nodes);
        for (int i = 0; i < net->Nnodes_local; i++)
            h_nodes(tagmap[i]).f = forces[i];
        Kokkos::deep_copy(net->nodes, h_nodes);
    #endif
}

void set_velocities(System* system, std::vector<Vec3>& vels, std::vector<NodeTag>& tags) {
    auto net = system->get_device_network();
    
    if (vels.size() != net->Nnodes_local)
        ExaDiS_fatal("Error: velocities list must have the same size as the number of nodes\n");
    
    // Map velocities to nodes using tags if needed
    std::vector<int> tagmap = map_node_tags(net, tags);
    
    #if EXADIS_FULL_UNIFIED_MEMORY
        for (int i = 0; i < net->Nnodes_local; i++)
            net->nodes(tagmap[i]).v = vels[i];
    #else
        T_nodes::HostMirror h_nodes = Kokkos::create_mirror_view(net->nodes);
        Kokkos::deep_copy(h_nodes, net->nodes);
        for (int i = 0; i < net->Nnodes_local; i++)
            h_nodes(tagmap[i]).v = vels[i];
        Kokkos::deep_copy(net->nodes, h_nodes);
    #endif
}

std::vector<Vec3> get_forces(System* system) {
    std::vector<Vec3> forces;
    
    auto net = system->get_device_network();
    #if EXADIS_FULL_UNIFIED_MEMORY
        for (int i = 0; i < net->Nnodes_local; i++)
            forces.emplace_back(net->nodes(i).f);
    #else
        T_nodes::HostMirror h_nodes = Kokkos::create_mirror_view(net->nodes);
        Kokkos::deep_copy(h_nodes, net->nodes);
        for (int i = 0; i < net->Nnodes_local; i++)
            forces.emplace_back(h_nodes(i).f);
    #endif

    return forces;
}

std::vector<Vec3> get_velocities(System* system) {
    std::vector<Vec3> vels;
    
    auto net = system->get_device_network();
    #if EXADIS_FULL_UNIFIED_MEMORY
        for (int i = 0; i < net->Nnodes_local; i++)
            vels.emplace_back(net->nodes(i).v);
    #else
        T_nodes::HostMirror h_nodes = Kokkos::create_mirror_view(net->nodes);
        Kokkos::deep_copy(h_nodes, net->nodes);
        for (int i = 0; i < net->Nnodes_local; i++)
            vels.emplace_back(h_nodes(i).v);
    #endif

    return vels;
}

std::vector<Vec3> get_positions(System* system) {
    std::vector<Vec3> pos;
    
    auto net = system->get_device_network();
    #if EXADIS_FULL_UNIFIED_MEMORY
        for (int i = 0; i < net->Nnodes_local; i++)
            pos.emplace_back(net->nodes(i).pos);
    #else
        T_nodes::HostMirror h_nodes = Kokkos::create_mirror_view(net->nodes);
        Kokkos::deep_copy(h_nodes, net->nodes);
        for (int i = 0; i < net->Nnodes_local; i++)
            pos.emplace_back(h_nodes(i).pos);
    #endif

    return pos;
}


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
    ExaDisNet() {}
    ExaDisNet(System* _system) : system(_system) {}
    
    ExaDisNet(Cell& cell,
              std::vector<std::vector<double> >& nodes_array, 
              std::vector<std::vector<double> >& segs_array)
    {
        SerialDisNet* net = new SerialDisNet(cell);
        net->set_nodes_array(nodes_array);
        net->set_segs_array(segs_array); 
        system = make_system(net, Crystal(), Params());
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

ExaDisNet generate_prismatic_config_system(Crystal crystal, double Lbox, int numsources,
                                           double radius, double maxseg=-1, int seed=1234)
{
    SerialDisNet* config = generate_prismatic_config(crystal, Lbox, numsources, radius, maxseg, seed);
    System* system = make_system(config, Crystal(crystal), Params());
    return ExaDisNet(system);
}

ExaDisNet generate_prismatic_config_system(Crystal crystal, Cell cell, int numsources,
                                           double radius, double maxseg=-1, int seed=1234)
{
    SerialDisNet* config = generate_prismatic_config(crystal, cell, numsources, radius, maxseg, seed);
    System* system = make_system(config, Crystal(crystal), Params());
    return ExaDisNet(system);
}

ExaDisNet read_paradis_system(const char *file)
{
    SerialDisNet* config = read_paradis(file);
    System* system = make_system(config, Crystal(), Params());
    return ExaDisNet(system);
}

struct SystemBind : ExaDisNet {
    SystemBind(ExaDisNet disnet, Params params) : ExaDisNet()
    {
        SerialDisNet* net = disnet.system->get_serial_network();
        system = make_system(net, Crystal(params.crystal, params.Rorient), params);
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
 *    Parameters binding
 *
 *-------------------------------------------------------------------------*/
Params::Params(
    std::string crystalname,
    double _burgmag,
    double _MU, double _NU, double _a,
    double _maxseg, double _minseg,
    double _rann, double _rtol,
    double _maxdt, double _nextdt,
    int _split3node)
{
    set_crystal(crystalname);
    burgmag = _burgmag;
    MU = _MU;
    NU = _NU;
    a = _a;
    maxseg = _maxseg;
    minseg = _minseg;
    rann = _rann;
    rtol = _rtol;
    maxdt = _maxdt;
    nextdt = _nextdt;
    split3node = _split3node;
}

void Params::set_crystal(std::string crystalname) {
    if (!crystalname.empty()) {
        if (crystalname == "bcc" || crystalname == "BCC") crystal = BCC_CRYSTAL;
        else if (crystalname == "fcc" || crystalname == "FCC") crystal = FCC_CRYSTAL;
        else ExaDiS_fatal("Error: unknown crystal type %s in the python binding\n", crystalname.c_str());
    }
}


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
        pyforce.attr("NodeForce")(disnet);
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

ForceBind make_force_python(Params& params, py::object pyforce)
{
    params.check_params();
    Force* force = exadis_new<ForcePython>(pyforce);
    return ForceBind(force, ForceBind::PYTHON_MODEL, params);
}

template<class F>
ForceBind make_force(Params& params, typename F::Params fparams)
{
    params.check_params();
    System* system = make_system(new SerialDisNet(), Crystal(params.crystal, params.Rorient), params);
    
    Force* force = exadis_new<F>(system, fparams);
    
    int model = -1;
    if (std::is_same<F, ForceType::LINE_TENSION_MODEL>::value)
        model = ForceBind::LINE_TENSION_MODEL;
    else if (std::is_same<F, ForceType::CUTOFF_MODEL>::value)
        model = ForceBind::CUTOFF_MODEL;
    else
        ExaDiS_fatal("Error: invalid force type requested in the python binding\n");
    
    double cutoff = system->neighbor_cutoff; // needed in topology force calculations
    
    exadis_delete(system);
    
    return ForceBind(force, model, params, cutoff);
}

template<bool subcycling>
ForceBind make_force_ddd_fft(Params& params, ForceType::CORE_SELF_PKEXT::Params coreparams,
                             std::vector<int> Ngrid, Cell& cell, bool drift)
{
    params.check_params();
    System* system = make_system(new SerialDisNet(cell), Crystal(params.crystal, params.Rorient), params);
    
    Force* force;
    if (subcycling) {
        ForceType::SUBCYCLING_MODEL::Params subcycparams(Ngrid[0], Ngrid[1], Ngrid[2], drift);
        subcycparams.FSegParams = coreparams;
        force = exadis_new<ForceType::SUBCYCLING_MODEL>(system, subcycparams);
    } else {
        force = exadis_new<ForceType::DDD_FFT_MODEL>(system, 
            coreparams,
            ForceType::LONG_FFT_SHORT_ISO::Params(Ngrid[0], Ngrid[1], Ngrid[2])
        );
    }
    double cutoff = system->neighbor_cutoff; // needed in topology force calculations
    
    exadis_delete(system);
    
    int model = subcycling ? ForceBind::SUBCYCLING_MODEL : ForceBind::DDD_FFT_MODEL;
    return ForceBind(force, model, params, cutoff);
}

std::vector<Vec3> compute_force(ExaDisNet& disnet, ForceBind& forcebind, 
                                std::vector<double> applied_stress)
{
    System* system = disnet.system;
    system->params = forcebind.params;
    if (system->crystal.type != forcebind.params.crystal ||
        system->crystal.R != forcebind.params.Rorient)
        system->crystal = Crystal(forcebind.params.crystal, forcebind.params.Rorient);
    
    system->extstress = Mat33().voigt(applied_stress.data());
    
    Force* force = forcebind.force;
    force->pre_compute(system);
    forcebind.pre_computed = true;
    force->compute(system);
    
    std::vector<Vec3> forces = get_forces(system);
    
    return forces;
}

std::vector<Vec3> compute_force_n2(ExaDisNet& disnet, double MU, double NU, double a)
{
    Params params;
    params.MU = MU;
    params.NU = NU;
    params.a = a;
    
    System* system = disnet.system;
    system->params = params;
    
    Force* force = new ForceType::BRUTE_FORCE_N2(system);
    force->compute(system);
    std::vector<Vec3> forces = get_forces(system);
    delete force;
    
    return forces;
}

std::vector<Vec3> compute_force_cutoff(ExaDisNet& disnet, double MU, double NU, double a,
                                       double cutoff, double maxseg)
{
    Params params;
    params.MU = MU;
    params.NU = NU;
    params.a = a;
    params.maxseg = maxseg;
    
    System* system = disnet.system;
    system->params = params;
    
    Force* force = new ForceType::FORCE_SEGSEG_ISO(system, cutoff);
    force->pre_compute(system);
    force->compute(system);
    std::vector<Vec3> forces = get_forces(system);
    delete force;
    
    return forces;
}

std::vector<Vec3> compute_force_segseglist(ExaDisNet& disnet, double MU, double NU, double a,
                                           std::vector<std::vector<int> >& segseglist)
{
    Params params;
    params.MU = MU;
    params.NU = NU;
    params.a = a;
    
    System* system = disnet.system;
    system->params = params;
    
    ForceType::FORCE_SEGSEG_ISO* force = new ForceType::FORCE_SEGSEG_ISO(system, 0.0);
    
    SegSegList* ssl = force->get_segseglist();
    ssl->Nsegseg = (int)segseglist.size();
    Kokkos::resize(ssl->segseglist, ssl->Nsegseg);
    for (size_t i = 0; i < segseglist.size(); i++) {
        int s1 = segseglist[i][0];
        int s2 = segseglist[i][1];
        if (s1 < 0 || s1 >= system->Nnodes_total() ||
            s2 < 0 || s2 >= system->Nnodes_total())
            ExaDiS_fatal("Error: invalid segment index in compute_force_segseglist\n");
        ssl->segseglist.h_view(i) = SegSeg(s1, s2);
    }
    Kokkos::deep_copy(ssl->segseglist.d_view, ssl->segseglist.h_view);
    force->compute(system);
    std::vector<Vec3> forces = get_forces(system);
    delete force;
    
    return forces;
}

void pre_compute_force(ExaDisNet& disnet, ForceBind& forcebind)
{
    System* system = disnet.system;
    system->params = forcebind.params;
    if (system->crystal.type != forcebind.params.crystal ||
        system->crystal.R != forcebind.params.Rorient)
        system->crystal = Crystal(forcebind.params.crystal, forcebind.params.Rorient);
    
    Force* force = forcebind.force;
    force->pre_compute(system);
    forcebind.pre_computed = true;
}

Vec3 compute_node_force(ExaDisNet& disnet, int i, ForceBind& forcebind, 
                        std::vector<double> applied_stress)
{
    System* system = disnet.system;
    system->params = forcebind.params;
    if (system->crystal.type != forcebind.params.crystal ||
        system->crystal.R != forcebind.params.Rorient)
        system->crystal = Crystal(forcebind.params.crystal, forcebind.params.Rorient);
    
    system->extstress = Mat33().voigt(applied_stress.data());
    
    Force* force = forcebind.force;
    // Warning: the user must ensure the pre_compute is up-to-date...
    if (!forcebind.pre_computed) {
        force->pre_compute(system);
        forcebind.pre_computed = true;
    }
    Vec3 f = force->node_force(system, i);
    return f;
}

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
    MobilityPython(py::object _pymobility) : pymobility(_pymobility) {}
    
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

MobilityBind make_mobility_python(Params& params, py::object pymobility)
{
    params.check_params();
    Mobility* mobility = exadis_new<MobilityPython>(pymobility);
    return MobilityBind(mobility, params);
}

template<class M>
MobilityBind make_mobility(Params& params, typename M::Params mobparams)
{
    params.check_params();
    System* system = make_system(new SerialDisNet(), Crystal(params.crystal, params.Rorient), params);
    
    Mobility* mobility = new M(system, mobparams);
    
    exadis_delete(system);
    
    return MobilityBind(mobility, params);
}

std::vector<Vec3> compute_mobility(ExaDisNet& disnet, MobilityBind& mobbind,
                                   std::vector<Vec3> forces,
                                   std::vector<NodeTag> tags)
{
    System* system = disnet.system;
    system->params = mobbind.params;
    if (system->crystal.type != mobbind.params.crystal ||
        system->crystal.R != mobbind.params.Rorient)
        system->crystal = Crystal(mobbind.params.crystal, mobbind.params.Rorient);
    
    // Set forces
    set_forces(system, forces, tags);
    
    Mobility* mobility = mobbind.mobility;
    mobility->compute(system);
    std::vector<Vec3> vels = get_velocities(system);
    
    return vels;
}

Vec3 compute_node_mobility(ExaDisNet& disnet, int i, MobilityBind& mobbind, Vec3 fi)
{
    System* system = disnet.system;
    system->params = mobbind.params;
    if (system->crystal.type != mobbind.params.crystal ||
        system->crystal.R != mobbind.params.Rorient)
        system->crystal = Crystal(mobbind.params.crystal, mobbind.params.Rorient);
    
    Mobility* mobility = mobbind.mobility;
    return mobility->node_velocity(system, i, fi);
}

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

template<class I>
IntegratorBind make_integrator(Params& params, typename I::Params itgrparams,
                               ForceBind& forcebind, MobilityBind& mobbind)
{
    params.check_params();
    System* system = make_system(new SerialDisNet(), Crystal(params.crystal, params.Rorient), params);
    
    Force* force = forcebind.force;
    Mobility* mobility = mobbind.mobility;
    Integrator* integrator = new I(system, force, mobility, itgrparams);
    
    exadis_delete(system);
    
    return IntegratorBind(integrator, params);
}

double integrate(ExaDisNet& disnet, IntegratorBind& itgrbind,
                 std::vector<Vec3> vels, std::vector<double> applied_stress,
                 std::vector<NodeTag> tags)
{
    System* system = disnet.system;
    system->params = itgrbind.params;
    if (system->crystal.type != itgrbind.params.crystal ||
        system->crystal.R != itgrbind.params.Rorient)
        system->crystal = Crystal(itgrbind.params.crystal, itgrbind.params.Rorient);
    
    system->extstress = Mat33().voigt(applied_stress.data());
    
    // Set velocities
    set_velocities(system, vels, tags);
    
    Integrator* integrator = itgrbind.integrator;
    integrator->integrate(system);
    double dt = system->realdt;
    
    // Warning: we also compute plastic strain here
    system->plastic_strain();
    
    // Warning: we also reset/update glide planes here
    system->reset_glide_planes();
    
    return dt;
}

double integrate_euler(ExaDisNet& disnet, Params& _params, double dt, std::vector<Vec3> vels,
                       std::vector<NodeTag> tags)
{
    Params params = _params;
    params.nextdt = dt;
    
    System* system = disnet.system;
    system->params = params;
    
    // Set velocities
    set_velocities(system, vels, tags);
    
    Integrator* integrator = new IntegratorEuler(system);
    integrator->integrate(system);
    delete integrator;
    
    // Warning: we also compute plastic strain here
    system->plastic_strain();
    
    // Warning: we also reset/update glide planes here
    system->reset_glide_planes();
    
    return dt;
}

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

CollisionBind make_collision(Params& params)
{
    System* system = make_system(new SerialDisNet(), Crystal(params.crystal, params.Rorient), params);
    
    Collision* collision = new CollisionRetroactive(system);
    
    exadis_delete(system);
    
    return CollisionBind(collision, params);
}

void handle_collision(ExaDisNet& disnet, CollisionBind& collisionbind,
                      std::vector<Vec3> xold, double dt)
{
    System* system = disnet.system;
    system->params = collisionbind.params;
    if (system->crystal.type != collisionbind.params.crystal ||
        system->crystal.R != collisionbind.params.Rorient)
        system->crystal = Crystal(collisionbind.params.crystal, collisionbind.params.Rorient);
    
    // Make sure we have a value set for rann
    if (system->params.rann < 0.0)
        ExaDiS_fatal("Error: value of rann undefined in collision_retroactive()\n");
    
    // We need to allocate xold positions
    SerialDisNet* net = system->get_serial_network();
    Kokkos::resize(system->xold, net->number_of_nodes());
    T_x::HostMirror h_xold = Kokkos::create_mirror_view(system->xold);
    if (xold.size() == 0) {
        // Use current positions
        for (int i = 0; i < net->number_of_nodes(); i++)
            h_xold(i) = net->nodes[i].pos;
    } else {
        // Use input positions
        if (xold.size() != net->number_of_nodes())
            ExaDiS_fatal("Error: old positions list must have the same size as the number of nodes\n");
        for (int i = 0; i < net->number_of_nodes(); i++)
            h_xold(i) = xold[i];
    }
    Kokkos::deep_copy(system->xold, h_xold);
    
    // TO FIX: we also need last nodal velocities here...
    system->realdt = dt; // used for time interval
    
    Collision* collision = collisionbind.collision;
    collision->handle(system);
}

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

template<class F>
Topology* make_topology_parallel(System* system, Force* force, Mobility* mobility,
                                 MobilityBind& mobbind, typename Topology::Params& topolparams)
{
    Topology* topology;
    if (strcmp(mobbind.mobility->name(), "MobilityBCC0b") == 0) {
        topology = new TopologyParallel<F,MobilityType::BCC_0B>(system, force, mobility, topolparams);
    } else if (strcmp(mobbind.mobility->name(), "MobilityFCC0") == 0) {
        topology = new TopologyParallel<F,MobilityType::FCC_0>(system, force, mobility, topolparams);
    } else if (strcmp(mobbind.mobility->name(), "MobilityGlide") == 0) {
        topology = new TopologyParallel<F,MobilityType::GLIDE>(system, force, mobility, topolparams);
    } else {
        ExaDiS_fatal("Error: invalid mobility type = %s for TopologyParallel binding\n", mobbind.mobility->name());
    }
    return topology;
}

TopologyBind make_topology(std::string topology_mode, Params& params,
                           typename Topology::Params topolparams,
                           ForceBind& forcebind, MobilityBind& mobbind)
{
    params.check_params();
    System* system = make_system(new SerialDisNet(), Crystal(params.crystal, params.Rorient), params);
    
    Force* force = forcebind.force;
    Mobility* mobility = mobbind.mobility;
    double cutoff = forcebind.neighbor_cutoff; // required for topology force calculation
    
    Topology* topology;
    if (topology_mode == "TopologyParallel") {
        if (forcebind.model == ForceBind::LINE_TENSION_MODEL) {
            topology = make_topology_parallel<ForceType::LINE_TENSION_MODEL>(system, force, mobility, mobbind, topolparams);
        } else if (forcebind.model == ForceBind::CUTOFF_MODEL) {
            topology = make_topology_parallel<ForceType::CUTOFF_MODEL>(system, force, mobility, mobbind, topolparams);
        } else if (forcebind.model == ForceBind::DDD_FFT_MODEL) {
            topology = make_topology_parallel<ForceType::DDD_FFT_MODEL>(system, force, mobility, mobbind, topolparams);
        } else if (forcebind.model == ForceBind::SUBCYCLING_MODEL) {
            topology = make_topology_parallel<ForceType::SUBCYCLING_MODEL>(system, force, mobility, mobbind, topolparams);
        } else {
            ExaDiS_fatal("Error: invalid force type for TopologyParallel binding\n");
        }
    } else if (topology_mode == "TopologySerial") {    
        topology = new TopologySerial(system, force, mobility, topolparams);
    } else {
        ExaDiS_fatal("Error: invalid topology mode %s\n", topology_mode.c_str());
    }
    
    exadis_delete(system);
    
    return TopologyBind(topology, params, cutoff);
}

void handle_topology(ExaDisNet& disnet, TopologyBind& topolbind, double dt)
{
    System* system = disnet.system;
    system->params = topolbind.params;
    if (system->crystal.type != topolbind.params.crystal ||
        system->crystal.R != topolbind.params.Rorient)
        system->crystal = Crystal(topolbind.params.crystal, topolbind.params.Rorient);
    
    system->neighbor_cutoff = topolbind.neighbor_cutoff; // required for topology force calculation
    system->realdt = dt; // used for determining the noise level
    
    Topology* topology = topolbind.topology;
    topology->handle(system);
}

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

RemeshBind make_remesh(Params& params)
{
    System* system = make_system(new SerialDisNet(), Crystal(params.crystal, params.Rorient), params);
    
    Remesh* remesh = new RemeshSerial(system);
    
    exadis_delete(system);
    
    return RemeshBind(remesh, params);
}

void remesh(ExaDisNet& disnet, RemeshBind& remeshbind)
{
    System* system = disnet.system;
    system->params = remeshbind.params;
    if (system->crystal.type != remeshbind.params.crystal ||
        system->crystal.R != remeshbind.params.Rorient)
        system->crystal = Crystal(remeshbind.params.crystal, remeshbind.params.Rorient);
    
    Remesh* remesh = remeshbind.remesh_class;
    remesh->remesh(system);
}


/*---------------------------------------------------------------------------
 *
 *    Driver binding
 *
 *-------------------------------------------------------------------------*/
class Driver : public ExaDiSApp {
public:
    Driver(const SystemBind& sysbind) : ExaDiSApp() {
        system = sysbind.system;
    }
    void set_modules_driver(ForceBind& forcebind, MobilityBind& mobbind,
                            IntegratorBind& integratorbind, CollisionBind& collisionbind,
                            TopologyBind& topolbind, RemeshBind& remeshbind) {
        force = forcebind.force;
        mobility = mobbind.mobility;
        integrator = integratorbind.integrator;
        collision = collisionbind.collision;
        topology = topolbind.topology;
        remesh = remeshbind.remesh_class;
    }
    void read_restart_driver(std::string restartfile) {
        ExaDiSApp::read_restart(restartfile);
        // replace with dummy system so that we don't delete 
        // the original system object upon destruction
        system = make_system(new SerialDisNet());
    }
};


/*---------------------------------------------------------------------------
 *
 *    Binding module
 *
 *-------------------------------------------------------------------------*/
PYBIND11_MODULE(pyexadis, m) {
    m.doc() = "ExaDiS python module";
    
    m.attr("__version__") = EXADIS_VERSION;
    
    // Constants
    m.attr("BCC_CRYSTAL") = py::int_((int)BCC_CRYSTAL);
    m.attr("FCC_CRYSTAL") = py::int_((int)FCC_CRYSTAL);
    
    // Classes
    py::class_<Params>(m, "Params")
        .def(py::init<>())
        .def(py::init<std::string, double, double, double, double, double, double, double, double, double, double, int>(),
             py::arg("crystal")="", py::arg("burgmag"), py::arg("mu"), py::arg("nu"), py::arg("a"), 
             py::arg("maxseg"), py::arg("minseg"), py::arg("rann")=-1.0, py::arg("rtol")=-1.0, 
             py::arg("maxdt")=1e-7, py::arg("nextdt")=1e-12, py::arg("split3node")=1)
        .def("set_crystal", &Params::set_crystal, "Set the crystal type")
        .def_readonly("crystal", &Params::crystal, "Index of the crystal type")
        .def_readwrite("Rorient", &Params::Rorient, "Crystal orientation matrix")
        .def_readwrite("burgmag", &Params::burgmag, "Burgers vector magnitude (scaling length)")
        .def_readwrite("mu", &Params::MU, "Shear modulus")
        .def_readwrite("nu", &Params::NU, "Poisson's ratio")
        .def_readwrite("a", &Params::a, "Dislocation core radius")
        .def_readwrite("maxseg", &Params::maxseg, "Maximum line discretization length")
        .def_readwrite("minseg", &Params::minseg, "Minimum line discretization length")
        .def_readwrite("rann", &Params::rann, "Annihilation distance")
        .def_readwrite("rtol", &Params::rtol, "Error tolerance")
        .def_readwrite("maxdt", &Params::maxdt, "Maximum timestep size")
        .def_readwrite("nextdt", &Params::nextdt, "Starting timestep size")
        .def_readwrite("split3node", &Params::split3node, "Enable splitting of 3-nodes");
        
    py::class_<Crystal>(m, "Crystal")
        .def(py::init<>())
        .def(py::init<int>())
        .def(py::init<int, Mat33>())
        .def_readonly("R", &Crystal::R, "Crystal orientation matrix")
        .def("set_orientation", (void (Crystal::*)(Mat33)) &Crystal::set_orientation, "Set crystal orientation matrix")
        .def("set_orientation", (void (Crystal::*)(Vec3)) &Crystal::set_orientation, "Set crystal orientation via Euler angles");
        
    py::class_<Cell>(m, "Cell")
        .def(py::init<>())
        .def(py::init<double>())
        .def(py::init<const Vec3&, const Vec3&>())
        .def(py::init<const Mat33&, const Vec3&, std::vector<int> >(), py::arg("h"), py::arg("origin")=Vec3(0.0),
             py::arg("is_periodic")=std::vector<int>({PBC_BOUND,PBC_BOUND,PBC_BOUND}))
        .def_readonly("h", &Cell::H, "Cell matrix")
        .def_readonly("origin", &Cell::origin, "Origin of the cell")
        .def("center", &Cell::center, "Returns the center of the cell")
        .def("closest_image", (std::vector<Vec3> (Cell::*)(std::vector<Vec3>&, std::vector<Vec3>&)) &Cell::pbc_position_array, 
             "Returns the closest image of an array of positions from another", py::arg("Rref"), py::arg("R"))
        .def("closest_image", (std::vector<Vec3> (Cell::*)(Vec3&, std::vector<Vec3>&)) &Cell::pbc_position_array, 
             "Returns the closest image of an array of positions from a reference position", py::arg("Rref"), py::arg("R"))
        .def("closest_image", (Vec3 (Cell::*)(Vec3&, Vec3&)) &Cell::pbc_position, 
             "Returns the closest image of a position from another", py::arg("Rref"), py::arg("R"))
        .def("pbc_fold", &Cell::pbc_fold_array, "Fold an array of positions to the primary cell")
        .def("is_triclinic", &Cell::is_triclinic, "Returns if the box is triclinic")
        .def("get_pbc", &Cell::get_pbc, "Get the cell pbc flags along the 3 dimensions")
        .def("get_bounds", &Cell::get_bounds, "Get the (orthorombic) bounds of the cell")
        .def("volume", &Cell::volume, "Returns the volume of the cell");
    
    py::class_<ExaDisNet>(m, "ExaDisNet")
        .def(py::init<>())
        .def(py::init<Cell&, std::vector<std::vector<double> >&, std::vector<std::vector<double> >&>(),
             py::arg("cell"), py::arg("nodes"), py::arg("segs"))
        .def("number_of_nodes", &ExaDisNet::number_of_nodes, "Returns the number of nodes in the network")
        .def("number_of_segs", &ExaDisNet::number_of_segs, "Returns the number of segments in the network")
        .def("get_cell", &ExaDisNet::get_cell, "Get the cell containing the network")
        .def("get_nodes_array", &ExaDisNet::get_nodes_array, "Get the list of nodes (dom,id,x,y,z,constraint) of the network")
        .def("get_segs_array", &ExaDisNet::get_segs_array, "Get the list of segments (n1,n2,burg,plane) of the network")
        .def("get_forces", &ExaDisNet::get_forces, "Get the list of node forces (fx,fy,fz) of the network")
        .def("get_velocities", &ExaDisNet::get_velocities, "Get the list of node velocities (vx,vy,vz) of the network")
        .def("set_positions", &ExaDisNet::set_positions, "Set the list of node positions (x,y,z) of the network")
        .def("set_forces", &ExaDisNet::set_forces, "Set the list of node forces (fx,fy,fz) of the network")
        .def("set_velocities", &ExaDisNet::set_velocities, "Set the list of node velocities (vx,vy,vz) of the network")
        .def("write_data", &ExaDisNet::write_data, "Write network in ParaDiS format")
        .def("get_plastic_strain", &ExaDisNet::get_plastic_strain, "Returns plastic strain as computed since the last integration step");
        
    py::class_<SystemBind, ExaDisNet>(m, "System")
        .def(py::init<ExaDisNet, Params>())
        .def("set_neighbor_cutoff", &SystemBind::set_neighbor_cutoff, "Set set_neighbor cutoff of the system")
        .def("set_applied_stress", &SystemBind::set_applied_stress, "Set applied stress of the system (xx,yy,zz,yz,xz,xy)")
        .def("print_timers", &SystemBind::print_timers, "Print simulation timers", py::arg("dev")=false);
    
    py::class_<ForceType::CORE_SELF_PKEXT::Params>(m, "Force_CORE_Params")
        .def(py::init<double, double>(), py::arg("Ecore")=-1.0, py::arg("Ecore_junc_fact")=1.0);
    py::class_<ForceType::CUTOFF_MODEL::Params>(m, "Force_CUTOFF_Params")
        .def(py::init([](ForceType::CORE_SELF_PKEXT::Params coreparams, double cutoff) {
            return new ForceType::CUTOFF_MODEL::Params(coreparams, ForceType::FORCE_SEGSEG_ISO::Params(cutoff));
        }), py::arg("coreparams"), py::arg("cutoff"));
    
    py::class_<MobilityType::GLIDE::Params>(m, "Mobility_GLIDE_Params")
        .def(py::init<double>(), py::arg("Mglide"))
        .def(py::init<double, double>(), py::arg("Medge"), py::arg("Mscrew"));
    py::class_<MobilityType::BCC_0B::Params>(m, "Mobility_BCC_0B_Params")
        .def(py::init<double, double, double, double>(), py::arg("Medge"), py::arg("Mscrew"), py::arg("Mclimb"), py::arg("vmax")=-1.0);
    py::class_<MobilityType::BCC_NL::Params>(m, "Mobility_BCC_NL_Params")
        .def(py::init<double, double>(), py::arg("tempK")=300.0, py::arg("vmax")=-1.0);
    py::class_<MobilityType::FCC_0::Params>(m, "Mobility_FCC_0_Params")
        .def(py::init<double, double, double>(), py::arg("Medge"), py::arg("Mscrew"), py::arg("vmax")=-1.0);
        
    py::class_<IntegratorTrapezoid::Params>(m, "Integrator_Trapezoid_Params")
        .def(py::init<>());
    py::class_<IntegratorMulti<IntegratorTrapezoid>::Params>(m, "Integrator_Trapezoid_Multi_Params")
        .def(py::init<int>(), py::arg("multi"));
    py::class_<IntegratorRKF::Params>(m, "Integrator_RKF_Params")
        .def(py::init<>())
        .def(py::init<double, double>(), py::arg("rtolth"), py::arg("rtolrel"));
    py::class_<IntegratorMulti<IntegratorRKF>::Params>(m, "Integrator_RKF_Multi_Params")
        .def(py::init<int>(), py::arg("multi"))
        .def(py::init<IntegratorRKF::Params, int>(), py::arg("intparams"), py::arg("multi"));
    py::class_<IntegratorSubcycling::Params>(m, "Integrator_Subcycling_Params")
        .def(py::init<std::vector<double>, double, double>(), py::arg("rgroups"), py::arg("rtolth")=1.0, py::arg("rtolrel")=0.1);
        
    py::class_<Topology::Params>(m, "Topology_Params")
        .def(py::init<double>(), py::arg("splitMultiNodeAlpha"));
    
    
    // Utility
    m.def("initialize", &initialize, "Initialize the python binding module",
          py::arg("num_threads")=-1, py::arg("device_id")=0);
    m.def("finalize", &finalize, "Finalize the python binding module");
    
    // Read / Generate
    m.def("read_paradis", &read_paradis_system, "Read ParaDiS data file");
    m.def("generate_prismatic_config", (ExaDisNet (*)(Crystal, double, int, double, double, int)) &generate_prismatic_config_system,
          "Generate a configuration made of prismatic loops",
          py::arg("crystal"), py::arg("Lbox"), py::arg("numsources"), py::arg("radius"), py::arg("maxseg")=-1, py::arg("seed")=1234);
    m.def("generate_prismatic_config", (ExaDisNet (*)(Crystal, Cell, int, double, double, int)) &generate_prismatic_config_system,
          "Generate a configuration made of prismatic loops",
          py::arg("crystal"), py::arg("cell"), py::arg("numsources"), py::arg("radius"), py::arg("maxseg")=-1, py::arg("seed")=1234);
    
    // Force
    py::class_<ForceBind>(m, "Force")
        .def(py::init<>())
        .def_readwrite("neighbor_cutoff", &ForceBind::neighbor_cutoff, "Neighbor cutoff")
        .def("pre_compute", &ForceBind::pre_compute, "Pre-compute force of the system")
        .def("compute", &ForceBind::compute, "Compute force of the system");
    m.def("make_force_lt", &make_force<ForceType::LINE_TENSION_MODEL>, "Instantiate a line-tension force model",
          py::arg("params"), py::arg("coreparams"));
    m.def("make_force_cutoff", &make_force<ForceType::CUTOFF_MODEL>, "Instantiate a cutoff force model",
          py::arg("params"), py::arg("cutoffparams"));
    m.def("make_force_ddd_fft", &make_force_ddd_fft<0>, "Instantiate a DDD-FFT force model",
          py::arg("params"), py::arg("coreparams"), py::arg("Ngrid"), py::arg("cell"), py::arg("drift")=0);
    m.def("make_force_subcycling", &make_force_ddd_fft<1>, "Instantiate a subcycling force model",
          py::arg("params"), py::arg("coreparams"), py::arg("Ngrid"), py::arg("cell"), py::arg("drift")=1);
    m.def("make_force_python", &make_force_python, "Instantiate a python-based force model",
          py::arg("params"), py::arg("force"));
    
    m.def("compute_force", &compute_force, "Wrapper to compute nodal forces",
          py::arg("net"), py::arg("force"), py::arg("applied_stress"));
    m.def("pre_compute_force", &pre_compute_force, "Wrapper to perform pre-computations before compute_node_force",
          py::arg("net"), py::arg("force"));
    m.def("compute_node_force", &compute_node_force, "Wrapper to compute the force on a single node",
          py::arg("net"), py::arg("i"), py::arg("force"), py::arg("applied_stress"));
    
    m.def("compute_force_n2", &compute_force_n2, "Compute elastic forces using the brute-force N^2 calculation",
          py::arg("net"), py::arg("mu"), py::arg("nu"), py::arg("a"));
    m.def("compute_force_cutoff", &compute_force_cutoff, "Compute elastic forces using a segment pair cutoff",
          py::arg("net"), py::arg("mu"), py::arg("nu"), py::arg("a"), py::arg("cutoff"), py::arg("maxseg")=0.0);
    m.def("compute_force_segseglist", &compute_force_segseglist, "Compute elastic forces given a list of segment pairs",
          py::arg("net"), py::arg("mu"), py::arg("nu"), py::arg("a"), py::arg("segseglist"));
    
    // Mobility
    py::class_<MobilityBind>(m, "Mobility")
        .def(py::init<>())
        .def("compute", &MobilityBind::compute, "Compute mobility of the system");
    m.def("make_mobility_glide", &make_mobility<MobilityType::GLIDE>, "Instantiate a GLIDE mobility law",
          py::arg("params"), py::arg("mobparams"));
    m.def("make_mobility_bcc_0b", &make_mobility<MobilityType::BCC_0B>, "Instantiate a BCC_0B mobility law",
          py::arg("params"), py::arg("mobparams"));
    m.def("make_mobility_bcc_nl", &make_mobility<MobilityType::BCC_NL>, "Instantiate a BCC_NL mobility law",
          py::arg("params"), py::arg("mobparams"));
    m.def("make_mobility_fcc_0", &make_mobility<MobilityType::FCC_0>, "Instantiate a FCC_0 mobility law",
          py::arg("params"), py::arg("mobparams"));
    m.def("make_mobility_python", &make_mobility_python, "Instantiate a python-based mobility model",
          py::arg("params"), py::arg("mobility"));
    
    m.def("compute_mobility", &compute_mobility, "Wrapper to compute nodal velocities",
          py::arg("net"), py::arg("mobility"), py::arg("nodeforces"), py::arg("nodetags")=std::vector<NodeTag>());
    m.def("compute_node_mobility", &compute_node_mobility, "Wrapper to compute the mobility of a single node",
          py::arg("net"), py::arg("i"), py::arg("mobility"), py::arg("fi"));
    
    // Integrator
    py::class_<IntegratorBind>(m, "Integrator")
        .def(py::init<>())
        .def("integrate", &IntegratorBind::integrate, "Integrate the system");
    m.def("make_integrator_trapezoid", &make_integrator<IntegratorTrapezoid>, "Instantiate a trapezoid integrator",
          py::arg("params"), py::arg("intparams"), py::arg("force"), py::arg("mobility"));
    m.def("make_integrator_trapezoid_multi", &make_integrator<IntegratorMulti<IntegratorTrapezoid> >, "Instantiate a multi-step trapezoid integrator",
          py::arg("params"), py::arg("intparams"), py::arg("force"), py::arg("mobility"));
    m.def("make_integrator_rkf", &make_integrator<IntegratorRKF>, "Instantiate a RKF integrator",
          py::arg("params"), py::arg("intparams"), py::arg("force"), py::arg("mobility"));
    m.def("make_integrator_rkf_multi", &make_integrator<IntegratorMulti<IntegratorRKF> >, "Instantiate a multi-step RKF integrator",
          py::arg("params"), py::arg("intparams"), py::arg("force"), py::arg("mobility"));
    m.def("make_integrator_subclycing", &make_integrator<IntegratorSubcycling>, "Instantiate a subcycling integrator",
          py::arg("params"), py::arg("intparams"), py::arg("force"), py::arg("mobility"));
    m.def("integrate", &integrate, "Wrapper to perform a time-integration step",
          py::arg("net"), py::arg("integrator"), py::arg("nodevels"), py::arg("applied_stress"), py::arg("nodetags")=std::vector<NodeTag>());
    
    m.def("integrate_euler", &integrate_euler, "Time-integrate positions using the euler integrator",
          py::arg("net"), py::arg("params"), py::arg("dt"), py::arg("nodevels"), py::arg("nodetags")=std::vector<NodeTag>());
    
    // Collision
    py::class_<CollisionBind>(m, "Collision")
        .def(py::init<>())
        .def("handle", &CollisionBind::handle, "Handle collision of the system");
    m.def("make_collision", &make_collision, "Instantiate a collision class", py::arg("params"));
    m.def("handle_collision", &handle_collision, "Wrapper to handle collisions",
          py::arg("net"), py::arg("collision"), py::arg("xold")=std::vector<Vec3>(), py::arg("dt")=0.0);
    
    // Topology
    py::class_<TopologyBind>(m, "Topology")
        .def(py::init<>())
        .def("handle", &TopologyBind::handle, "Handle topology of the system");
    m.def("make_topology", &make_topology, "Instantiate a topology class",
          py::arg("topology_mode"), py::arg("params"), py::arg("topolparams"), py::arg("force"), py::arg("mobility"));
    m.def("handle_topology", &handle_topology, "Wrapper to handle topological operations",
          py::arg("net"), py::arg("topology"), py::arg("dt"));
    
    // Remesh
    py::class_<RemeshBind>(m, "Remesh")
        .def(py::init<>())
        .def("remesh", &RemeshBind::remesh, "Remesh the system");
    m.def("make_remesh", &make_remesh, "Instantiate a remesh class", py::arg("params"));
    m.def("remesh", &remesh, "Wrapper to remesh the network",
          py::arg("net"), py::arg("remesh"));
          
    
    // Driver
    py::class_<ExaDiSApp>(m, "ExaDiSApp");
    py::class_<Driver, ExaDiSApp> driver(m, "Driver");
    driver.def(py::init<const SystemBind&>(), py::arg("system"))
          .def_readwrite("outputdir", &Driver::outputdir, "Output directory path for the simulation")
          .def("read_restart", &Driver::read_restart_driver, "Read restart file")
          .def("set_modules", &Driver::set_modules_driver, "Set modules for the simulation")
          .def("set_simulation", &Driver::set_simulation, "Set things up before running the simulation", py::arg("restart")="")
          .def("initialize", &Driver::initialize, "Initialize simulation")
          .def("step", &Driver::step, "Execute a simulation step")
          .def("run", &Driver::run, "Run the simulation");
    // Driver control
    py::class_<Driver::Control>(driver, "Control")
        .def(py::init<>())
        .def_readwrite("nsteps", &Driver::Control::nsteps, "nsteps")
        .def_readwrite("loading", &Driver::Control::loading, "Loading type")
        .def_readwrite("erate", &Driver::Control::erate, "Loading rate")
        .def_readwrite("edir", &Driver::Control::edir, "Loading direction")
        .def_readwrite("appstress", &Driver::Control::appstress, "Applied stress")
        .def_readwrite("rotation", &Driver::Control::rotation, "Enable crystal rotation")
        .def_readwrite("printfreq", &Driver::Control::printfreq, "Print frequency")
        .def_readwrite("propfreq", &Driver::Control::propfreq, "Properties output frequency")
        .def_readwrite("outfreq", &Driver::Control::outfreq, "Configuration and restart output frequency")
        .def("set_props", &Driver::Control::set_props, "Set property fields for the output");
    py::enum_<Driver::Loadings>(driver, "Loadings")
        .value("STRESS_CONTROL", Driver::Loadings::STRESS_CONTROL)
        .value("STRAIN_RATE_CONTROL", Driver::Loadings::STRAIN_RATE_CONTROL)
        .export_values();
    // Driver stepper
    py::class_<Driver::Stepper> stepper(driver, "Stepper");
    stepper.def(py::init<int, int>())
           .def(py::init<int, double>())
           .def("iterate", &Driver::Stepper::iterate, "Iterate a simulation step");
    driver.def("NUM_STEPS", &Driver::NUM_STEPS, "Iterate to a number of steps")
          .def("MAX_STEPS", &Driver::MAX_STEPS, "Iterate to a maximum number of steps")
          .def("MAX_STRAIN", &Driver::MAX_STRAIN, "Iterate to a maximum strain")
          .def("MAX_TIME", &Driver::MAX_TIME, "Iterate to a maximum simulation time")
          .def("MAX_WALLTIME", &Driver::MAX_WALLTIME, "Iterate to a maximum wall clock time");
}
