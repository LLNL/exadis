/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  OpRec module
 *
 *  This module implements functions to record all network / topolgical
 *  operations that are performed during the course of a simulation.
 *  Using this data, a player implemented in the driver class can re-play 
 *  a simulation from previously saved OpRec files, bypassing all
 *  computations.
 *
 *  Note: The OpRec output option (from the driver) uses 3 frequencies:
 *      - oprecwritefreq: frequency at which the list of operations stored
 *        in memory is dumped to the current oprec file (oprec.XXX.exadis).
 *      - oprecfilefreq: frequency at which new OpRec files are being
 *        created to dump further operations.
 *      - oprecposfreq: frequency at which nodal motion is recorded.
 *        A value of 1 means that the exact trajectory is saved.
 *        A value > 1 means that node positions are only saved every 
 *        oprecposfreq frame.
 *        A value of 0 means that only topological operations are recorded.
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "oprec.h"
#include "driver.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Struct:       OpRec::write_file()
 *
 *-------------------------------------------------------------------------*/
void OpRec::write_file(std::string oprec_file)
{
    if (!record) return;
    
    printf("Writing oprec file\n");
    printf(" Oprec file: %s\n", oprec_file.c_str());
    
    FILE* fp = fopen(oprec_file.c_str(), "a");
    if (fp == NULL)
        ExaDiS_fatal("Error: cannot open oprec file %s\n", oprec_file.c_str());
    
    for (const auto& op : ops) {
        if (op.type() == typeid(TimeIntegrate)) {
            std::any_cast<TimeIntegrate>(op).write(fp);
        } else if (op.type() == typeid(PlasticStrain)) {
            std::any_cast<PlasticStrain>(op).write(fp);
        } else if (op.type() == typeid(MoveNode)) {
            std::any_cast<MoveNode>(op).write(fp);
        } else if (op.type() == typeid(SplitSeg)) {
            std::any_cast<SplitSeg>(op).write(fp);
        } else if (op.type() == typeid(MergeNodes)) {
            std::any_cast<MergeNodes>(op).write(fp);
        } else if (op.type() == typeid(SplitMultiNode)) {
            std::any_cast<SplitMultiNode>(op).write(fp);
        } else if (op.type() == typeid(UpdateSegPlane)) {
            std::any_cast<UpdateSegPlane>(op).write(fp);
        } else if (op.type() == typeid(PurgeNetwork)) {
            std::any_cast<PurgeNetwork>(op).write(fp);
        } else if (op.type() == typeid(UpdateOutput)) {
            std::any_cast<UpdateOutput>(op).write(fp);
        }
    }
    
    fclose(fp);
    
    clear();
}

/*---------------------------------------------------------------------------
 *
 *    Struct:       OpRec::read_file()
 *
 *-------------------------------------------------------------------------*/
void OpRec::read_file(std::string oprec_file)
{
    clear();
    
    FILE* fp = fopen(oprec_file.c_str(), "r");
    if (fp == NULL)
        ExaDiS_fatal("Error: cannot open oprec file %s\n", oprec_file.c_str());
    
    char* line = NULL;
    size_t len = 0;
    
    while (getline(&line, &len, fp) != -1) {
        int optype;
        sscanf(line, "%d", &optype);
        
        switch (optype) {
            case TIME_INTEGRATE:
                ops.push_back(TimeIntegrate(line));
                break;
            case PLASTIC_STRAIN:
                ops.push_back(PlasticStrain(line));
                break;
            case MOVE_NODE:
                ops.push_back(MoveNode(line));
                break;
            case SPLIT_SEG:
                ops.push_back(SplitSeg(line));
                break;
            case MERGE_NODES:
                ops.push_back(MergeNodes(line));
                break;
            case SPLIT_MULTI_NODE:
                ops.push_back(SplitMultiNode(line));
                break;
            case UPDATE_SEG_PLANE:
                ops.push_back(UpdateSegPlane(line));
                break;
            case PURGE_NETWORK:
                ops.push_back(PurgeNetwork(line));
                break;
            case UPDATE_OUTPUT:
                ops.push_back(UpdateOutput(line));
                break;
            default:
                break;
        }
    }
    
    fclose(fp);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::oprec_save_integration
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::oprec_save_integration(Control& ctrl)
{
    if (!system->oprec) return;
    if (!system->oprec->record) return;
    
    bool rec_pos = (ctrl.oprecposfreq > 0) ? (istep % ctrl.oprecposfreq == 0) : false;
    
    // Start time step
    system->oprec->add_op(OpRec::TimeIntegrate(rec_pos, system->realdt));
    
    if (rec_pos) {
        // Save new node positions
        DeviceDisNet* net = system->get_device_network();
        auto h_nodes = Kokkos::create_mirror_view(net->nodes);
        Kokkos::deep_copy(h_nodes, net->nodes);
        for (int i = 0; i < net->Nnodes_local; i++)
            system->oprec->add_op(OpRec::MoveNode(h_nodes(i).tag, h_nodes(i).pos));
    }
}

/*---------------------------------------------------------------------------
 *
 *    Struct:       NodeMap
 *                  Struct to track and retrieve nodes by their tag
 *
 *-------------------------------------------------------------------------*/
struct NodeMap {
    std::map<NodeTag,int> map;
    
    void reset(SerialDisNet* network) {
        map.clear();
        for (int i = 0; i < network->Nnodes_local; i++)
            map.emplace(network->nodes[i].tag, i);
    }
    
    inline void add(const NodeTag& tag, int i) {
        if (find(tag, false) != -1)
            ExaDiS_fatal("Error: node with tag = (%d,%d) already exists\n", tag.domain, tag.index);
        map.emplace(tag, i);
    }
    
    inline int find(const NodeTag& tag, bool required=true) {
        auto iter = map.find(tag);
        if (iter == map.end()) {
            if (required)
                ExaDiS_fatal("Error: cannot find node with tag = (%d,%d)\n", tag.domain, tag.index);
            return -1;
        } else {
            return iter->second;
        }
    }
    
    inline int find_seg(SerialDisNet* network, const NodeTag& tag1, const NodeTag& tag2) {
        int n1 = find(tag1);
        int s = -1;
        for (int i = 0; i < network->conn[n1].num; i++) {
            if (network->nodes[network->conn[n1].node[i]].tag == tag2) {
                s = network->conn[n1].seg[i]; break;
            }
        }
        if (s < 0)
            ExaDiS_fatal("Error: cannot find segment with tag = (%d,%d)-(%d,%d)\n",
            tag1.domain, tag1.index, tag2.domain, tag2.index);
        return s;
    }
    
    inline int find_arm(SerialDisNet* network, const NodeTag& tag1, const NodeTag& tag2) {
        int n1 = find(tag1);
        int c = -1;
        for (int i = 0; i < network->conn[n1].num; i++) {
            if (network->nodes[network->conn[n1].node[i]].tag == tag2) {
                c = i; break;
            }
        }
        if (c < 0)
            ExaDiS_fatal("Error: cannot find arm with tag = (%d,%d)-(%d,%d)\n",
            tag1.domain, tag1.index, tag2.domain, tag2.index);
        return c;
    }
    
    inline void check_size(SerialDisNet* network, std::string name) {
        if (map.size() != network->nodes.size())
            ExaDiS_fatal("Error: inconsistent node sizes %lu - %lu after %s\n",
            map.size(), network->nodes.size(), name.c_str());
    }
};

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::oprec_replay()
 *                  Function to replay a simulation from OpRec files.
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::oprec_replay(Control& ctrl, std::string oprec_file)
{
    ExaDiS_log("ExaDiS OpRec replay mode\n");
    
    // Iniatialize and check that everything is setup properly
    initialize(ctrl, false);
    
    OpRec* oprec = system->oprec;
    oprec->deactivate();
    
    NodeMap nodemap;
    nodemap.reset(system->get_serial_network());
    
    timer.reset();
    
    int num_files = 0;
    bool read_files = true;
    while (read_files) {
    
        std::string file;
        if (oprec_file.find("*") == std::string::npos) {
            file = oprec_file;
            read_files = false;
        } else {
            file = replace_string(oprec_file, "*", std::to_string(num_files));
        }
        
        FILE* fp = fopen(file.c_str(), "r");
        if (fp == NULL) {
            if (num_files == 0) {
                ExaDiS_fatal("Error: cannot open data file %s\n", file.c_str());
            } else {
                fclose(fp);
                break;
            }
        }
        fclose(fp);
        
        ExaDiS_log("Reading oprec file %s\n", file.c_str());
        oprec->read_file(file);
        num_files++;
    
        // Main loop
        while (oprec->step()) {
        
            OpRec::OpAny* op = oprec->current();
            
            if (op->type() == typeid(OpRec::TimeIntegrate))
            {
                // Time-integration
                OpRec::TimeIntegrate* opcur = std::any_cast<OpRec::TimeIntegrate>(op);
                
                system->realdt = opcur->dt;
                
                if (opcur->rec_pos) {
                    DeviceDisNet* net = system->get_device_network();
                    Kokkos::resize(system->xold, net->Nnodes_local);
                    System s = *system;
                    auto nodes = net->get_nodes();
                    Kokkos::parallel_for(net->Nnodes_local, KOKKOS_LAMBDA(const int i) {
                        s.xold(i) = nodes[i].pos;
                    });
                    Kokkos::fence();
                }
            }
            else if (op->type() == typeid(OpRec::PlasticStrain))
            {
                // Plastic strain
                OpRec::PlasticStrain* opcur = std::any_cast<OpRec::PlasticStrain>(op);
                system->density = opcur->density;
                system->dEp = opcur->dEp;
                system->dWp = opcur->dWp;
            }
            else if (op->type() == typeid(OpRec::MoveNode))
            {
                // Move node
                OpRec::MoveNode* opcur = std::any_cast<OpRec::MoveNode>(op);
                int i = nodemap.find(opcur->tag);
                Vec3 pos = opcur->pos;
                
                SerialDisNet* network = system->get_serial_network();
                network->move_node(i, pos, system->dEp);
            }
            else if (op->type() == typeid(OpRec::SplitSeg))
            {
                // Split segment
                OpRec::SplitSeg* opcur = std::any_cast<OpRec::SplitSeg>(op);
                
                SerialDisNet* network = system->get_serial_network();
                int s = nodemap.find_seg(network, opcur->tag1, opcur->tag2);
                Vec3 pos = opcur->pos;
                int nnew = network->split_seg(s, pos);
                
                NodeTag tagnew = network->nodes[nnew].tag;
                if (!(tagnew == opcur->tagnew))
                    ExaDiS_fatal("Error: inconsistent new node tag after SplitSeg\n");
                nodemap.add(tagnew, nnew);
                nodemap.check_size(network,"SPLIT_SEG");
            }
            else if (op->type() == typeid(OpRec::MergeNodes))
            {
                // Merge nodes
                OpRec::MergeNodes* opcur = std::any_cast<OpRec::MergeNodes>(op);
                int n1 = nodemap.find(opcur->tag1);
                int n2 = nodemap.find(opcur->tag2);
                Vec3 pos = opcur->pos;
                
                SerialDisNet* network = system->get_serial_network();
                bool error = network->merge_nodes_position(n1, n2, pos, system->dEp);
                if (!error && system->crystal.use_glide_planes)
                    system->crystal.reset_node_glide_planes(network, n1);
                nodemap.check_size(network,"MERGE_NODES");
            }
            else if (op->type() == typeid(OpRec::SplitMultiNode))
            {
                // Split multi node
                OpRec::SplitMultiNode* opcur = std::any_cast<OpRec::SplitMultiNode>(op);
                int i = nodemap.find(opcur->tag);
                Vec3 p0 = opcur->p0;
                Vec3 p1 = opcur->p1;
                
                SerialDisNet* network = system->get_serial_network();
                std::vector<int> arms;
                for (const auto& t : opcur->tagarms) {
                    int c = nodemap.find_arm(network, opcur->tag, t);
                    arms.push_back(c);
                }
                int inew = Topology::execute_split(system, network, i, arms, p0, p1);
                
                NodeTag tagnew = network->nodes[inew].tag;
                if (!(tagnew == opcur->tagnew))
                    ExaDiS_fatal("Error: inconsistent new node tag after SplitMultiNode\n");
                nodemap.add(tagnew, inew);
                nodemap.check_size(network,"SPLIT_MULTI_NODE");
            }
            else if (op->type() == typeid(OpRec::UpdateSegPlane))
            {
                // Udpate segment plane
                OpRec::UpdateSegPlane* opcur = std::any_cast<OpRec::UpdateSegPlane>(op);
                
                SerialDisNet* network = system->get_serial_network();
                int s = nodemap.find_seg(network, opcur->tag1, opcur->tag2);
                network->segs[s].plane = opcur->plane;
            }
            else if (op->type() == typeid(OpRec::PurgeNetwork))
            {
                // Purge network
                SerialDisNet* network = system->get_serial_network();
                network->purge_network();
                nodemap.reset(network);
                nodemap.check_size(network,"PURGE_NETWORK");
            }
            else if (op->type() == typeid(OpRec::UpdateOutput))
            {
                // Stepping
                istep++;
                // Update stress
                update_mechanics(ctrl);
                // Output
                output(ctrl);
            }
            
        } // end loop over op
    
    } // end loop over files
    
    Kokkos::fence();
    double totaltime = timer.seconds();
    system->print_timers();
    ExaDiS_log("REPLAY TIME: %f sec\n", totaltime);
}

} // namespace ExaDiS
