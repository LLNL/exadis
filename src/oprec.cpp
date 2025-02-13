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
 *    Struct:       OpRec::Op
 *
 *-------------------------------------------------------------------------*/
OpRec::Op::Op(int _optype, char* line) : optype(_optype)
{
    int type;
    switch (optype) {
        case TIME_INTEGRATE:
            sscanf(line, "%d %d %lf",
                   &type, &i1, &d1);
            break;
        case PLASTIC_STRAIN:
            sscanf(line, "%d %lf %lf %lf %lf %lf %lf %lf",
                   &type, &d1, &v1.x, &v1.y, &v1.z, &v2.x, &v2.y, &v2.z);
            break;
        case MOVE_NODE:
        case SPLIT_SEG:
        case UPDATE_SEG_PLANE:
            sscanf(line, "%d %d %lf %lf %lf",
                   &type, &i1, &v1.x, &v1.y, &v1.z);
            break;
        case MERGE_NODES:
            sscanf(line, "%d %d %d %lf %lf %lf",
                   &type, &i1, &i2, &v1.x, &v1.y, &v1.z);
            break;
        case SPLIT_MULTI_NODE:
            sscanf(line, "%d %d %d %lf %lf %lf %lf %lf %lf",
                   &type, &i1, &i2, &v1.x, &v1.y, &v1.z, &v2.x, &v2.y, &v2.z);
            break;
        case PURGE_NETWORK:
        case UPDATE_OUTPUT:
            sscanf(line, "%d",
                   &type);
            break;
        default:
            break;
    }
}

void OpRec::Op::write(FILE* fp)
{
    switch (optype) {
        case TIME_INTEGRATE:
            fprintf(fp, "%d %d %e\n",
                    optype, i1, d1);
            break;
        case PLASTIC_STRAIN:
            fprintf(fp, "%d %e %e %e %e %e %e %e\n",
                    optype, d1, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
            break;
        case MOVE_NODE:
        case SPLIT_SEG:
        case UPDATE_SEG_PLANE:
            fprintf(fp, "%d %d %e %e %e\n",
                    optype, i1, v1.x, v1.y, v1.z);
            break;
        case MERGE_NODES:
            fprintf(fp, "%d %d %d %e %e %e\n",
                    optype, i1, i2, v1.x, v1.y, v1.z);
            break;
        case SPLIT_MULTI_NODE:
            fprintf(fp, "%d %d %d %e %e %e %e %e %e\n",
                    optype, i1, i2, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
            break;
        default:
            fprintf(fp, "%d\n",
                    optype);
            break;
    }
}

/*---------------------------------------------------------------------------
 *
 *    Struct:       OpRec
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
    
    for (size_t i = 0; i < ops.size(); i++)
        ops[i].write(fp);
    
    fclose(fp);
    
    clear();
}

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
        ops.emplace_back(optype, line);
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
    system->oprec->add_op(OpRec::TimeIntegrate(), rec_pos, system->realdt);
    
    if (rec_pos) {
        // Save new node positions
        DeviceDisNet* net = system->get_device_network();
        Kokkos::View<Vec3*> pos("pos", net->Nnodes_local);
        Kokkos::parallel_for(net->Nnodes_local, KOKKOS_LAMBDA(const int i) {
            auto nodes = net->get_nodes();
            pos(i) = nodes[i].pos;
        });
        Kokkos::fence();
        
        auto h_pos = Kokkos::create_mirror_view(pos);
        Kokkos::deep_copy(h_pos, pos);
        for (int i = 0; i < net->Nnodes_local; i++)
            system->oprec->add_op(OpRec::MoveNode(), i, h_pos(i));
    }
    
    // End time step
    system->oprec->add_op(OpRec::TimeIntegrate(), rec_pos, system->realdt);
}

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
            
            OpRec::Op* op = oprec->current();
            switch (op->optype) {
                
                case OpRec::TIME_INTEGRATE:
                {
                    // Time-integration
                    int rec_pos = op->i1;
                    system->realdt = op->d1;
                    
                    if (rec_pos) {
                        DeviceDisNet* net = system->get_device_network();
                        Kokkos::resize(system->xold, net->Nnodes_local);
                        Kokkos::parallel_for(net->Nnodes_local, KOKKOS_LAMBDA(const int i) {
                            auto nodes = net->get_nodes();
                            system->xold(i) = nodes[i].pos;
                        });
                        
                        SerialDisNet* network = system->get_serial_network();
                        op = oprec->iterate();
                        while (op->optype == OpRec::MOVE_NODE) {
                            int i = op->i1;
                            Vec3 pos = op->v1;
                            network->nodes[i].pos = pos;
                            op = oprec->iterate();
                        }
                    }
                    
                    Kokkos::fence();
                    break;
                }
                
                case OpRec::PLASTIC_STRAIN:
                {
                    // Plastic strain
                    system->density = op->d1;
                    system->dEp = Mat33().symmetric(op->v1.x, op->v1.y, op->v1.z,
                                                    op->v2.x, op->v2.y, op->v2.z);
                    op = oprec->iterate();
                    system->dWp = Mat33().symmetric(op->v1.x, op->v1.y, op->v1.z,
                                                    op->v2.x, op->v2.y, op->v2.z);                    
                    break;
                }
                
                case OpRec::MOVE_NODE:
                {
                    SerialDisNet* network = system->get_serial_network();
                    int i = op->i1;
                    Vec3 pos = op->v1;
                    network->move_node(i, pos, system->dEp);
                    break;
                }
                
                case OpRec::SPLIT_SEG:
                {
                    SerialDisNet* network = system->get_serial_network();
                    int i = op->i1;
                    Vec3 pos = op->v1;
                    network->split_seg(i, pos);
                    break;
                }
                
                case OpRec::MERGE_NODES:
                {
                    SerialDisNet* network = system->get_serial_network();
                    int n1 = op->i1;
                    int n2 = op->i2;
                    Vec3 pos = op->v1;
                    bool error = network->merge_nodes_position(n1, n2, pos, system->dEp);
                    if (!error && system->crystal.use_glide_planes)
                        system->crystal.reset_node_glide_planes(network, n1);
                    break;
                }
                
                case OpRec::SPLIT_MULTI_NODE:
                {
                    SerialDisNet* network = system->get_serial_network();
                    int i = op->i1;
                    int kmax = op->i2;
                    Vec3 p0 = op->v1;
                    Vec3 p1 = op->v2;
                    int numsets, **armsets;
                    int nconn = network->conn[i].num;
                    Topology::get_arm_sets(nconn, &numsets, &armsets);
                    std::vector<int> arms;
                    for (int l = 0; l < nconn; l++)
                        if (armsets[kmax][l] == 1) arms.push_back(l);
                    Topology::execute_split(system, network, i, kmax, arms, p0, p1);
                    for (int k = 0; k < numsets; k++) free(armsets[k]);
                    free(armsets);
                    break;
                }
                
                case OpRec::UPDATE_SEG_PLANE:
                {
                    SerialDisNet* network = system->get_serial_network();
                    int i = op->i1;
                    Vec3 plane = op->v1;
                    network->segs[i].plane = plane;
                    break;
                }
                
                case OpRec::PURGE_NETWORK:
                {
                    SerialDisNet* network = system->get_serial_network();
                    network->purge_network();
                    break;
                }
                
                case OpRec::UPDATE_OUTPUT:
                {
                    // Stepping
                    istep++;
                    
                    // Update stress
                    update_mechanics(ctrl);
                    
                    // Output
                    output(ctrl);
                    
                    break;
                }
                    
                default:
                    break;
            }
        } // end loop over op
    
    } // end loop over files
    
    Kokkos::fence();
    double totaltime = timer.seconds();
    system->print_timers();
    ExaDiS_log("REPLAY TIME: %f sec\n", totaltime);
}

} // namespace ExaDiS
