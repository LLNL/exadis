/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_REMESH_H
#define EXADIS_REMESH_H

#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        Remesh
 *
 *-------------------------------------------------------------------------*/
class Remesh {
public:
    Remesh() {}
    Remesh(System *system) {}
    
    virtual void remesh(System *system) {}
    virtual const char* name() { return "RemeshNone"; }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        RemeshSerial
 *
 *-------------------------------------------------------------------------*/
class RemeshSerial : public Remesh {
public:
    struct Params {
        bool remove_small_loops = 0;
        int coarsen_mode = 0;
        Params() {}
        Params(bool _remove_small_loops, int _coarsen_mode) : 
        remove_small_loops(_remove_small_loops), coarsen_mode(_coarsen_mode) {}
    };
    Params params;
    
    RemeshSerial(System* system, Params _params=Params()) {
        params = _params;
        if (params.coarsen_mode < 0 || params.coarsen_mode > 1)
            ExaDiS_fatal("Error: invalid remesh coarsen mode = %d\n", params.coarsen_mode);
    }
    
    void refine_coarsen(System* system, SerialDisNet* network)
    {
        double maxseg = system->params.maxseg;
        double minseg = system->params.minseg;
        double rann = system->params.rann;
        
        int nadd = 0;
        int nrem = 0;
        
        int nsegs = network->number_of_segs();
        for (int i = 0; i < nsegs; i++) {
            
            int n1 = network->segs[i].n1;
            int n2 = network->segs[i].n2;
            Vec3 r1 = network->nodes[n1].pos;
            Vec3 r2 = network->cell.pbc_position(r1, network->nodes[n2].pos);
            double length = (r2-r1).norm();
            Vec3 rmid = 0.5*(r1+r2);
            
            if (length > maxseg) {
                // Do not refine segments between pinned nodes
                if (network->nodes[n1].constraint == PINNED_NODE &&
                    network->nodes[n2].constraint == PINNED_NODE) continue;
                
                // Bisect the segment (refine)
                int nnew = network->split_seg(i, network->cell.pbc_fold(rmid));
                nadd++;
                //check_node_plane_violation(network, conn, nnew, "after remesh split_link");
                
            } else if (length < minseg && params.coarsen_mode == 0) {
                // Merge segment nodes (coarsen)
                if (system->crystal.enforce_glide_planes) {
                    // Do not remesh if node arms are on different planes
                    // to avoid creating glide plane violations
                    if (network->conn[n1].num == 2) {
                        int s0 = network->conn[n1].seg[0];
                        int s1 = network->conn[n1].seg[1];
                        Vec3 p0 = network->segs[s0].plane;
                        Vec3 p1 = network->segs[s1].plane;
                        if (cross(p0, p1).norm2() < 1e-3) {
                            network->merge_nodes_position(n2, n1, r2, system->dEp);
                            system->crystal.reset_node_glide_planes(network, n2);
                            nrem++;
                            continue;
                        }
                    }
                    if (network->conn[n2].num == 2) {
                        int s0 = network->conn[n2].seg[0];
                        int s1 = network->conn[n2].seg[1];
                        Vec3 p0 = network->segs[s0].plane;
                        Vec3 p1 = network->segs[s1].plane;
                        if (cross(p0, p1).norm2() < 1e-3) {
                            network->merge_nodes_position(n1, n2, r1, system->dEp);
                            system->crystal.reset_node_glide_planes(network, n1);
                            nrem++;
                            continue;
                        }
                    }
                    if (length < rann) {
                        // Merge anyway if the segment is very small
                        network->merge_nodes_position(n1, n2, rmid, system->dEp);
                        system->crystal.reset_node_glide_planes(network, n1);
                        nrem++;
                        continue;
                    }
                } else {
                    if (network->nodes[n1].constraint == CORNER_NODE ||
                        network->nodes[n2].constraint == CORNER_NODE) {
                        // Do not merge with a physical corner node unless
                        // both nodes are very close
                        if (length > 2.0*rann) continue;
                    }
                    network->merge_nodes(n1, n2, system->dEp);
                    nrem++;
                }
                //check_node_plane_violation(network, conn, n1, "after remesh merge_nodes");
            }
        }
        
        if (params.coarsen_mode == 1) {
            int nnodes = network->number_of_nodes();
            for (int i = 0; i < nnodes; i++) {
                if (network->conn[i].num != 2) continue;
                if (network->nodes[i].constraint == PINNED_NODE ||
                    network->nodes[i].constraint == CORNER_NODE) continue;
                
                Vec3 ri = network->nodes[i].pos;
                
                // neighbor 0
                int s0 = network->conn[i].seg[0];
                int n0 = network->conn[i].node[0];
                Vec3 r0 = network->cell.pbc_position(ri, network->nodes[n0].pos);
                double l0 = (r0-ri).norm();
                
                // neighbor 1
                int s1 = network->conn[i].seg[1];
                int n1 = network->conn[i].node[1];
                Vec3 r1 = network->cell.pbc_position(ri, network->nodes[n1].pos);
                double l1 = (r1-ri).norm();
                
                if (l0 > minseg && l1 > minseg) continue;
                
                if (system->crystal.enforce_glide_planes) {
                    // Check if we should allow for glide plane violation
                    bool allow_violation = 0;
                    if (l0 < rann || l1 < rann ||
                        fmax(l0, l1) < fmax(rann, 0.2*minseg) ||
                        network->find_connection(n0, n1) >= 0) {
                        allow_violation = 1;
                    }
                    if (!allow_violation) {
                        Vec3 p0 = network->segs[s0].plane;
                        Vec3 p1 = network->segs[s1].plane;
                        if (cross(p0, p1).norm2() > 1e-3) continue;
                    }
                }
                
                if (l0 < l1) {
                    // merge i into n0 at r0
                    network->merge_nodes_position(n0, i, r0, system->dEp);
                    system->crystal.reset_node_glide_planes(network, n0);
                    nrem++;
                } else {
                    // merge i into n1 at r1
                    network->merge_nodes_position(n1, i, r1, system->dEp);
                    system->crystal.reset_node_glide_planes(network, n1);
                    nrem++;
                }
            }
        }
        
        if (nrem > 0)
            network->purge_network();
        
        //printf("refine add: %d, rem: %d\n",nadd,nrem);
    }
    
    void remove_small_loops(System* system, SerialDisNet* network)
    {
        double minseg = system->params.minseg;
        double rann = system->params.rann;
        double a = system->params.a;
        
        // Parse the network into its physical links
        std::vector<std::vector<int> > links = network->physical_links();
        
        // Loop through the links and remove loops that have 4 or
        // less nodes and whose length is less than some criterion
        double minlength = fmax(fmax(1.5*minseg, 2.0*rann), 2.0*a);
        
        int nrem = 0;
        for (int i = 0; i < links.size(); i++) {
            if (links[i].size() <= 1 || links[i].size() > 4) continue;
            double length = 0.0;
            for (int j = 0; j < links[i].size(); j++)
                length += network->seg_length(links[i][j]);
            if (length > minlength) continue;
            
            // Make sure it is a loop
            int s1 = links[i][0];
            int n11 = network->segs[s1].n1;
            int n12 = network->segs[s1].n2;
            int n1 = (!network->discretization_node(n11)) ? n11 : n12;
            int s2 = links[i][links[i].size()-1];
            int n21 = network->segs[s2].n1;
            int n22 = network->segs[s2].n2;
            int n2 = (!network->discretization_node(n21)) ? n21 : n22;
                
            if (n1 == n2) {
                // Zero-out Burgers vectors to remove the loop
                for (int j = 0; j < links[i].size(); j++)
                    network->segs[links[i][j]].burg = Vec3(0.0);
                nrem++;
            }
        }
        
        if (nrem > 0)
            network->purge_network();
    }
    
    void remesh(System *system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_REMESH].start();
        
        SerialDisNet *local_network = system->get_serial_network();
        
        refine_coarsen(system, local_network);
        
        if (params.remove_small_loops)
            remove_small_loops(system, local_network);
        
        Kokkos::fence();
        system->timer[system->TIMER_REMESH].stop();
    }
    
    const char* name() { return "RemeshSerial"; }
};

} // namespace ExaDiS

#endif
