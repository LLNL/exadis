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

#if 0
/*---------------------------------------------------------------------------
 *
 *    Class:        RemeshRefine
 *
 *-------------------------------------------------------------------------*/
class RemeshRefine : public Remesh {
public:
    RemeshRefine(System *system) {}
    
    int refine_bisect(System *system)
    {
        int Nnodes_local = system->Nnodes_local;
        int Nsegs_local = system->Nsegs_local;
        double maxseg = system->params.maxseg;
        
        Kokkos::View<int*> new_idx("new_idx", 2);
        Kokkos::deep_copy(new_idx, 0);
        
        Kokkos::parallel_for(system->Nsegs_local, KOKKOS_LAMBDA(const int i) {
            
            int n1 = system->segs(i).n1;
            int n2 = system->segs(i).n2;
            Vec3 r1 = system->nodes(n1).pos;
            Vec3 r2 = system->cell.pbc_position(r1, system->nodes(n2).pos);
            double length = (r2-r1).norm();
            
            if (length > maxseg) {
                // Bisect the segment
                Vec3 rmid = 0.5*(r1+r2);
                rmid = system->cell.pbc_fold(rmid);
                
                int idx_node = Kokkos::atomic_fetch_add(&new_idx(0), 1);
                int idx_link = Kokkos::atomic_fetch_add(&new_idx(1), 1);
                
                system->segs(i).n2 = Nnodes_local+idx_node;
                system->nodes(Nnodes_local+idx_node) = DisNode(rmid);
                system->segs(Nsegs_local+idx_link) = DisSeg(Nnodes_local+idx_node, n2, system->segs(i).burg);
            }
        });
        
        Kokkos::View<int*>::HostMirror h_new_idx = Kokkos::create_mirror_view(new_idx);
        Kokkos::deep_copy(h_new_idx, new_idx);
        
        system->Nnodes_local += h_new_idx(0);
        system->Nsegs_local += h_new_idx(1);
        
        int refine = (h_new_idx(0) > 0 || h_new_idx(1) > 0);
        return refine;
    }
    
    void remesh(System *system)
    {
        Kokkos::fence();
        //system->timer.reset();
        
        if (refine_bisect(system)) {
            // do some communication here
            system->Nnodes_tot = system->Nnodes_local;
            system->Nsegs_tot = system->Nsegs_local;
            
            system->update_connectivity();
        }
        
        Kokkos::fence();
        //system->accumtime[system->TIMER_REMESH] += system->timer.seconds();
    }
    
    const char* name() { return "RemeshRefine"; }
};
#endif

/*---------------------------------------------------------------------------
 *
 *    Class:        RemeshSerial
 *
 *-------------------------------------------------------------------------*/
class RemeshSerial : public Remesh {
private:
    double maxseg;
    double minseg;
    bool do_remove_small_loops;
    
public:
    struct Params {
        bool remove_small_loops;
        Params() { remove_small_loops = 0; }
        Params(bool _remove_small_loops) : remove_small_loops(_remove_small_loops) {}
    };
    
    RemeshSerial(System* system, Params params=Params()) {
        do_remove_small_loops = params.remove_small_loops;
    }
    
    void refine_coarsen(System* system, SerialDisNet* network)
    {
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
                // Bisect the segment (refine)
                int nnew = network->split_seg(i, network->cell.pbc_fold(rmid));
                nadd++;
                //check_node_plane_violation(network, conn, nnew, "after remesh split_link");
            
            } else if (length < minseg) {
                // Merge segment nodes (coarsen)
                if (system->crystal.use_glide_planes) {
                    // Do not remesh if node arms are on different planes
                    // to avoid creating glide plane violations
                    if (network->conn[n1].num == 2) {
                        int s0 = network->conn[n1].seg[0];
                        int s1 = network->conn[n1].seg[1];
                        Vec3 p0 = network->segs[s0].plane;
                        Vec3 p1 = network->segs[s1].plane;
                        if (cross(p0, p1).norm2() > 1e-3) continue;
                        
                        network->merge_nodes_position(n2, n1, r2, system->dEp);
                        system->crystal.reset_node_glide_planes(network, n2);
                        nrem++;
                    } else if (network->conn[n2].num == 2) {
                        int s0 = network->conn[n2].seg[0];
                        int s1 = network->conn[n2].seg[1];
                        Vec3 p0 = network->segs[s0].plane;
                        Vec3 p1 = network->segs[s1].plane;
                        if (cross(p0, p1).norm2() > 1e-3) continue;
                        
                        network->merge_nodes_position(n1, n2, r1, system->dEp);
                        system->crystal.reset_node_glide_planes(network, n1);
                        nrem++;
                    } else if (length < 1.0) {
                        // Merge anyway if the segment is very small (<1b)
                        network->merge_nodes_position(n1, n2, rmid, system->dEp);
                        system->crystal.reset_node_glide_planes(network, n1);
                        nrem++;
                    }
                } else {
                    if (network->nodes[n1].constraint == CORNER_NODE ||
                        network->nodes[n2].constraint == CORNER_NODE) {
                        // Do not merge with a physical corner node unless
                        // both nodes are very close
                        if (length > 2.0*system->params.rann) continue;
                    }
                    network->merge_nodes(n1, n2, system->dEp);
                    nrem++;
                }
                //check_node_plane_violation(network, conn, n1, "after remesh merge_nodes");
    		}
        }
        
        if (nrem > 0)
            network->purge_network();
        
        //printf("refine add: %d, rem: %d\n",nadd,nrem);
    }
    
    void remove_small_loops(System* system, SerialDisNet* network)
    {
        // Parse the network into its physical links
        std::vector<std::vector<int> > links = network->physical_links();
        
        // Loop through the links and remove loops that have 3 or
        // less nodes and whose length is less than some criterion
        double minlength = fmax(fmax(0.2*minseg, 2.0*system->params.rann), 2.0*system->params.a);
        
        int nrem = 0;
        for (int i = 0; i < links.size(); i++) {
            if (links[i].size() <= 1 || links[i].size() >= 4) continue;
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
        
        maxseg = system->params.maxseg;
        minseg = system->params.minseg;
        
        SerialDisNet *local_network = system->get_serial_network();
        
        refine_coarsen(system, local_network);
        
        if (do_remove_small_loops)
            remove_small_loops(system, local_network);
        
        Kokkos::fence();
        system->timer[system->TIMER_REMESH].stop();
    }
    
    const char* name() { return "RemeshSerial"; }
};

} // namespace ExaDiS

#endif
