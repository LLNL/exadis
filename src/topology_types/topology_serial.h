/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_TOPOLOGY_SERIAL_H
#define EXADIS_TOPOLOGY_SERIAL_H

#include "topology.h"
#include "force.h"
#include "mobility.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        TopologySerial
 *
 *-------------------------------------------------------------------------*/
class TopologySerial : public Topology {
private:
    Force* force;
    Mobility* mobility;
    double splitMultiNodeAlpha;
    
public:
    TopologySerial(System* system, Force* _force, Mobility* _mobility, Params params=Params()) : 
    force(_force), mobility(_mobility) {
        // To reproduce ParaDiS results
        splitMultiNodeAlpha = (system->crystal.type == BCC_CRYSTAL) ? 1e-3 : 1.0;
        if (params.splitMultiNodeAlpha >= 0.0)
            splitMultiNodeAlpha = params.splitMultiNodeAlpha;
    }
    
    void split_multi_nodes(System *system)
    {
        //printf("split_multi_nodes\n");
        //if (!network->form_junctions) return;
        
        int debug = 0;
        
        int nmultinodes = 0;
        int nsplits = 0;
        int maxdegree = 0;
        
        double rann = system->params.rann;
        double eps = 1e-12;
        double splitDist = 2.0*rann + eps;
        double shortseg = fmin(5.0, system->params.minseg * 0.1);
        double vNoise = (system->realdt > 0.0) ? system->params.rtol / system->realdt : 0.0;
        vNoise = splitMultiNodeAlpha * vNoise;
        
        int splitArmsMin = 4;
        if (system->params.split3node == 1) splitArmsMin = 3;
        
        SerialDisNet *network = system->get_serial_network();
        
        int numsets, **armsets;
        int nnodes = network->number_of_nodes();
        
        for (int i = 0; i < network->number_of_nodes(); i++) {
            int nconn = network->conn[i].num;
            
            // Unflag corner nodes that are not 2-nodes
            if (network->nodes[i].constraint == CORNER_NODE && nconn != 2)
                network->nodes[i].constraint = UNCONSTRAINED;
            
            if (nconn < splitArmsMin) continue;
            
            // need to skip node if collision just happened?
            // because force/velocity would be inacurrate...
        
            if (debug) printf("step[%d] multi node %d (nconn=%d)\n",0,i,nconn);
            
            // Three-arm nodes are special cases. Let's only handle
            // the splitting of BCC binary junction nodes whose junction
            // arm does not belong in the intersection of parent planes.
            int binaryjunc = -1;
            if (nconn == 3) {
                if (system->crystal.type != BCC_CRYSTAL) continue;
                if (i >= nnodes) continue;
                int planarjunc;
                Vec3 tjunc;
                binaryjunc = BCC_binary_junction_node(system, network, i, tjunc, &planarjunc);
                if (binaryjunc == -1) continue;
                if (planarjunc) continue;
            }
            
            DisNode node0 = network->nodes[i];
            std::vector<int> nei(nconn);
            for (int k = 0; k < nconn; k++)
                nei[k] = network->conn[i].node[k];
            
            // If any of the nodes arms is too short, skip the split
            // to avoid creating nodes with very short segments
            bool skipsplit = 0;
            Vec3 r0 = node0.pos;
            for (int k = 0; k < nconn; k++) {
                Vec3 rk = network->cell.pbc_position(r0, network->nodes[nei[k]].pos);
                if ((rk-r0).norm2() < shortseg*shortseg) {
                    skipsplit = 1;
                    continue;
                }
            }
            if (skipsplit) continue;
            
            // Save original configuration
            SerialDisNet::SaveNode saved_node = network->save_node(i);
            
            nmultinodes++;
            maxdegree = MAX(maxdegree, nconn);
            
            // Build the list of possible splits
            get_arm_sets(nconn, &numsets, &armsets);
            
            // We may not need to compute this if force/velocity
            // is accurate upon leaving the integrator/collision
            Vec3 fi = force->node_force(system, i);
            Vec3 vi = mobility->node_velocity(system, i, fi);
            
            // Power dissipation
            int kmax = -1;
            double powerMax = dot(fi, vi);
            Vec3 p0, p1;
            if (debug) printf("  fi = %e %e %e\n",fi[0],fi[1],fi[2]);
            if (debug) printf("  vi = %e %e %e\n",vi[0],vi[1],vi[2]);
            if (debug) printf("  power0 = %e\n", powerMax);
            
            for (int k = 0; k < numsets; k++) {
                
                // If we are dealing with a 3-arm node in BCC, let's
                // make sure that we are only allowing a split along
                // the junction arm. Any other split is identical to
                // a remesh operation (non-physical) and will thus
                // likely result in a higher (artificial) dissipation.
                // In this case we skip the set.
                if (system->crystal.type == BCC_CRYSTAL && nconn == 3) {
                    int splitarm = -1;
                    for (int l = 0; l < nconn; l++) {
                        if (armsets[k][l] == 0) {
                            splitarm = l;
                            break;
                        }
                    }
                    // Abort if we are trying to split along a <111> glissile arm
                    if (splitarm != binaryjunc) continue;
                }
                
                if (debug) printf("  SPLIT %d\n",k);
                
                nsplits++;
                
                std::vector<int> arms;
                for (int l = 0; l < nconn; l++) {
                    if (armsets[k][l] == 1) 
                        arms.push_back(network->find_connection(i, nei[l]));
                }
                int nnew = network->split_node(i, arms);
                
                int cnew = network->find_connection(i, nnew);
                if (cnew != -1) {
                    int snew = network->conn[i].seg[cnew];
                    Vec3 bnew = network->segs[snew].burg;
                    if (debug) printf("   bnew = %e %e %e\n",bnew[0],bnew[1],bnew[2]);
                }
                
                // Recomputed forces with new set of arms
                Vec3 f0 = force->node_force(system, i);
                Vec3 f1 = force->node_force(system, nnew);
                if (debug) printf("   f0split = %e %e %e\n",f0[0],f0[1],f0[2]);
                if (debug) printf("   f1split = %e %e %e\n",f1[0],f1[1],f1[2]);
                
                // Compute velocities
                Vec3 v0 = mobility->node_velocity(system, i, f0);
                Vec3 v1 = mobility->node_velocity(system, nnew, f1);
                
                // If we are dealing with a 3-arm node splitting, we should
                // only allow the new 3-arm node to move, the other node being
                // a new discritization node of one of the original arms.
                if (nconn == 3) {
                    if (network->conn[i].num == 3 && network->conn[nnew].num == 2) {
                        v1 = Vec3(0.0);
                    } else if (network->conn[i].num == 2 && network->conn[nnew].num == 3) {
                        v0 = Vec3(0.0);
                    }
                }
                
                double v0mag = v0.norm();
                double v1mag = v1.norm();
                if (debug) printf("   v0mag = %e, v1mag = %e\n",v0mag,v1mag);
                
                if (v0mag > eps || v1mag > eps) {
                
                    if (v0mag > eps) v0 = 1.0/v0mag*v0;
                    if (v1mag > eps) v1 = 1.0/v1mag*v1;
                    
                    //Vec3 vdiff0 = v1 - v0;
                    //network->nodes[i].pos += 0.5*splitDist*v0;
                    //network->nodes[nnew].pos += 0.5*splitDist*v1;
                    
                    int reposition0 = 0;
                    int reposition1 = 0;
                    
                    // This may create glide plane violations
                    if (!system->crystal.use_glide_planes && 
                        fabs(dot(v0, v1)/v0mag/v1mag+1.0) < 0.01) {
                        // If velocities of both nodes are nearly opposite,
                        // let's treat them as exactly opposite 
                        reposition0 = 1;
                        reposition1 = 1;
                        if (v0mag > v1mag) {
                            v1 = -1.0*v0;
                            v1mag = v0mag;
                        } else {
                            v0 = -1.0*v1;
                            v0mag = v1mag;
                        }
                    }
                    
                    Vec3 vdir;
                    if (v0mag > v1mag) {
                        reposition0 = 1;
                        vdir = -1.0*v0;
                    } else {
                        reposition1 = 1;
                        vdir = v1;
                    }
                    
                    double minSplitDist = splitDist;
                    int repositionBoth = reposition0 * reposition1;
                    if (repositionBoth) minSplitDist = 0.5 * splitDist;
                    
                    network->nodes[i].pos -= reposition0 * minSplitDist * vdir;
                    network->nodes[nnew].pos += reposition1 * minSplitDist * vdir;
                    
                    //network->nodes[i].pos = network->cell.pbc_fold(network->nodes[i].pos);
                    //network->nodes[nnew].pos = network->cell.pbc_fold(network->nodes[nnew].pos);
                    
                    // Set glide plane for new segment if needed
                    if (cnew != -1 && system->crystal.use_glide_planes) {
                        int snew = network->conn[i].seg[cnew];
                        Vec3 bnew = network->segs[snew].burg;
                        Vec3 pnew = system->crystal.find_precise_glide_plane(bnew, vdir);
                        if (pnew.norm2() < 1e-3)
                            pnew = system->crystal.pick_screw_glide_plane(network, bnew);
                        network->segs[snew].plane = pnew;
                        if (debug) printf("set new trial plane = %e %e %e\n",pnew.x,pnew.y,pnew.z);
                    }
                    
                    if (debug) printf("   repositionNode = %d %d\n",reposition0,reposition1);
                    if (debug) printf("   dir = %e %e %e\n",vdir[0],vdir[1],vdir[2]);
                    if (debug) printf("   p0split = %e %e %e\n",network->nodes[i].pos[0],network->nodes[i].pos[1],network->nodes[i].pos[2]);
                    if (debug) printf("   p1split = %e %e %e\n",network->nodes[nnew].pos[0],network->nodes[nnew].pos[1],network->nodes[nnew].pos[2]);
                    
                    // Recompute forces and velocities at new splitting positions
                    f0 = force->node_force(system, i);
                    v0 = mobility->node_velocity(system, i, f0);
                    
                    f1 = force->node_force(system, nnew);
                    v1 = mobility->node_velocity(system, nnew, f1);
                    
                    if (debug) printf("   v0split = %e %e %e\n",v0[0],v0[1],v0[2]);
                    if (debug) printf("   v1split = %e %e %e\n",v1[0],v1[1],v1[2]);
                    
                    // If we are dealing with a 3-arm node, make sure we only
                    // compute the dissipation from the new 3-arm node. The 
                    // other node is a 2-arm node that is not supposed to move.
                    if (nconn == 3) {
                        if (network->conn[i].num == 3 && network->conn[nnew].num == 2) {
                            v1 = Vec3(0.0);
                        } else if (network->conn[i].num == 2 && network->conn[nnew].num == 3) {
                            v0 = Vec3(0.0);
                        }
                    }
                    
                    if (debug) printf("   v0splitmag = %e, v1splitmag = %e\n",v0.norm(),v1.norm());
                    if (debug) printf("   f0v0 = %e, f1v1 = %e, fvnoise = %e, vNoise = %e\n",dot(f0, v0),dot(f1, v1),vNoise * (f0.norm() + f1.norm()),vNoise);
                    
                    Vec3 vdiff = v1 - v0;
                    
                    // Compute power dissipation
                    if (dot(vdiff, vdir) > 0) {
                        double powerTest = dot(f0, v0) + dot(f1, v1) - vNoise * (f0.norm() + f1.norm());
                        if (debug) printf("   powerTest[%d] = %e\n", k, powerTest);
                        
                        if (powerTest > powerMax && powerTest > 1.0) {
                            kmax = k;
                            powerMax = powerTest;
                            p0 = network->nodes[i].pos;
                            p1 = network->nodes[nnew].pos;
                            
                            //p0.print("   p0");
                            //p1.print("   p1");
                            //network->export_data("output/config.test.data");
                        }
                    }
                }
                //printf("verify_Burgers\n");
                //network->verify_Burgers();
                //output("output/config." + std::to_string(k+1) + ".ca");
                
                // Restore original configuration
                // We need to do this because otherwise the order of the nodes
                // connectivity may change, which may cause an issue when
                // identifying the junction segment for 3-node splitting
                network->free_tag(network->nodes[nnew].tag);
                network->restore_node(saved_node);
                network->nodes.pop_back();
                if (cnew != -1) network->segs.pop_back();
                network->conn.pop_back();
            }
            
            // Execute the favorable split
            if (debug) printf("  kmax = %d, powerMax = %e\n", kmax, powerMax);
            if (kmax >= 0) {
                std::vector<int> arms;
                for (int l = 0; l < nconn; l++) {
                    if (armsets[kmax][l] == 1) 
                        arms.push_back(network->find_connection(i, nei[l]));
                }
                execute_split(system, network, i, arms, p0, p1);
            }
            
            for (int k = 0; k < numsets; k++) free(armsets[k]);
            free(armsets);
        }
        
        if (debug) printf(" nmultinodes = %d, nsplits = %d, maxdegree = %d\n", nmultinodes, nsplits, maxdegree);
    }
    
    void handle(System *system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_TOPOLOGY].start();
        
        split_multi_nodes(system);
        
        Kokkos::fence();
        system->timer[system->TIMER_TOPOLOGY].stop();
    }
    
    const char* name() { return "TopologySerial"; }
};

} // namespace ExaDiS

#endif
