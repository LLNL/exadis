/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  This module implements cross-slip for FCC crystals in parallel fashion.
 *  It is the parallelization of file cross_slip_serial.cpp
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_CROSS_SLIP_PARALLEL_H
#define EXADIS_CROSS_SLIP_PARALLEL_H

#include "force.h"
#include "cross_slip_serial.h"
#include "topology_parallel.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        CrossSlipParallel
 *
 *-------------------------------------------------------------------------*/
template<class F>
class CrossSlipParallel : public CrossSlip {
private:
    F* force;
    
public:
    CrossSlipParallel(System* system, Force* _force)
    {
        // Check and assign force kernel
        force = dynamic_cast<F*>(_force);
        if (force == nullptr)
            ExaDiS_fatal("Error: inconsistent force type in CrossSlipParallel\n");
    }
    
    /*-----------------------------------------------------------------------
     *    Struct:     CrossSlipEvent
     *                Structure to hold cross-slip event information.
     *---------------------------------------------------------------------*/
    struct CrossSlipEvent {
        int type; 
        Vec3 p0, p1;
        Vec3 plane;
    };
    
    /*-----------------------------------------------------------------------
     *    Struct:     FindCrossSlipEvents
     *                Structure that implements the kernel to determine the
     *                cross-slip events that must be executed. Each node
     *                is assigned a team of threads for parallel force
     *                calculations.
     *---------------------------------------------------------------------*/
    struct FindCrossSlipEvents
    {
        System* system;
        DeviceDisNet* net;
        F* force;
        NeighborList* neilist;
        
        double eps, thetacrit, sthetacrit, s2thetacrit;
        double shearModulus, areamin;
        double noiseFactor, weightFactor;
        
        Mat33 R, Rinv;
        
        Kokkos::View<int*, T_memory_shared> count;
        Kokkos::View<int*, T_memory_space> csnodes;
        Kokkos::View<CrossSlipEvent*, T_memory_shared> events;
        
        FindCrossSlipEvents(System* _system, DeviceDisNet* _net, F* _force) :
        system(_system), net(_net), force(_force)
        {
            eps = 1e-6;
            if (system->crystal.type == FCC_CRYSTAL)
                thetacrit = 2.0 / 180.0 * M_PI;
            else if (system->crystal.type == BCC_CRYSTAL)
                thetacrit = 0.5 / 180.0 * M_PI;
            sthetacrit = sin(thetacrit);
            s2thetacrit = sthetacrit * sthetacrit;
            shearModulus = system->params.MU;
            
            areamin = 2.0 * system->params.rtol * system->params.maxseg;
            areamin = MIN(areamin, system->params.minseg * system->params.minseg * sqrt(3.0) / 4.0);
            
            noiseFactor = 1e-5;
            weightFactor = 1.0;
            
            R = system->crystal.R;
            Rinv = system->crystal.Rinv;
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const int& i) const
        {
            auto nodes = net->get_nodes();
            auto segs = net->get_segs();
            auto conn = net->get_conn();
            auto cell = net->cell;
            
            if (conn[i].num != 2) return;
            if (nodes[i].constraint != UNCONSTRAINED) return;
            
            int s = conn[i].seg[0];
            Vec3 burg = segs[s].burg;
            double burgSize = burg.norm();
            burg = burg.normalized();
            
            Vec3 burgCrystal = Rinv * burg;
            burgCrystal = burgCrystal.normalized();
            
            if (system->crystal.type == FCC_CRYSTAL) {
            
                // Only consider glide dislocations. If the Burgers vector is not
                // a [1 1 0] type, ignore it.
                if (!((fabs(fabs(burgCrystal.x)-fabs(burgCrystal.y)) < eps) &&
                      (fabs(burgCrystal.z) < eps)) &&
                    !((fabs(fabs(burgCrystal.y)-fabs(burgCrystal.z)) < eps) &&
                      (fabs(burgCrystal.x) < eps )) &&
                    !((fabs(fabs(burgCrystal.z)-fabs(burgCrystal.x)) < eps) &&
                      (fabs(burgCrystal.y) < eps))) {
                    return;
                }

                if ((fabs(burgCrystal.x) < eps) && (fabs(burgCrystal.y) < eps) &&
                    (fabs(burgCrystal.z) < eps)) {
                    return;
                }
                
                // Also test that the segment resides on a (1 1 1) plane, since these
                // are the only planes where cross-slip occurs.
                Vec3 plane = segs[s].plane.normalized();
                Vec3 planeCrystal = Rinv * plane;
                if ((fabs(fabs(planeCrystal.x) - fabs(planeCrystal.y)) > eps) ||
                    (fabs(fabs(planeCrystal.y) - fabs(planeCrystal.z)) > eps)) {
                    return; // not a {111} plane
                }
                
            } else if (system->crystal.type == BCC_CRYSTAL) {
                
                // Only consider <111> dislocations
                if (fabs(burgCrystal.x * burgCrystal.y * burgCrystal.z) < eps) {
                    return;
                }
                
            }
            
            int n1 = conn[i].node[0];
            int n2 = conn[i].node[1];
            
            Vec3 nodep = nodes[i].pos;
            Vec3 nbr1p = cell.pbc_position(nodep, nodes[n1].pos);
            Vec3 nbr2p = cell.pbc_position(nodep, nodes[n2].pos);
            
            // If the node is a point on a long screw then we can consider
            // it for possible cross slip.
            Vec3 vec1 = nbr1p - nbr2p;
            Vec3 vec2 = nodep - nbr1p;
            Vec3 vec3 = nodep - nbr2p;
            
            // Calculate some test conditions
            double test1 = dot(vec1, burg);
            double test2 = dot(vec2, burg);
            double test3 = dot(vec3, burg);
            
            test1 = test1 * test1;
            test2 = test2 * test2;
            test3 = test3 * test3;

            double testmax1 = dot(vec1, vec1);
            double testmax2 = dot(vec2, vec2);
            double testmax3 = dot(vec3, vec3);
            
            // Set up the tests to see if this dislocation is close enough to
            // screw to be considered for cross slip.  For a segment to be close
            //to screw it must be within 2*thetacrit defined above
            bool seg1_is_screw = ((testmax2 - test2) < (testmax2 * s2thetacrit));
            bool seg2_is_screw = ((testmax3 - test3) < (testmax3 * s2thetacrit));
            bool bothseg_are_screw =
                (((testmax2 - test2) < (4.0 * testmax2 * s2thetacrit)) &&
                 ((testmax3 - test3) < (4.0 * testmax3 * s2thetacrit)) &&
                 ((testmax1 - test1) < (testmax1 * s2thetacrit)));
                 
            if (seg1_is_screw || seg2_is_screw || bothseg_are_screw) {
                int idx = Kokkos::atomic_fetch_add(&count(0), 1);
                csnodes(idx) = i;
            }
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const team_handle& team) const
        {
            int tid = team.team_rank();
            int lid = team.league_rank();
            int i = csnodes(lid); // node id
            
            // Flag no event type
            if (tid == 0) events(lid).type = -1;
            
            auto nodes = net->get_nodes();
            auto segs = net->get_segs();
            auto conn = net->get_conn();
            auto cell = net->cell;
            
            // Recompute some info about the local node
            int s = conn[i].seg[0];
            Vec3 burg = segs[s].burg;
            double burgSize = burg.norm();
            burg = burg.normalized();
            
            Vec3 burgCrystal = Rinv * burg;
            burgCrystal = burgCrystal.normalized();
            Vec3 plane = segs[s].plane.normalized();
            Vec3 planeCrystal = Rinv * plane;
            
            int n1 = conn[i].node[0];
            int n2 = conn[i].node[1];
            
            Vec3 nodep = nodes[i].pos;
            Vec3 nbr1p = cell.pbc_position(nodep, nodes[n1].pos);
            Vec3 nbr2p = cell.pbc_position(nodep, nodes[n2].pos);
            
            Vec3 vec1 = nbr1p - nbr2p;
            Vec3 vec2 = nodep - nbr1p;
            Vec3 vec3 = nodep - nbr2p;
            
            double test1 = dot(vec1, burg);
            double test2 = dot(vec2, burg);
            double test3 = dot(vec3, burg);
            
            test1 = test1 * test1;
            test2 = test2 * test2;
            test3 = test3 * test3;

            double testmax1 = dot(vec1, vec1);
            double testmax2 = dot(vec2, vec2);
            double testmax3 = dot(vec3, vec3);
            
            bool seg1_is_screw = ((testmax2 - test2) < (testmax2 * s2thetacrit));
            bool seg2_is_screw = ((testmax3 - test3) < (testmax3 * s2thetacrit));
            bool bothseg_are_screw =
                (((testmax2 - test2) < (4.0 * testmax2 * s2thetacrit)) &&
                 ((testmax3 - test3) < (4.0 * testmax3 * s2thetacrit)) &&
                 ((testmax1 - test1) < (testmax1 * s2thetacrit)));
            
            // Since we will likely need to locally modify the network
            // let's first create a new configuration within a temporary, 
            // local SplitNet instance (from Topology)
            SplitDisNet splitnet(net, neilist);
            
            // Compute the nodal force (initially in laboratory frame)
            Vec3 fLab = force->node_force(system, &splitnet, i, team);
            
            // Set the force threshold for noise level within the code
            double L1 = sqrt(testmax2);
            double L2 = sqrt(testmax3);
            double fnodeThreshold = noiseFactor * shearModulus * burgSize * 
                                    0.5 * (L1 + L2);
            
            Mat33 glideDirCrystal = Mat33().zero();
            int numGlideDir = 0; // Number of cross-slip glide directions
            
            if (system->crystal.type == FCC_CRYSTAL) {
                // Find which glide planes the segments are on
                // e.g. for burg = [ 1  1  0 ], the two glide directions are
                //                 [ 1 -1  2 ] and
                //                 [ 1 -1 -2 ]
                // Use Burgers vectors in crystal frame to generate initial glide
                // planes in crystal frame.
                numGlideDir = 2;
                double tmp = 1.0;
                for (int j = 0; j < 3; j++) {
                    if (fabs(burgCrystal[j]) > eps) {
                        glideDirCrystal[0][j] = (burgCrystal[j]*tmp > 0) ? 1.0 : -1.0;
                        glideDirCrystal[1][j] = (burgCrystal[j]*tmp > 0) ? 1.0 : -1.0;
                        tmp = -1.0;
                    } else {
                        glideDirCrystal[0][j] =  2.0;
                        glideDirCrystal[1][j] = -2.0;
                    }
                }
                
                // Normalization
                glideDirCrystal[0] = sqrt(1.0/6.0) * glideDirCrystal[0];
                glideDirCrystal[1] = sqrt(1.0/6.0) * glideDirCrystal[1];
                
            } else if (system->crystal.type == BCC_CRYSTAL) {
                // Find which glide planes the segments are on. Initial
                // glidedir array contains glide directions in crystal frame
                // For BCC geometry burgCrystal should be of <1 1 1> type
                numGlideDir = 3;
                Mat33 tmp33 = outer(burgCrystal, burgCrystal);
                for (int m = 0; m < 3; m++)
                    for (int n = 0; n < 3; n++)
                        glideDirCrystal[m][n] = ((m==n)-tmp33[m][n]) * sqrt(1.5);

                // glideDirCrystal should now contain the three <112> type
                // directions that a screw dislocation may move in if glide
                // is restricted to <110> type glide planes
            }
            
            int s1 = conn[i].seg[0];
            int s2 = conn[i].seg[1];
            Vec3 segplane1 = segs[s1].plane;
            Vec3 segplane2 = segs[s2].plane;
            
            // Rotations
            Mat33 glideDirLab;
            for (int j = 0; j < 3; j++)
                glideDirLab[j] = R * glideDirCrystal[j];
            segplane1 = Rinv * segplane1;
            segplane2 = Rinv * segplane2;
            Vec3 fCrystal = Rinv * fLab;
            
            Vec3 tmp3  = glideDirCrystal * segplane1;
            Vec3 tmp3B = glideDirCrystal * segplane2;
            Vec3 tmp3C = glideDirCrystal * fCrystal;
            
            // For FCC there are only two slip planes for screw dislocation
            int plane1 = 0;
            int plane2 = 0;
            int fplane = 0;
            
            for (int j = 1; j < numGlideDir; j++) {
                plane1 = (fabs(tmp3[j])  < fabs(tmp3[plane1]) ) ? j : plane1;
                plane2 = (fabs(tmp3B[j]) < fabs(tmp3B[plane2])) ? j : plane2;
                fplane = (fabs(tmp3C[j]) > fabs(tmp3C[fplane])) ? j : fplane;
            }
            
            // Calculate the new plane in the lab frame
            Vec3 newplane = cross(burg, glideDirLab[fplane]).normalized();
            
            if (bothseg_are_screw && (plane1 == plane2) && (plane1 != fplane) &&
                (fabs(tmp3C[fplane]) > (weightFactor*fabs(tmp3C[plane1])+fnodeThreshold))) {
                
                // Both segments are close to screw and the average direction
                // is close to screw.
                
                // Determine if the neighbor nodes should be considered immobile
                bool pinned1 = node_pinned(system, net, n1, plane1, glideDirLab, numGlideDir);
                bool pinned2 = node_pinned(system, net, n2, plane2, glideDirLab, numGlideDir);
                
                if (pinned1) {
                    if ((!pinned2) || ((testmax1-test1) < (eps*eps*burgSize*burgSize))) {
                        
                        double vec1dotb = dot(vec1, burg);
                        double vec2dotb = dot(vec2, burg);
                        
                        if (!pinned2) {
                            // Neighbor 2 can be moved, so proceed with the
                            // cross-slip operation.
                            nbr2p = nbr1p - vec1dotb * burg;
                        }
                        
                        // If neighbor2 is pinned, it is already perfectly
                        // aligned with neighbor1 in the screw direction
                        // so there is no need to move it.
                        nodep = nbr1p + vec2dotb * burg;
                        
                        double fdotglide = dot(fLab, glideDirLab[fplane]);
                        double tmp = areamin / fabs(vec1dotb) * 2.0 * (1.0 + eps) * SIGN(fdotglide);
                        nodep += tmp * glideDirLab[fplane];
                        
                        // It looks like we should do the cross-slip, but to
                        // be sure, we need to move the nodes and evaluate
                        // the force on node in the new configuration. If
                        // it appears the node will not continue to move out
                        // on the new plane, skip the cross-slip event.
                        
                        // Compute force on segment by moving the node positions
                        // within the temporary, local SplitNet instance
                        // Override nodes i and n2 and set new positions
                        splitnet.nid[0] = i;
                        splitnet.nodes[0] = nodes[i];
                        splitnet.conn[0] = conn[i];
                        splitnet.nid[1] = n2;
                        splitnet.nodes[1] = nodes[n2];
                        splitnet.conn[1] = conn[n2];
                        splitnet.nconn = 2;
                        
                        splitnet.nodes[0].pos = nodep;
                        splitnet.nodes[1].pos = nbr2p;
                        
                        // Evaluate force on temporary configuration
                        Vec3 newforce = force->node_force(system, &splitnet, i, team);
                        double newfdotglide = dot(newforce, glideDirLab[fplane]);
                        
                        if ((SIGN(newfdotglide) * SIGN(fdotglide)) < 0.0) {
                            return;
                        }
                        
                        // Save the new node positions and plane
                        if (tid == 0) {
                            events(lid).type = 0;
                            events(lid).p0 = nodep;
                            events(lid).p1 = nbr2p;
                            events(lid).plane = newplane;
                        }
                    }
                } else {
                    // Neighbor 1 can be moved, so proceed with the
                    // cross-slip operation.
                    
                    double vec1dotb = dot(vec1, burg);
                    nbr1p = nbr2p + vec1dotb * burg;
                    
                    double vec3dotb = dot(vec3, burg);
                    nodep = nbr2p + vec3dotb * burg;
                    
                    double fdotglide = dot(fLab, glideDirLab[fplane]);
                    double tmp = areamin / fabs(vec1dotb) * 2.0 * (1.0 + eps) * SIGN(fdotglide);
                    nodep += tmp * glideDirLab[fplane];
                    
                    // It looks like we should do the cross-slip, but to
                    // be sure, we need to move the nodes and evaluate
                    // the force on node in the new configuration. If
                    // it appears the node will not continue to move out
                    // on the new plane, skip the cross-slip event.
                    
                    // Compute force on segment by moving the node positions
                    // within the temporary, local SplitNet instance
                    // Override nodes i and n1 and set new positions
                    splitnet.nid[0] = i;
                    splitnet.nodes[0] = nodes[i];
                    splitnet.conn[0] = conn[i];
                    splitnet.nid[1] = n1;
                    splitnet.nodes[1] = nodes[n1];
                    splitnet.conn[1] = conn[n1];
                    splitnet.nconn = 2;
                    
                    splitnet.nodes[0].pos = nodep;
                    splitnet.nodes[1].pos = nbr1p;
                    
                    Vec3 newforce = force->node_force(system, &splitnet, i, team);
                    double newfdotglide = dot(newforce, glideDirLab[fplane]);
                    
                    if ((SIGN(newfdotglide) * SIGN(fdotglide)) < 0.0) {
                        return;
                    }
                    
                    // Save the new node positions and plane
                    if (tid == 0) {
                        events(lid).type = 1;
                        events(lid).p0 = nodep;
                        events(lid).p1 = nbr1p;
                        events(lid).plane = newplane;
                    }
                }
            
            } else if (seg1_is_screw && (plane1 != plane2) && (plane2 == fplane) &&
                       (fabs(tmp3C[fplane]) > (weightFactor*fabs(tmp3C[plane1])+fnodeThreshold))) {
                
                // Zipper condition met for first segment.  If the first
                // neighbor is either not pinned or pinned but already
                // sufficiently aligned, proceed with the cross-slip event
                
                bool pinned1 = node_pinned(system, net, n1, plane1, glideDirLab, numGlideDir);
                
                if ((!pinned1) || ((testmax2-test2) < (eps*eps*burgSize*burgSize))) {
                    
                    // Before 'zippering' a segment, try a quick sanity check
                    // to see if it makes sense.  If the force on the segment
                    // to be 'zippered' is less than 5% larger on the new
                    // plane than the old plane, leave the segment alone.
                    
                    // Compute force on segment by creating a temporary new node
                    // within the temporary, local SplitNet instance
                    Vec3 pmid = 0.5 * (nodep + nbr1p);
                    int nnew = splitnet.split_seg(s1, pmid);
                    
                    Vec3 newSegForce = force->node_force(system, &splitnet, nnew, team);
                    
                    double zipperThreshold = noiseFactor * shearModulus *
                                             burgSize *  L1;
                    double f1dotplane1 = fabs(dot(newSegForce, glideDirLab[plane1]));
                    double f1dotplanef = fabs(dot(newSegForce, newplane));
                    
                    if (f1dotplanef < zipperThreshold + f1dotplane1) {
                        return;
                    }
                    
                    if (!pinned1) {
                        double vec2dotb = dot(vec2, burg);
                        nbr1p = nodep - vec2dotb * burg;
                    }
                    
                    // Save the new node position and plane
                    if (tid == 0) {
                        events(lid).type = 2;
                        events(lid).p1 = nbr1p;
                        events(lid).plane = newplane;
                    }
                }
                
            } else if (seg2_is_screw && (plane1 != plane2) && (plane1 == fplane) &&
                       (fabs(tmp3C[fplane]) > (weightFactor*fabs(tmp3C[plane2])+fnodeThreshold))) {
                
                // Zipper condition met for second segment
                
                bool pinned2 = node_pinned(system, net, n2, plane2, glideDirLab, numGlideDir);
                
                if ((!pinned2) || ((testmax2-test2) < (eps*eps*burgSize*burgSize))) {
                    
                    // Compute force on segment by creating a temporary new node
                    // within the temporary, local SplitNet instance
                    Vec3 pmid = 0.5 * (nodep + nbr2p);
                    int nnew = splitnet.split_seg(s2, pmid);
                    
                    Vec3 newSegForce = force->node_force(system, &splitnet, nnew, team);
                    
                    double zipperThreshold = noiseFactor * shearModulus *
                                             burgSize *  L2;
                    double f1dotplane2 = fabs(dot(newSegForce, glideDirLab[plane2]));
                    double f1dotplanef = fabs(dot(newSegForce, newplane));

                    if (f1dotplanef < zipperThreshold + f1dotplane2) {
                        return;
                    }
                    
                    if (!pinned2) {
                        double vec3dotb = dot(vec3, burg);
                        nbr2p = nodep - vec3dotb * burg;
                    }
                    
                    // Save the new node position and plane
                    if (tid == 0) {
                        events(lid).type = 3;
                        events(lid).p1 = nbr2p;
                        events(lid).plane = newplane;
                    }
                }
            }
        }
    };
    
    void handle(System* system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_CROSSSLIP].start();
        
        if (system->crystal.type != FCC_CRYSTAL && system->crystal.type != BCC_CRYSTAL)
            ExaDiS_fatal("Error: CrossSlipParallel only implemented for FCC and BCC crystals\n");
        
        if (!system->crystal.use_glide_planes)
            ExaDiS_fatal("Error: CrossSlipParallel requires use_glide_planes option\n");
        
        int active_net = system->net_mngr->get_active();
        DeviceDisNet* net = system->get_device_network();
        
        // Initialize the FindCrossSlipEvents structure
        FindCrossSlipEvents* cs = exadis_new<FindCrossSlipEvents>(system, net, force);
        
        // Identify nodes attached to screw segments that need
        // to be considered for a cross-slip event
        Kokkos::resize(cs->count, 1);
        Kokkos::deep_copy(cs->csnodes, 0);
        Kokkos::resize(cs->csnodes, net->Nnodes_local);
        
        Kokkos::parallel_for(net->Nnodes_local, *cs);
        Kokkos::fence();
        
        int numcsnodes = cs->count(0);
        Kokkos::resize(cs->csnodes, numcsnodes);
        
        // If we need a neighbor list, let's build a contiguous
        // one for only the subset of split nodes so that access 
        // on device will be much faster
        double cutoff = system->neighbor_cutoff;
        NeighborList* neilist;
        if (cutoff > 0.0) {
            NeighborBox* neighbox = exadis_new<NeighborBox>(system, cutoff, Neighbor::NeiSeg);
            // Build a neighbor list of the nodes wrt to the segs
            neilist = neighbox->build_neighbor_list(system, net, Neighbor::NeiNode, cs->csnodes);
            cs->neilist = neilist;
            exadis_delete(neighbox);
        }
        
        // Find all cross-slip events that we need to handle.
        // This is done in parallel where each node previously
        // identified is now assigned a team of threads.
        Kokkos::resize(cs->events, numcsnodes);
        Kokkos::parallel_for(Kokkos::TeamPolicy<>(numcsnodes, Kokkos::AUTO), *cs);
        Kokkos::fence();
        
        // We are done with determining the cross-slip events, now
        // execute the changes. We do this in serial for simplicity.
        auto h_events = Kokkos::create_mirror_view(cs->events);
        auto h_csnodes = Kokkos::create_mirror_view(cs->csnodes);
        
        // We did not make any changes to the network yet, so
        // let's avoid making unnecessary memory copies
        system->net_mngr->set_active(active_net);
        SerialDisNet* network = system->get_serial_network();
        
        std::vector<int> eventflag(numcsnodes, 0);
        // -1: skip, 0: not executed, 1: done
        for (int k = 0; k < numcsnodes; k++) {
            if (h_events(k).type < 0) eventflag[k] = -1;
        }
        
        // Start with the zipper events
        for (int k = 0; k < numcsnodes; k++) {
            if (eventflag[k] != 0) continue;
            
            CrossSlipEvent& event = h_events(k);
            int type = event.type;
            if (type < 2) continue; // zipper events = {2,3}
            
            int i = h_csnodes(k); // node id
            int n = network->conn[i].node[type-2]; // neighbor id
            int s = network->conn[i].seg[type-2]; // seg id
            Vec3 newplane = event.plane;
            
            // If the current zipper event affects a node that was
            // involved in a previous zipper event, we will only allow
            // the current event to proceeed if the glide planes
            // for the two events match.
            bool skip = 0;
            for (int l = 0; l < k; l++) {
                if (eventflag[l] != 1) continue;
                
                CrossSlipEvent& prev_event = h_events(l);
                int prev_type = prev_event.type;
                if (prev_type < 2) continue; // zipper events = {2,3}
                
                int prev_i = h_csnodes(l); // node id
                int prev_n = network->conn[prev_i].node[prev_type-2]; // neighbor id
                
                if ((i == prev_n) || (n == prev_i) || (n == prev_n)) {
                    if (cross(prev_event.plane, newplane).norm2() > 1.0e-3) {
                        skip = 1;
                        break;
                    }
                }
            }
            if (skip) continue;
            eventflag[k] = 1;
            
            // Reposition neighbor node
            network->move_node(n, event.p1, system->dEp);
            // Update the segment glide plane
            update_seg_plane(network, s, newplane);
        }
        
        // Continue with the cross-slip events
        for (int k = 0; k < numcsnodes; k++) {
            if (eventflag[k] != 0) continue;
            
            CrossSlipEvent& event = h_events(k);
            int type = event.type;
            if (type > 1) continue; // cross-slip events = {0,1}
            
            int i = h_csnodes(k); // node id
            int n = network->conn[i].node[1-type]; // neighbor id
            int n1 = network->conn[i].node[0];
            int n2 = network->conn[i].node[1];
            int s1 = network->conn[i].seg[0]; // seg 1 id
            int s2 = network->conn[i].seg[1]; // seg 2 id
            Vec3 newplane = event.plane;
            
            // If the current cross-slip event affects a node that was
            // involved in a previous cross-slip event, we will only allow
            // the current event to proceeed if the glide planes
            // for the two events match.
            bool skip = 0;
            for (int l = 0; l < k; l++) {
                if (eventflag[l] != 1) continue;
                
                CrossSlipEvent& prev_event = h_events(l);
                int prev_type = prev_event.type;
                int prev_i = h_csnodes(l); // node id
                
                if (prev_type > 1) {
                    // check against zipper events = {2,3}
                    int prev_n = network->conn[prev_i].node[prev_type-2]; // neighbor id
                    
                    if ((i == prev_n) || (n1 == prev_i) || (n1 == prev_n) || 
                        (n2 == prev_i) || (n2 == prev_n)) {
                        if (cross(prev_event.plane, newplane).norm2() > 1.0e-3) {
                            skip = 1;
                            break;
                        }
                    }
                } else {
                    // check against cross-slip events = {0,1}
                    int prev_n1 = network->conn[prev_i].node[0];
                    int prev_n2 = network->conn[prev_i].node[1];
                    
                    if ((i == prev_n1) || (i == prev_n2) ||
                        (n1 == prev_i) || (n1 == prev_n1) || (n1 == prev_n2) || 
                        (n2 == prev_i) || (n2 == prev_n1) || (n2 == prev_n2)) {
                        if (cross(prev_event.plane, newplane).norm2() > 1.0e-3) {
                            skip = 1;
                            break;
                        }
                    }
                }
            }
            if (skip) continue;
            eventflag[k] = 1;
            
            // Reposition nodes
            network->move_node(i, event.p0, system->dEp);
            network->move_node(n, event.p1, system->dEp);
            // Update segments glide plane
            update_seg_plane(network, s1, newplane);
            update_seg_plane(network, s2, newplane);
        }
        
        
        if (cutoff > 0.0)
            exadis_delete(neilist);
            
        exadis_delete(cs);
        
        Kokkos::fence();
        system->timer[system->TIMER_CROSSSLIP].stop();
    }
    
    const char* name() { return "CrossSlipParallel"; }
};

} // namespace ExaDiS

#endif
