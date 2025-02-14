/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  This module implements cross-slip for FCC crystals in serial fashion.
 *  It is a direct translation in ExaDiS of ParaDiS source file
 *  ParaDiS/src/CrossSlipFCC.cc.
 *
 *  It uses a simple force criterion to determine whether a cross-slip
 *  event should occur. If the segments connected to a node are considered 
 *  close to screw then the node is considered for a possible cross-slip
 *  operation. A test is conducted to determine which glide direction of 
 *  the screw in its possible glide planes sees the greatest projection of 
 *  force. A threshold is defined so a node's preference is to remain on 
 *  the primary (current) plane, so only if the projection of the force is 
 *  greatest on a glide plane other than the primary plane, and the force 
 *  on the cross-slip plane exceeds the threshold, is a cross-slip event 
 *  attempted.
 *
 *  There are two possibilities for cross-slip that are considered:
 *      a) both segments are on same plane (classic case)
 *      b) the segments are on two different planes with one plane
 *         being the intended cross slip plane (we call this a zipper)
 *
 *  For case a) either one or both neighboring nodes are moved into perfect 
 *  screw alignment, the segments are flipped to the cross slip plane and 
 *  the cross slip node is slightly moved into the cross slip plane to
 *  create an areal nucleus. For case b) the node on the primary plane is 
 *  moved into screw alignment with the cross slipping node, the segment 
 *  is flipped into the cross slip plane and the node is moved into the
 *  cross-slip plane. 
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "cross_slip_serial.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Function:     CrossSlipSerial::handle()
 *
 *-------------------------------------------------------------------------*/
void CrossSlipSerial::handle(System* system)
{
    Kokkos::fence();
    system->timer[system->TIMER_CROSSSLIP].start();
    
    if (system->crystal.type != FCC_CRYSTAL && system->crystal.type != BCC_CRYSTAL)
        ExaDiS_fatal("Error: CrossSlipSerial only implemented for FCC and BCC crystals\n");
    
    if (!system->crystal.use_glide_planes)
        ExaDiS_fatal("Error: CrossSlipSerial requires use_glide_planes option\n");
    
    double eps = 1e-6;
    double thetacrit = 0.0;
    if (system->crystal.type == FCC_CRYSTAL)
        thetacrit = 2.0 / 180.0 * M_PI;
    else if (system->crystal.type == BCC_CRYSTAL)
        thetacrit = 0.5 / 180.0 * M_PI;
    double sthetacrit = sin(thetacrit);
    double s2thetacrit = sthetacrit * sthetacrit;
    double shearModulus = system->params.MU;
    
    double areamin = 2.0 * system->params.rtol * system->params.maxseg;
    areamin = MIN(areamin, system->params.minseg * system->params.minseg * sqrt(3.0) / 4.0);
    
    double noiseFactor = 1e-5;
    double weightFactor = 1.0;
    
    SerialDisNet* network = system->get_serial_network();
    
    for (int i = 0; i < network->number_of_nodes(); i++) {
        if (network->conn[i].num != 2) continue;
        if (network->nodes[i].constraint != UNCONSTRAINED) continue;
        
        int s = network->conn[i].seg[0];
        int order = network->conn[i].order[0];
        Vec3 burg = order*network->segs[s].burg;
        double burgSize = burg.norm();
        burg = burg.normalized();
        
        Vec3 burgCrystal = system->crystal.Rinv * burg;
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
                continue;
            }

            if ((fabs(burgCrystal.x) < eps) && (fabs(burgCrystal.y) < eps) &&
                (fabs(burgCrystal.z) < eps)) {
                continue;
            }
            
            // Also test that the segment resides on a (1 1 1) plane, since these
            // are the only planes where cross-slip occurs.
            Vec3 plane = network->segs[s].plane.normalized();
            Vec3 planeCrystal = system->crystal.Rinv * plane;
            if ((fabs(fabs(planeCrystal.x) - fabs(planeCrystal.y)) > eps) ||
                (fabs(fabs(planeCrystal.y) - fabs(planeCrystal.z)) > eps)) {
                continue; // not a {111} plane
            }
            
        } else if (system->crystal.type == BCC_CRYSTAL) {
            
            // Only consider <111> dislocations
            if (fabs(burgCrystal.x * burgCrystal.y * burgCrystal.z) < eps) {
                continue;
            }
            
        }
        
        int n1 = network->conn[i].node[0];
        int n2 = network->conn[i].node[1];
        
        Vec3 nodep = network->nodes[i].pos;
        Vec3 nbr1p = network->cell.pbc_position(nodep, network->nodes[n1].pos);
        Vec3 nbr2p = network->cell.pbc_position(nodep, network->nodes[n2].pos);
        
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
            
            // Force vector (initially in laboratory frame)
            Vec3 fLab = force->node_force(system, i);
            
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
            
            int s1 = network->conn[i].seg[0];
            int s2 = network->conn[i].seg[1];
            Vec3 segplane1 = network->segs[s1].plane;
            Vec3 segplane2 = network->segs[s2].plane;
            
            // Rotations
            Mat33 glideDirLab;
            for (int j = 0; j < 3; j++)
                glideDirLab[j] = system->crystal.R * glideDirCrystal[j];
            segplane1 = system->crystal.Rinv * segplane1;
            segplane2 = system->crystal.Rinv * segplane2;
            Vec3 fCrystal = system->crystal.Rinv * fLab;
            
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
                bool pinned1 = node_pinned(system, network, n1, plane1, glideDirLab, numGlideDir);
                bool pinned2 = node_pinned(system, network, n2, plane2, glideDirLab, numGlideDir);
                
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
                        // the force on node in the new configuration.  If
                        // it appears the node will not continue to move out
                        // on the new plane, skip the cross-slip event and
                        // restore the old configuration.
                        Vec3 nodep0 = network->nodes[i].pos;
                        Vec3 nbr2p0 = network->nodes[n2].pos;
                        
                        network->nodes[i].pos = nodep;
                        network->nodes[n2].pos = nbr2p;
                        
                        Vec3 newforce = force->node_force(system, i);
                        double newfdotglide = dot(newforce, glideDirLab[fplane]);
                        
                        // Reset the original positions. We will need the original positions
                        // even if we accept the cross-slip operation to call move_node().
                        network->nodes[i].pos = nodep0;
                        network->nodes[n2].pos = nbr2p0;
                        
                        if ((SIGN(newfdotglide) * SIGN(fdotglide)) < 0.0) {
                            continue;
                        }
                        
                        // Execute node motion
                        network->move_node(i, nodep, system->dEp);
                        network->move_node(n2, nbr2p, system->dEp);
                        
                        // Now update the glide plane for both segments
                        update_seg_plane(network, s1, newplane);
                        update_seg_plane(network, s2, newplane);
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
                    // the force on node in the new configuration.  If
                    // it appears the node will not continue to move out
                    // on the new plane, skip the cross-slip event and
                    // restore the old configuration.
                    Vec3 nodep0 = network->nodes[i].pos;
                    Vec3 nbr1p0 = network->nodes[n1].pos;
                    
                    network->nodes[i].pos = nodep;
                    network->nodes[n1].pos = nbr1p;
                    
                    Vec3 newforce = force->node_force(system, i);
                    double newfdotglide = dot(newforce, glideDirLab[fplane]);
                    
                    // Reset the original positions. We will need the original positions
                    // even if we accept the cross-slip operation to call move_node().
                    network->nodes[i].pos = nodep0;
                    network->nodes[n1].pos = nbr1p0;
                    
                    if ((SIGN(newfdotglide) * SIGN(fdotglide)) < 0.0) {
                        continue;
                    }
                    
                    // Execute node motion
                    network->move_node(i, nodep, system->dEp);
                    network->move_node(n1, nbr1p, system->dEp);
                    
                    // Now update the glide plane for both segments
                    update_seg_plane(network, s1, newplane);
                    update_seg_plane(network, s2, newplane);
                }
            
            } else if (seg1_is_screw && (plane1 != plane2) && (plane2 == fplane) &&
                       (fabs(tmp3C[fplane]) > (weightFactor*fabs(tmp3C[plane1])+fnodeThreshold))) {
                
                // Zipper condition met for first segment.  If the first
                // neighbor is either not pinned or pinned but already
                // sufficiently aligned, proceed with the cross-slip event
                
                bool pinned1 = node_pinned(system, network, n1, plane1, glideDirLab, numGlideDir);
                
                if ((!pinned1) || ((testmax2-test2) < (eps*eps*burgSize*burgSize))) {
                    
                    // Before 'zippering' a segment, try a quick sanity check
                    // to see if it makes sense.  If the force on the segment
                    // to be 'zippered' is less than 5% larger on the new
                    // plane than the old plane, leave the segment alone.
                    
                    // Compute force on segment by creating a temporary new node
                    SerialDisNet::SaveNode saved_nodep = network->save_node(i);
                    SerialDisNet::SaveNode saved_node1 = network->save_node(n1);
                    
                    Vec3 pmid = 0.5 * (nodep + nbr1p);
                    int nnew = network->split_seg(s1, pmid);
                    Vec3 newSegForce = force->node_force(system, nnew);
                    
                    // Restore the original configuration
                    network->restore_node(saved_nodep);
                    network->restore_node(saved_node1);
                    network->free_tag(network->nodes[nnew].tag);
                    network->nodes.pop_back();
                    network->conn.pop_back();
                    network->segs.pop_back();
                    
                    double zipperThreshold = noiseFactor * shearModulus *
                                             burgSize *  L1;
                    double f1dotplane1 = fabs(dot(newSegForce, glideDirLab[plane1]));
                    double f1dotplanef = fabs(dot(newSegForce, newplane));
                    
                    if (f1dotplanef < zipperThreshold + f1dotplane1) {
                        continue;
                    }
                    
                    if (!pinned1) {
                        double vec2dotb = dot(vec2, burg);
                        nbr1p = nodep - vec2dotb * burg;
                        network->move_node(n1, nbr1p, system->dEp);
                    }
                    
                    // Update the segment glide plane
                    update_seg_plane(network, s1, newplane);
                }
                
            } else if (seg2_is_screw && (plane1 != plane2) && (plane1 == fplane) &&
                       (fabs(tmp3C[fplane]) > (weightFactor*fabs(tmp3C[plane2])+fnodeThreshold))) {
                
                // Zipper condition met for second segment
                
                bool pinned2 = node_pinned(system, network, n2, plane2, glideDirLab, numGlideDir);
                
                if ((!pinned2) || ((testmax2-test2) < (eps*eps*burgSize*burgSize))) {
                    
                    // Compute force on segment by creating a temporary new node
                    SerialDisNet::SaveNode saved_nodep = network->save_node(i);
                    SerialDisNet::SaveNode saved_node2 = network->save_node(n2);
                    
                    Vec3 pmid = 0.5 * (nodep + nbr2p);
                    int nnew = network->split_seg(s2, pmid);
                    Vec3 newSegForce = force->node_force(system, nnew);
                    
                    // Restore the original configuration
                    network->restore_node(saved_nodep);
                    network->restore_node(saved_node2);
                    network->free_tag(network->nodes[nnew].tag);
                    network->nodes.pop_back();
                    network->conn.pop_back();
                    network->segs.pop_back();
                    
                    double zipperThreshold = noiseFactor * shearModulus *
                                             burgSize *  L2;
                    double f1dotplane2 = fabs(dot(newSegForce, glideDirLab[plane2]));
                    double f1dotplanef = fabs(dot(newSegForce, newplane));

                    if (f1dotplanef < zipperThreshold + f1dotplane2) {
                        continue;
                    }
                    
                    if (!pinned2) {
                        double vec3dotb = dot(vec3, burg);
                        nbr2p = nodep - vec3dotb * burg;
                        network->move_node(n2, nbr2p, system->dEp);
                    }
                    
                    // Update the segment glide plane
                    update_seg_plane(network, s2, newplane);
                }
                
            }
            
        } // end of screw check
    } // end loop over all nodes
    
    Kokkos::fence();
    system->timer[system->TIMER_CROSSSLIP].stop();
}

} // namespace ExaDiS
