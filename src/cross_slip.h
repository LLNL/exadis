/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_CROSS_SLIP_H
#define EXADIS_CROSS_SLIP_H

#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        CrossSlip
 *
 *-------------------------------------------------------------------------*/
class CrossSlip {
public:
    CrossSlip() {}
    CrossSlip(System* system) {}
    virtual void handle(System* system) {}
    virtual ~CrossSlip() {}
    virtual const char* name() { return "CrossSlipNone"; }
    
    
    /*-----------------------------------------------------------------------
     *    Function:     update_seg_plane()
     *                  Update a segment glide plane
     *---------------------------------------------------------------------*/
    inline void update_seg_plane(SerialDisNet* network, int i, const Vec3& newplane)
    {
        network->segs[i].plane = newplane;
        
        if (network->oprec) {
            NodeTag& tag1 = network->nodes[network->segs[i].n1].tag;
            NodeTag& tag2 = network->nodes[network->segs[i].n2].tag;
            network->oprec->add_op(OpRec::UpdateSegPlane(tag1, tag2, newplane));
        }
    }
};

/*---------------------------------------------------------------------------
 *
 *    Function:     CrossSlipSerial::node_pinned()
 *                  Determine if a node should be considered 'pinned' 
 *                  during the a cross slip procedure.  Nodes are
 *                  to be pinned if any of the following are true:
 *                    - the node is pinned
 *                    - the node is owned by another domain
 *                    - the node has any segments in a plane other
 *                      than the one indicated by <planeIndex>.
 *                    - the node is attached to any segments owned
 *                      by a remote domain.
 *
 *-------------------------------------------------------------------------*/
template<class N>
KOKKOS_INLINE_FUNCTION
bool node_pinned(System* system, N* net, int i, int planeIndex,
                 const Mat33& glidedir, int numglidedir)
{
    auto nodes = net->get_nodes();
    auto segs = net->get_segs();
    auto conn = net->get_conn();
    
    if (nodes[i].constraint == PINNED_NODE) return 1;
    
    // If the node is not owned by the current domain, it
    // may not be repositioned.
    //if (network->nodes[i].tag.domain != rank_id) return 1;

    for (int j = 0; j < conn[i].num; j++) {
        
        //if (!domain_owns_seg(rank_id, network, network->conn[i].seg[j]))
        //    return 1;

        // Check the glide plane for the segment, and if the segment
        // has a different glide plane index than <planeIndex>, the
        // node should be considered 'pinned'
        int s = conn[i].seg[j];
        Vec3 segplane = segs[s].plane;

        int planetest = 0;
        double planetestmin = 10.0;
        for (int k = 0; k < numglidedir; k++) {
            double ptest = fabs(dot(glidedir[k], segplane));
            double ptest2 = ptest * ptest;
            if (ptest2 < planetestmin) {
                planetest = k;
                planetestmin = ptest2;
            }
        }

        if (planeIndex != planetest) return 1;
    }

    return 0;
}

} // namespace ExaDiS


// Available cross-slip types
#include "cross_slip_serial.h"
#include "cross_slip_parallel.h"

#endif
