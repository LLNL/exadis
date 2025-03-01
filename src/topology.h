/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_TOPOLOGY_H
#define EXADIS_TOPOLOGY_H

#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        Topology
 *
 *-------------------------------------------------------------------------*/
class Topology {
public:
    Topology() {}
    Topology(System *system) {}
    virtual void handle(System *system) {}
    virtual ~Topology() {}
    virtual const char* name() { return "TopologyNone"; }
    
    struct Params {
        double splitMultiNodeAlpha;
        Params() { splitMultiNodeAlpha = -1.0; }
        Params(double _splitMultiNodeAlpha) : splitMultiNodeAlpha(_splitMultiNodeAlpha) {}
    };
    
    /*-----------------------------------------------------------------------
     *    Functions:    Helper functions and definitions to generate 
     *                  splitting arm sets for the SplitMultiNode procedure
     *---------------------------------------------------------------------*/
    static const int MAX_POSSIBLE_SPLITS = 16;
    constexpr static int POSSIBLE_SPLITS[16] =
    {
        0,0,0,3,3,10,25,56,119,246,501,1012,2035,4082,8177,16368
    };
    
    /*-----------------------------------------------------------------------
     *    Function:     build_split_list() 
     *                  Recursive function to build the list of arm-splitting 
     *                  possibilities for SplitMultiNode
     *---------------------------------------------------------------------*/
    static int build_split_list(int totalArms, int splitCnt, int level, int countOnly,
                                int *currentSplit, int **splitList)
    {
        int nextArm = currentSplit[level] + 1;
        int maxLevel = splitCnt - 1;
        int listIndex = 0;
        level++;

        for (int i = nextArm; i < totalArms; i++) {
            currentSplit[level] = i;
            if (level < maxLevel) {
                int newSplits = build_split_list(totalArms, splitCnt,
                                                 level, countOnly,
                                                 currentSplit,
                                                 &splitList[listIndex]);
                listIndex += newSplits;
            } else {
                if (countOnly) {
                    listIndex++;
                    continue;
                }
                for (int j = 0; j < splitCnt; j++)
                    splitList[listIndex][currentSplit[j]] = 1;
                splitList[listIndex][totalArms] = splitCnt;
                listIndex++;
            }
        }

        return listIndex;
    }
    
    /*-----------------------------------------------------------------------
     *    Function:     get_arm_sets()
     *                  Generates an array of flags (1 per arm of the node)
     *                  for each possible way in which a node can split.
     *                  Arms with flag=1 will be moved to the new node.
     *---------------------------------------------------------------------*/
    static void get_arm_sets(int numNbrs, int *setCnt, int ***armSetList)
    {
        if (numNbrs > MAX_CONN)
            ExaDiS_fatal("Topology found node with too many segs (%d)", numNbrs);

        int maxSets = POSSIBLE_SPLITS[numNbrs];
        int maxSplitCnt = numNbrs >> 1;

        // Special case: for 3-arm node set maxSplitCnt to 2
        // to build the proper armSet list
        if (numNbrs == 3) maxSplitCnt = 2;

        int totalSets = 0;
        int level = 0;

        int *currSet = (int *)malloc(sizeof(int) * maxSplitCnt);
        int **armSets = (int **)calloc(1, maxSets * sizeof(int *));

        for (int j = 0; j < maxSets; j++)
            armSets[j] = (int *)calloc(1, (numNbrs + 1) * sizeof(int));

        for (int splitCnt = 2; splitCnt <= maxSplitCnt; splitCnt++) {
            int maxStartArm = (numNbrs - splitCnt) + 1;

            for (int startArm = 0; startArm < maxStartArm; startArm++) {
                currSet[0] = startArm;
                int numSets = build_split_list(numNbrs, splitCnt,
                                               level, 0, currSet,
                                               &armSets[totalSets]);
                totalSets += numSets;
                if ((splitCnt << 1) == numNbrs) break;
            }
        }

        if (totalSets != maxSets) {
            ExaDiS_fatal("%s: expected %d %s %d-node, but found %d",
                         "Topology", maxSets,
                         "split possibilities for", numNbrs, totalSets);
        }

        free(currSet);

        *setCnt = totalSets;
        *armSetList = armSets;
    }
    
    /*-----------------------------------------------------------------------
     *    Function:     execute_split()
     *                  Execute the favorable node splitting
     *---------------------------------------------------------------------*/
    static int execute_split(System* system, SerialDisNet* network, int i,
                             std::vector<int>& arms, Vec3& p0, Vec3& p1)
    {
        std::vector<NodeTag> tagarms;
        if (system->oprec) {
            for (const auto& c : arms)
                tagarms.push_back(network->nodes[network->conn[i].node[c]].tag);
        }
        
        int nconn = network->conn[i].num;
        int inew = network->split_node(i, arms);
        // Update the plastic strain to avoid topological flickers
        network->update_node_plastic_strain(i, network->nodes[i].pos, p0, system->dEp);
        network->update_node_plastic_strain(inew, network->nodes[inew].pos, p1, system->dEp);
        // Update nodes position
        network->nodes[i].pos = network->cell.pbc_fold(p0);
        network->nodes[inew].pos = network->cell.pbc_fold(p1);
        
        // Flag physical corner nodes for 3-node splitting
        if (nconn == 3) {
            if (network->conn[i].num == 2) network->nodes[i].constraint = CORNER_NODE;
            if (network->conn[inew].num == 2) network->nodes[inew].constraint = CORNER_NODE;
        }
        
        // Find glide plane for new segment if it exists
        int cnew = network->find_connection(i, inew);
        if (cnew != -1 && system->crystal.use_glide_planes) {
            int snew = network->conn[i].seg[cnew];
            Vec3 bnew = network->segs[snew].burg;
            Vec3 pnew = system->crystal.find_precise_glide_plane(bnew, p1-p0);
            if (pnew.norm2() < 1e-3)
                pnew = system->crystal.pick_screw_glide_plane(network, bnew);
            network->segs[snew].plane = pnew;
        }
        
        if (system->oprec) {
            NodeTag& tag = network->nodes[i].tag;
            NodeTag& tagnew = network->nodes[inew].tag;
            system->oprec->add_op(OpRec::SplitMultiNode(tag, tagarms, p0, p1, tagnew));
        }
        
        return inew;
    }
};

/*---------------------------------------------------------------------------
 *
 *    Function:     check_node_for_split()
 *                  Preliminary checks to see if we should do a split
 *
 *-------------------------------------------------------------------------*/
template<class N>
KOKKOS_INLINE_FUNCTION
bool check_node_for_split(System* system, N* net, const int& i, int& nsplit)
{
    double shortseg = fmin(5.0, system->params.minseg * 0.1);

    auto nodes = net->get_nodes();
    auto segs = net->get_segs();
    auto conn = net->get_conn();
    auto cell = net->cell;
    
    int nconn = conn[i].num;
    
    // Minimum degree of multi-nodes to consider for a split
    int splitArmsMin = 4;
    if (system->params.split3node == 1) splitArmsMin = 3;
    if (nconn < splitArmsMin) return false;
    
    // If any of the nodes arms is too short, skip the split
    // to avoid creating nodes with very short segments
    Vec3 r0 = nodes[i].pos;
    for (int k = 0; k < nconn; k++) {
        int nei = conn[i].node[k];
        Vec3 rk = cell.pbc_position(r0, nodes[nei].pos);
        if ((rk-r0).norm2() < shortseg*shortseg) {
            return false;
        }
    }
    
    // Three-arm nodes are special cases. Let's only handle
    // the splitting of BCC binary junction nodes whose junction
    // arm does not belong in the intersection of parent planes.
    int binaryjunc = -1;
    if (nconn == 3) {
        //if (system->crystal->type != BCC_CRYSTAL) return false;
        if (system->crystal.type != BCC_CRYSTAL) return false;
        int planarjunc;
        Vec3 tjunc;
        binaryjunc = BCC_binary_junction_node(system, net, i, tjunc, &planarjunc);
        if (binaryjunc == -1) return false;
        if (planarjunc) return false;
    }
    
    // If we are dealing with a 3-arm node, let's
    // make sure that we are only allowing a split along
    // the junction arm. Any other split is identical to
    // a remesh operation (non-physical) and will thus
    // likely result in a higher (artificial) dissipation.
    // So let's only add 1 split for the 3 nodes case.
    if (nconn == 3) nsplit = 1;
    
    return true;
}

} // namespace ExaDiS


// Available topology types
#include "topology_serial.h"
#include "topology_parallel.h"

#endif
