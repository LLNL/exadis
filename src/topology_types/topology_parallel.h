/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_TOPOLOGY_PARALLEL_H
#define EXADIS_TOPOLOGY_PARALLEL_H

#include "topology.h"
#include "force.h"
#include "mobility.h"
#include "neighbor.h"

namespace ExaDiS {

typedef Kokkos::View<bool**, T_memory_shared> T_armset;

/*---------------------------------------------------------------------------
 *
 *    Class:        SplitDisNet
 *                  Data structure to hold a temporary, local network
 *                  instance associated with a given trial configuration.
 *                  It provides methods to peform trial splitting operations
 *                  and accessors that override the original network 
 *                  topology locally while making it accessible in parallel.
 *
 *-------------------------------------------------------------------------*/
class SplitDisNet {
public:
    typedef typename DeviceDisNet::ExecutionSpace ExecutionSpace;
    
    static const int MAX_SPLITTABLE_DEGREE = MAX_CONN;
    
    DeviceDisNet* net;
    Cell cell;
    
    bool splitted = 0;
    
    int nid[MAX_SPLITTABLE_DEGREE+2];
    DisNode nodes[2];
    int nconn = 0;
    Conn conn[MAX_SPLITTABLE_DEGREE+2];
    
    int nsegs = 0;
    int sid[MAX_SPLITTABLE_DEGREE+1];
    DisSeg segs[MAX_SPLITTABLE_DEGREE+1];
    
    int Nnodes_local;
    int Nsegs_local;
    
    KOKKOS_INLINE_FUNCTION SplitDisNet(DeviceDisNet* _net, NeighborList* _neilist) : 
    net(_net), neilist(_neilist) {
        cell = net->cell;
        Nnodes_local = net->Nnodes_local;
        Nsegs_local = net->Nsegs_local;
        
        a_nodes.s = this;
        a_segs.s = this;
        a_conn.s = this;
        
        a_count.s = this;
        a_nei.s = this;
    }
    
    KOKKOS_INLINE_FUNCTION
    int split_node(int i, const T_armset& armset, int splitid)
    {
        if (splitted) return -1;
        splitted = 1;
        
        int inew = Nnodes_local++;
        
        // Override node i and create new node inew
        nid[0] = i;
        nodes[0] = net->nodes(i);
        conn[0] = net->conn(i);
        nid[1] = inew;
        nodes[1] = net->nodes(i);
        conn[1] = Conn();
        nconn = 2;
        
        // Setup segments and connectivity
        Vec3 bnew(0.0); // Burgers vector of new arm
        
        for (int j = 0; j < net->conn(i).num; j++) {
            if (armset(splitid,j)) {
                // We need to move arm j to the second node
                int k = net->conn(i).node[j]; // neighbor node
                int s = net->conn(i).seg[j]; // seg i-k
                int order = net->conn(i).order[j];
                Vec3 b = net->segs(s).burg;
                
                // Override and change segment
                int p = nsegs++;
                sid[p] = s;
                segs[p].n1 = (order == 1) ? inew : k;
                segs[p].n2 = (order == 1) ? k : inew;
                segs[p].burg = b;
                segs[p].plane = net->segs(s).plane;
                
                // Override and change connection of node k
                int c = nconn++;
                nid[c] = k;
                conn[c] = net->conn(k);
                for (int l = 0; l < conn[c].num; l++) {
                    if (conn[c].seg[l] == s) {
                        conn[c].node[l] = inew;
                        //conn[c].seg[l] = s;
                        //conn[c].order[l] = -1;
                        break;
                    }
                }
                
                // Add connection at node inew
                conn[1].add_connection(k, s, order);
                
                // Remove connection at node i
                for (int l = 0; l < conn[0].num; l++) {
                    if (conn[0].seg[l] == s) {
                        conn[0].remove_connection(l);
                        break;
                    }
                }
                
                // increment Burgers vector of the new arm
                bnew += (order * b);
            }
        }
        
        // Add the new arm if it exists
        // The new arm is going from r0 to r1 with Burgers bnew
        if (bnew.norm2() > 1e-5) {
            int s = nsegs++;
            int snew = Nsegs_local++;
            
            // Add segment
            sid[s] = snew;
            segs[s].n1 = i;
            segs[s].n2 = inew;
            segs[s].burg = bnew;
            
            // Add connection to node i
            conn[0].add_connection(inew, snew, 1);
            
            // Add connection to node inew
            conn[1].add_connection(i, snew, -1);
            
            // Add snew to neilist of nodes i and inew
            newseg = snew;
        }
        
        return inew;
    }
    
    KOKKOS_INLINE_FUNCTION
    int split_seg(int i, const Vec3& pos)
    {
        if (splitted) return -1;
        splitted = 1;
        
        int n1 = net->segs(i).n1;
        int n2 = net->segs(i).n2;
        int nnew = Nnodes_local++;
        
        // Override node n1 and create new node nnew
        nid[0] = n1;
        nodes[0] = net->nodes(n1);
        conn[0] = net->conn(n1);
        nid[1] = nnew;
        nodes[1] = nodes[0];
        nodes[1].pos = pos;
        conn[1] = Conn();
        nconn = 2;
        
        // Override original segment n1-n2 with n1-nnew
        nsegs++;
        sid[0] = i;
        segs[0] = net->segs(i);
        segs[0].n2 = nnew;
        
        // Update connection of node n1
        for (int l = 0; l < conn[0].num; l++) {
            if (conn[0].node[l] == n2) {
                conn[0].node[l] = nnew;
                break;
            }
        }
        
        // Add new segment nnew-n2
        nsegs++;
        int snew = Nsegs_local++;
        sid[1] = snew;
        segs[1] = net->segs(i);
        segs[1].n1 = nnew;
        newseg = snew; // for neighbor list
        
        // Override connection of node n2
        nconn++;
        nid[2] = n2;
        conn[2] = net->conn(n2);
        for (int l = 0; l < conn[2].num; l++) {
            if (conn[2].node[l] == n1) {
                conn[2].node[l] = nnew;
                conn[2].seg[l] = snew;
                break;
            }
        }
        
        // Add connection of new node
        conn[1].add_connection(n2, snew, 1);
        conn[1].add_connection(n1, i, -1);
        
        return nnew;
    }
    
    // Network accessors
    struct NodeAccessor {
        SplitDisNet* s;
        KOKKOS_FORCEINLINE_FUNCTION
        DisNode& operator[](const int& i) {
            if (s->nconn > 0 && i == s->nid[0]) return s->nodes[0];
            if (s->nconn > 1 && i == s->nid[1]) return s->nodes[1];
            return s->net->nodes(i);
        }
    };
    
    struct SegAccessor {
        SplitDisNet* s;
        KOKKOS_FORCEINLINE_FUNCTION
        DisSeg& operator[](const int& i) {
            for (int j = 0; j < s->nsegs; j++)
                if (i == s->sid[j]) return s->segs[j];
            return s->net->segs(i);
        }
    };
    
    struct ConnAccessor {
        SplitDisNet* s;
        KOKKOS_FORCEINLINE_FUNCTION
        Conn& operator[](const int& i) {
            for (int j = 0; j < s->nconn; j++)
                if (i == s->nid[j]) return s->conn[j];
            return s->net->conn(i);
        }
    };
    
    NodeAccessor a_nodes;
    SegAccessor a_segs;
    ConnAccessor a_conn;
    
    KOKKOS_INLINE_FUNCTION NodeAccessor get_nodes() { return a_nodes; }
    KOKKOS_INLINE_FUNCTION SegAccessor get_segs() { return a_segs; }
    KOKKOS_INLINE_FUNCTION ConnAccessor get_conn() { return a_conn; }
    
    // Neighbor list accessors
    NeighborList* neilist;
    int newseg = -1;
    
    struct NeiCountAccessor {
        SplitDisNet* s;
        KOKKOS_FORCEINLINE_FUNCTION
        int operator[](int i) {
            if (s->nconn > 1 && i == s->nid[1]) i = s->nid[0];
            if (s->newseg >= 0 && s->nconn > 0 && i == s->nid[0]) return s->neilist->count(i)+1;
            return s->neilist->count(i);
        }
    };
    
    struct NeiListAccessor {
        SplitDisNet* s;
        KOKKOS_FORCEINLINE_FUNCTION
        int operator()(int i, const int& n) {
            if (s->nconn > 1 && i == s->nid[1]) i = s->nid[0];
            if (s->newseg >= 0 && s->nconn > 0 && i == s->nid[0]) {
                if (n == s->neilist->count(i)) return s->newseg;
            }
            int beg = s->neilist->beg(i);
            return s->neilist->list(beg+n);
        }
    };
    
    NeiCountAccessor a_count;
    NeiListAccessor a_nei;
    
    KOKKOS_INLINE_FUNCTION NeiCountAccessor get_count() { return a_count; }
    KOKKOS_INLINE_FUNCTION NeiListAccessor get_nei() { return a_nei; }
    
    static const char* name() { return "SplitDisNet"; }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        TopologyParallel
 *                  Class to perform topological operations (SplitMultiNodes)
 *                  efficiently in parallel on the device/host.
 *
 *-------------------------------------------------------------------------*/
template<class F, class M>
class TopologyParallel : public Topology {
public:
    struct SplitMultiNode; // forward declaration
    
private:
    F* force;
    M* mobility;
    typedef typename M::Mob Mob;
    Mob* mob;
    
    SplitMultiNode* smn;
    double splitMultiNodeAlpha;
    NeighborList* neilist;

    static const int MAX_SPLITTABLE_DEGREE = MAX_CONN;
    T_armset armsets[MAX_SPLITTABLE_DEGREE+1];

public:
    TopologyParallel(System* system, Force* _force, Mobility* _mobility, Params params=Params()) 
    {
        // Precompute arms sets splits
        int numSets, **armSets;
        for (int nconn = 3; nconn <= MAX_SPLITTABLE_DEGREE; nconn++) {
            
            get_arm_sets(nconn, &numSets, &armSets);
            
            Kokkos::resize(armsets[nconn], numSets, nconn);
            for (int i = 0; i < numSets; i++)
                for (int j = 0; j < nconn; j++)
                    armsets[nconn](i,j) = (bool)armSets[i][j];
            
            for (int i = 0; i < numSets; i++) free(armSets[i]);
            free(armSets);
        }
        
        // To reproduce ParaDiS results
        splitMultiNodeAlpha = (system->crystal.type == BCC_CRYSTAL) ? 1e-3 : 1.0;
        if (params.splitMultiNodeAlpha >= 0.0)
            splitMultiNodeAlpha = params.splitMultiNodeAlpha;
        
        // Check and assign force/velocity kernels
        force = dynamic_cast<F*>(_force);
        if (force == nullptr)
            ExaDiS_fatal("Error: inconsistent force type in TopologyParallel\n");
            
        mobility = dynamic_cast<M*>(_mobility);
        if (mobility == nullptr)
            ExaDiS_fatal("Error: inconsistent mobility type in TopologyParallel\n");
        mob = mobility->mob;
        
        neilist = exadis_new<NeighborList>();
    }
    
    /*-----------------------------------------------------------------------
     *    Struct:     SplitMultiNode
     *                Structure that implements the kernel to evaluate the
     *                power dissipation for each trial configuration for all
     *                nodes in parallel. Each team is assigned a trial split
     *                and the force calculations are scattered accros team
     *                members using the team node_force() implementations.
     *                The trial configurations are created locally using
     *                the SplitDisNet data structure.
     *---------------------------------------------------------------------*/
    struct SplitMultiNode
    {
        T_armset armsets[MAX_SPLITTABLE_DEGREE+1];
        double splitDist, vNoise;
        double eps = 1e-12;
        
        System* system;
        DeviceDisNet* net;
        F* force;
        Mob* mob;
        NeighborList* neilist;

        Kokkos::View<int**, T_memory_shared> splits;
        Kokkos::View<double**, T_memory_shared> power;
        Kokkos::View<Vec3**, T_memory_shared> splitpos;

        SplitMultiNode(System* _system, DeviceDisNet* _net, TopologyParallel* topology, 
                       F* _force, Mob* _mob) : 
        system(_system), net(_net), force(_force), mob(_mob)
        {
            double rann = system->params.rann;
            splitDist = 2.0*rann + eps;
            vNoise = (system->realdt > 0.0) ? system->params.rtol / system->realdt : 0.0;
            vNoise = topology->splitMultiNodeAlpha * vNoise;
            
            // Copy precomputed armsets for each split
            for (int nconn = 3; nconn <= MAX_SPLITTABLE_DEGREE; nconn++)
                armsets[nconn] = topology->armsets[nconn];
        }

        KOKKOS_INLINE_FUNCTION
        void operator() (const Kokkos::TeamPolicy<>::member_type& team) const
        {
            int ts = team.team_size(); // returns TEAM_SIZE
            int tid = team.team_rank(); // returns a number between 0 and TEAM_SIZE
            //int ls = team.league_size(); // returns N
            int lid = team.league_rank(); // returns a number between 0 and N
            
            auto nodes = net->get_nodes();
            auto segs = net->get_segs();
            auto conn = net->get_conn();
            
            int n = lid; // id in the splits array
            int i = splits(n,0); // node id
            
            int nconn = conn[i].num;
            
            // Create an instance of the SplitDisNet data structure
            // At this stage it only points to the original network
            SplitDisNet splitnet(net, neilist);

            // Compute force and mobility of the node before the split
            // This is a per-node basis so we could do that before
            Vec3 fi = force->node_force(system, &splitnet, i, team);
            
            // Mobility
            Vec3 vi;
            if (tid == 0)
                vi = mob->node_velocity(system, net, i, fi);
            team.team_broadcast(vi, 0);
            
            // Compute power dissipation of the original configuration
            double powerMax = dot(fi, vi);
            
            // Now do the split locally: add new segment and change connectivity
            int splitid = splits(n,1); // split id
            
            // If we are dealing with a 3-arm node in BCC, let's
            // make sure that we are only allowing a split along
            // the junction arm. Any other split is identical to
            // a remesh operation (non-physical) and will thus
            // likely result in a higher (artificial) dissipation.
            // In this case we skip the set.
            if (system->crystal.type == BCC_CRYSTAL && nconn == 3) {
                // Find the armset splitting the junction segment
                splitid = -1;
                for (int k = 0; k < 3; k++) {
                    for (int j = 0; j < nconn; j++) {
                        if (!armsets[3](k,j)) {
                            // j is the splitarm, check if it is the junction seg
                            //if (system->crystal->is_junction_seg(system, net, conn[i].seg[j])) {
                            if (segs[conn[i].seg[j]].burg.norm2() > 1.01) {
                                splitid = k;
                            }
                        }
                        if (splitid >= 0) break;
                    }
                    if (splitid >= 0) break;
                }
                if (tid == 0) splits(n,1) = splitid; // save split id
            }
            
            // Create the new configuration with splitted node
            // within the temporary, local splitnet instance
            int inew = splitnet.split_node(i, armsets[nconn], splitid);
            int s = splitnet.nsegs-1;
            Vec3 bnew(0.0);
            if (splitnet.newseg >= 0)
                bnew = splitnet.segs[s].burg;
            
            // Recomputed forces with new set of arms
            Vec3 f0 = force->node_force(system, &splitnet, i, team);
            Vec3 f1 = force->node_force(system, &splitnet, inew, team);
            
            // Compute mobility of both splitted nodes
            Vec3 v0, v1;
            if (ts > 1) {
                if (tid == 0)
                    v0 = mob->node_velocity(system, &splitnet, i, f0);
                if (tid == 1)
                    v1 = mob->node_velocity(system, &splitnet, inew, f1);
                team.team_broadcast(v0, 0);
                team.team_broadcast(v1, 1);
            } else {
                v0 = mob->node_velocity(system, &splitnet, i, f0);
                v1 = mob->node_velocity(system, &splitnet, inew, f1);
            }
            
            // If we are dealing with a 3-arm node splitting, we should
            // only allow the new 3-arm node to move, the other node being
            // a new discritization node of one of the original arms.
            if (nconn == 3) {
                if (splitnet.conn[0].num == 3 && splitnet.conn[1].num == 2) {
                    v1 = Vec3(0.0);
                } else if (splitnet.conn[0].num == 2 && splitnet.conn[1].num == 3) {
                    v0 = Vec3(0.0);
                }
            }
            
            double powerTest = 0.0;
            
            // Check if velocities are non-zero to continue
            double v0mag = v0.norm();
            double v1mag = v1.norm();
            
            if (v0mag > eps || v1mag > eps) {
            
                if (v0mag > eps) v0 = 1.0/v0mag*v0;
                if (v1mag > eps) v1 = 1.0/v1mag*v1;
                
                // Set nodes to their splitting positions
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
                
                splitnet.nodes[0].pos -= reposition0 * minSplitDist * vdir;
                splitnet.nodes[1].pos += reposition1 * minSplitDist * vdir;
                
                // Set glide plane of new segment if needed
                if (splitnet.newseg >= 0 && system->crystal.use_glide_planes) {
                    Vec3 pnew = system->crystal.find_precise_glide_plane(bnew, vdir);
                    if (pnew.norm2() < 1e-3)
                        pnew = system->crystal.pick_screw_glide_plane(&splitnet, bnew);
                    splitnet.segs[s].plane = pnew;
                }
                
                // Recompute forces and velocities at new splitting positions
                f0 = force->node_force(system, &splitnet, i, team);
                f1 = force->node_force(system, &splitnet, inew, team);
                
                if (ts > 1) {
                    if (tid == 0)
                        v0 = mob->node_velocity(system, &splitnet, i, f0);
                    if (tid == 1)
                        v1 = mob->node_velocity(system, &splitnet, inew, f1);
                    team.team_broadcast(v0, 0);
                    team.team_broadcast(v1, 1);
                } else {
                    v0 = mob->node_velocity(system, &splitnet, i, f0);
                    v1 = mob->node_velocity(system, &splitnet, inew, f1);
                }
                
                // If we are dealing with a 3-arm node, make sure we only
                // compute the dissipation from the new 3-arm node. The 
                // other node is a 2-arm node that is not supposed to move.
                if (nconn == 3) {
                    if (splitnet.conn[0].num == 3 && splitnet.conn[1].num == 2) {
                        v1 = Vec3(0.0);
                    } else if (splitnet.conn[0].num == 2 && splitnet.conn[1].num == 3) {
                        v0 = Vec3(0.0);
                    }
                }
                
                Vec3 vdiff = v1 - v0;
                
                // Compute power dissipation of the trial configuration
                // and store splitting positions
                if (dot(vdiff, vdir) > 0) {
                    powerTest = dot(f0, v0) + dot(f1, v1) - vNoise * (f0.norm() + f1.norm());
                    if (tid == 0) {
                        splitpos(n,0) = splitnet.nodes[0].pos;
                        splitpos(n,1) = splitnet.nodes[1].pos;
                    }
                }
            }
            
            // Write the power dissipation results for this split and then leave
            // The topology change was local so there is nothing to revert
            if (tid == 0) {
                power(n,0) = powerMax;
                power(n,1) = powerTest;
            }
        }
    };
    
    /*-----------------------------------------------------------------------
     *    Function:   split_multi_nodes_parallel()
     *                Main function to perform the split multi-nodes 
     *                procedure in parallel.
     *---------------------------------------------------------------------*/
    void split_multi_nodes_parallel(System* system)
    {
        // Copy the splits array locally so we can access it on the device
        int num_possible_splits[MAX_POSSIBLE_SPLITS];
        for (int i = 0; i < MAX_POSSIBLE_SPLITS; i++)
            num_possible_splits[i] = POSSIBLE_SPLITS[i];
        
        int active_net = system->net_mngr->get_active();
        DeviceDisNet* net = system->get_device_network();
        

        // Initialize the SplitMultiNode structure
        smn = new SplitMultiNode(system, net, this, force, mob);
        
        
        // Find nodes to split and the number of splits for each node
        Kokkos::View<int*, T_memory_shared> nsplits("nsplits", 3);
        Kokkos::View<int*, T_memory_shared> splitnum("splitnum", net->Nnodes_local);
        Kokkos::parallel_for(net->Nnodes_local, KOKKOS_LAMBDA(const int& i) {
            auto nodes = net->get_nodes();
            auto conn = net->get_conn();
            int nconn = conn[i].num;
            // Unflag corner nodes that are not 2-nodes
            if (nodes[i].constraint == CORNER_NODE && nconn != 2)
                nodes[i].constraint = UNCONSTRAINED;
            int nsplit = num_possible_splits[MIN(nconn, MAX_POSSIBLE_SPLITS-1)];
            if (nsplit > 0) {
                if (nconn > MAX_SPLITTABLE_DEGREE) {
                    Kokkos::atomic_add(&nsplits(2), 1);
                }
                // Preliminary checks to see if we should do the split
                else if (check_node_for_split(system, net, i, nsplit)) {
                    splitnum(i) = nsplit;
                    Kokkos::atomic_add(&nsplits(0), nsplit);
                    Kokkos::atomic_add(&nsplits(1), 1);
                }
            }
        });
        Kokkos::fence();
        int numsplits = nsplits(0);
        int nsplitnodes = nsplits(1);
        //printf(" TopologyParallel: nsplits = %d for %d nodes\n", numsplits, nsplitnodes);
        if (nsplits(2) > 0)
            printf("Warning: ignoring n=%d nodes with degree > %d = max splittable degree\n",
            nsplits(2), MAX_SPLITTABLE_DEGREE);

        // Create an array of all individual splits (trial configurations) 
        // These will be fully processed in parallel
        Kokkos::deep_copy(nsplits, 0);
        Kokkos::resize(smn->splits, numsplits, 2);
        Kokkos::View<int**, T_memory_shared>& splits = smn->splits;
        Kokkos::View<int*, T_memory_space> splitnodes("splitnodes", nsplitnodes);
        
        Kokkos::parallel_for(net->Nnodes_local, KOKKOS_LAMBDA(const int& i) {
            int nsplit = splitnum(i);
            if (nsplit > 0) {
                int idx = Kokkos::atomic_fetch_add(&nsplits(0), nsplit);
                int idx1 = Kokkos::atomic_fetch_add(&nsplits(1), 1);
                for (int j = 0; j < nsplit; j++) {
                    splits(idx+j,0) = i;
                    splits(idx+j,1) = j;
                }
                splitnodes(idx1) = i;
            }
        });
        Kokkos::fence();
        if (nsplits(0) != numsplits)
            ExaDiS_fatal("Error: inconsistent number of splits: %d != %d\n", nsplits(0), numsplits);
        
        
        // If we need a neighbor list, let's build a contiguous
        // one for only the subset of split nodes so that access 
        // on device will be much faster.
        double cutoff = system->neighbor_cutoff;
        if (cutoff > 0.0) {
            NeighborBox* neighbox = exadis_new<NeighborBox>(system, cutoff, Neighbor::NeiSeg);
            // Build a neighbor list of the nodes wrt to the segs
            NeighborBox::BuildNeighborList<DeviceDisNet>(system, net, neighbox, neilist, Neighbor::NeiNode, splitnodes);
            smn->neilist = neilist;
            exadis_delete(neighbox);
        }
        
        
        // Now we evaluate the power dissipation of all trial
        // configurations for all nodes in parallel
        Kokkos::resize(smn->power, numsplits, 2);
        Kokkos::resize(smn->splitpos, numsplits, 2);
        Kokkos::parallel_for(Kokkos::TeamPolicy<>(numsplits, Kokkos::AUTO), *smn);
        Kokkos::fence();
        
        
        // We are done with evaluating the power dissipation for all
        // trial configurations. Now execute the most favorable split
        // for each node, if any. We do this in serial for simplicity
        // for now.
        
        // We did not make any changes to the network yet, so
        // let's avoid making unnecessary memory copies
        system->net_mngr->set_active(active_net);
        SerialDisNet* network = system->get_serial_network();
        
        std::vector<bool> nodeflag(network->Nnodes_local, 0);
        
        int isplit = 0;
        for (int n = 0; n < nsplitnodes; n++) {
            
            int i = splits(isplit,0); // node id
            int nsplit = splitnum(i); // number of trial splits for node i
            int nconn = network->conn[i].num;
            
            // Only consider if node and all neighbors have not
            // experienced another split, otherwise skip the node
            int skip = 0;
            if (nodeflag[i]) skip = 1;
            else {
                for (int l = 0; l < nconn; l++) {
                    if (nodeflag[network->conn[i].node[l]]) {
                        skip = 1;
                        break;
                    }
                }
            }
            if (skip) {
                isplit += nsplit;
                continue;
            }
            
            double powerMax = smn->power(isplit,0);
            
            // Select most favorable split for the node
            int kmax = -1;
            Vec3 p0, p1;
            for (int k = 0; k < nsplit; k++) {
                if (splits(isplit,0) != i)
                    ExaDiS_fatal("Error: inconsistent split list\n");
                
                double powerTest = smn->power(isplit,1);
                if (powerTest > powerMax && powerTest > 1.0) {
                    powerMax = powerTest;
                    kmax = splits(isplit,1); // split id
                    p0 = smn->splitpos(isplit,0);
                    p1 = smn->splitpos(isplit,1);
                }
                isplit++;
            }
            
            // Execute the favorable split and flag nodes
            //printf(" node[%d]: kmax = %d, powerMax = %e\n", i, kmax, powerMax);
            if (kmax >= 0) {
                nodeflag[i] = 1;
                std::vector<int> arms;
                for (int l = 0; l < nconn; l++) {
                    if (armsets[nconn](kmax,l)) {
                        arms.push_back(l);
                        nodeflag[network->conn[i].node[l]] = 1;
                    }
                }
                execute_split(system, network, i, arms, p0, p1);
            }
        }
        
        delete smn;
    }
    
    /*-----------------------------------------------------------------------
     *    Function:   handle()
     *---------------------------------------------------------------------*/
    void handle(System *system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_TOPOLOGY].start();
        
        split_multi_nodes_parallel(system);
        
        Kokkos::fence();
        system->timer[system->TIMER_TOPOLOGY].stop();
    }
    
    ~TopologyParallel() {
        exadis_delete(neilist);
    }
    
    const char* name() { return "TopologyParallel"; }
};

} // namespace ExaDiS

#endif
