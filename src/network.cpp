/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    Nicolas Bertin
 *    bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "types.h"
#include "oprec.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Function:     SerialDisNet::constrained_node()
 *
 *-------------------------------------------------------------------------*/
bool SerialDisNet::constrained_node(int i)
{
    if (conn[i].num != 2 || nodes[i].constraint == PINNED_NODE) return 1;
    /*
    SlipPlane plane0 = network->links[conn[i][0].link].plane;
    SlipPlane plane1 = network->links[conn[i][1].link].plane;
    if (!plane0.set || !plane1.set) return 0;
    return !(plane0 == plane1);
    */
    return 0;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     SerialDisNet::discretization_node()
 *
 *-------------------------------------------------------------------------*/
bool SerialDisNet::discretization_node(int i)
{
    // Must be a 2-node with same planes on both sides if planes are defined
    // 2-node if planes are not defined
    return !constrained_node(i);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     SerialDisNet::move_node()
 *
 *-------------------------------------------------------------------------*/
void SerialDisNet::move_node(int i, const Vec3& pos, Mat33& dEp)
{
    update_node_plastic_strain(i, nodes[i].pos, pos, dEp);
    nodes[i].pos = cell.pbc_fold(pos);
    
    if (oprec)
        oprec->add_op(OpRec::MoveNode(nodes[i].tag, pos));
}

/*---------------------------------------------------------------------------
 *
 *    Function:     SerialDisNet::update_node_plastic_strain()
 *                  Update plastic strain (swept area) as a node is being
 *                  moved, e.g. during a topological operation. This is 
 *                  important to avoid topological flickers at low strain 
 *                  rate that can artifically accomodate the imposed strain 
 *                  if not accounted for (plastic strain leakage).
 *
 *-------------------------------------------------------------------------*/
void SerialDisNet::update_node_plastic_strain(int i, const Vec3& pold, const Vec3& pnew, Mat33& dEp)
{
    double vol = cell.volume();
    for (int j = 0; j < conn[i].num; j++) {
        
        int k = conn[i].node[j];
        Vec3 pk = nodes[k].pos;
        Vec3 e = cell.pbc_position(pold, pk) - pold;
        Vec3 h = cell.pbc_position(pk, pnew) - pk;
        Vec3 n = 0.5 * cross(e, h);
        
        int s = conn[i].seg[j];
        int order = conn[i].order[j];
        Vec3 b = order * segs[s].burg;
        
        Mat33 P = 1.0/vol * outer(n, b);
        dEp += 0.5 * (P + P.transpose());
        // We could update the plastic spin as well...
    }
}

/*---------------------------------------------------------------------------
 *
 *    Function:     SerialDisNet::split_seg()
 *                  Split a dislocation segment by bisection at the middle.
 *
 *-------------------------------------------------------------------------*/
int SerialDisNet::split_seg(int i, const Vec3& pos, bool update_conn)
{
    int n1 = segs[i].n1;
    int n2 = segs[i].n2;
    
    // Add new node and seg
    int nnew = number_of_nodes();
    add_node(pos);
    segs[i].n2 = nnew;
    int snew = number_of_segs();
    add_seg(nnew, n2, segs[i].burg, segs[i].plane);
    
    if (update_conn) {
        // Update connections
        int c12 = find_connection(n1, n2);
        conn[n1].node[c12] = nnew;
        int c21 = find_connection(n2, n1);
        conn[n2].node[c21] = nnew;
        conn[n2].seg[c21] = snew;
        Conn cnew;
        cnew.add_connection(n2, snew, 1);
        cnew.add_connection(n1, i, -1);
        conn.emplace_back(cnew);
    }
    
    // Update network pointers in case std::vector
    // triggered a realloc under the hood
    update_ptr();
    
    if (oprec)
        oprec->add_op(OpRec::SplitSeg(nodes[n1].tag, nodes[n2].tag, pos, nodes[nnew].tag));
    
    return nnew;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     SerialDisNet::split_node()
 *                  Split a dislocation node given a list of arms that are
 *                  to be transferred to the new node.
 *
 *-------------------------------------------------------------------------*/
int SerialDisNet::split_node(int i, std::vector<int>& arms)
{
    if (arms.size() == 0) return i;
    
    // Make sure we have unique arms id
    sort(arms.begin(), arms.end());
    arms.erase(unique(arms.begin(), arms.end()), arms.end());
    
    if (arms.size() > MAX_CONN) return -1;
    
    // Add new node
    int nnew = number_of_nodes();
    add_node(nodes[i].pos);
    
    bool error = false;
    Conn cnew;
    Vec3 bnew(0.0);
    for (int k = arms.size()-1; k >=0; k--) {
        int j = arms[k];
        
        bnew += conn[i].order[j] * segs[conn[i].seg[j]].burg;
        
        if (conn[i].order[j] == 1) {
            segs[conn[i].seg[j]].n1 = nnew;
        } else {
            segs[conn[i].seg[j]].n2 = nnew;
        }
        
        int n = conn[i].node[j];
        error |= cnew.add_connection(n, conn[i].seg[j], conn[i].order[j]);
        
        int c = find_connection(n, i);
        conn[n].node[c] = nnew;
        
        conn[i].remove_connection(j);
    }
    conn.emplace_back(cnew);
    
    // Add new link. Warning: New link plane is set to zero
    // and should be assigned outside of this function.
    if (bnew.norm2() > 1e-5) {
        int lnew = number_of_segs();
        add_seg(i, nnew, bnew);
        error |= conn[i].add_connection(nnew, lnew, 1);
        error |= conn[nnew].add_connection(i, lnew, -1);
    }
    
    // Throw an error for now. We may want to revert the split instead.
    if (error)
        ExaDiS_fatal("Error: MAX_CONN = %d exceeded during split_node()\n", MAX_CONN);
    
    // Update network pointers in case std::vector
    // triggered a realloc under the hood
    update_ptr();
    
    return nnew;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     SerialDisNet::merge_nodes_position()
 *                  Merge two dislocation nodes with resulting merged node 
 *                  lying at a specified position.
 *
 *-------------------------------------------------------------------------*/
bool SerialDisNet::merge_nodes_position(int n1, int n2, const Vec3& pos, Mat33& dEp)
{
    // Save original nodes in case we need to revert the merge
    SaveNode saved_node1 = save_node(n1);
    SaveNode saved_node2 = save_node(n2);
    
    // Update the plastic strain as needed
    Mat33 dEp_updt = Mat33().zero();
    update_node_plastic_strain(n1, nodes[n1].pos, pos, dEp_updt);
    update_node_plastic_strain(n2, nodes[n2].pos, pos, dEp_updt);
    
    // Update nodes connectivity
    bool error = false;
    int c, nn;
    for (int j = 0; j < conn[n2].num; j++) {
        //if (conn[n2].node[j] == n1) {
        if (conn[n2].node[j] == n1 || conn[n2].node[j] == n2) {
            // Remove self-connection
            segs[conn[n2].seg[j]].burg = Vec3(0.0);
        } else if ((c = find_connection(n1, conn[n2].node[j])) != -1) {
            // Merge common connection
            Vec3 bj = conn[n1].order[c]*segs[conn[n1].seg[c]].burg +
                      conn[n2].order[j]*segs[conn[n2].seg[j]].burg;
            segs[conn[n1].seg[c]].burg = conn[n1].order[c]*bj;
            segs[conn[n2].seg[j]].burg = Vec3(0.0);
        } else {
            // Transfer neighbor connection
            if (conn[n2].order[j] == 1) {
                segs[conn[n2].seg[j]].n1 = n1;
            } else {
                segs[conn[n2].seg[j]].n2 = n1;
            }
            nn = conn[n2].node[j];
            c = find_connection(nn, n2);
            conn[nn].node[c] = n1;
            error |= conn[n1].add_connection(conn[n2], j);
            conn[n2].remove_connection(j);
            j--;
        }
    }
    
    // The merge would create a node with too many connections.
    // Revert the merge by restoring the original nodes.
    if (error) {
        ExaDiS_log("Error: MAX_CONN = %d exceeded during merge_node()\n", MAX_CONN);
        restore_node(saved_node1);
        restore_node(saved_node2);
    } else {
        // Update merged node position
        nodes[n1].pos = cell.pbc_fold(pos);
        // Update the plastic strain
        dEp += dEp_updt;
    }
    
    // Update network pointers in case std::vector
    // triggered a realloc under the hood
    update_ptr();
    
    if (oprec)
        oprec->add_op(OpRec::MergeNodes(nodes[n1].tag, nodes[n2].tag, pos));
    
    return error;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     SerialDisNet::merge_nodes()
 *                  Merge two dislocation nodes. The resulting merged node
 *                  will lie at the average position.
 *
 *-------------------------------------------------------------------------*/
bool SerialDisNet::merge_nodes(int n1, int n2, Mat33& dEp)
{
    // New node position
    Vec3 p1 = nodes[n1].pos;
    Vec3 p2 = cell.pbc_position(p1, nodes[n2].pos);
    Vec3 pos = 0.5*(p1 + p2);
    
    // Handle end/junction nodes
    if (constrained_node(n1) && 
        constrained_node(n2)) return 0;
    else if (constrained_node(n1)) pos = p1;
    else if (constrained_node(n2)) pos = p2;
    
    return merge_nodes_position(n1, n2, pos, dEp);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     SerialDisNet::remove_segs()
 *                  Remove segments from the network.
 *
 *-------------------------------------------------------------------------*/
void SerialDisNet::remove_segs(std::vector<int> seglist)
{
    if (seglist.size() == 0) return;
    sort(seglist.begin(), seglist.end());
    seglist.erase(unique(seglist.begin(), seglist.end()), seglist.end());
    
    //Remove the links while not preserving the order (faster)
    std::vector<int> offset(number_of_segs(), 0);
    std::vector<int> lid(number_of_segs(), 0);
    for (int i = 0; i < number_of_segs(); i++) offset[i] = lid[i] = i;
    
    for (int i = seglist.size()-1; i >= 0; i--) {
        segs[seglist[i]] = segs.back();
        segs.pop_back();
        lid[seglist[i]] = lid[segs.size()];
        offset[lid[seglist[i]]] = seglist[i];
        offset[seglist[i]] = -1;
    }
    
    for (int i = 0; i < number_of_nodes(); i++) {
        for (int j = 0; j < conn[i].num; j++) {
            if (offset[conn[i].seg[j]] < 0) {
                conn[i].remove_connection(j);
                j--;
            } else {
                conn[i].seg[j] = offset[conn[i].seg[j]];
            }
        }
    }
    
    // Update network pointers in case std::vector
    // triggered a realloc under the hood
    update_ptr();
}

/*---------------------------------------------------------------------------
 *
 *    Function:     SerialDisNet::remove_nodes()
 *                  Remove nodes from the network.
 *
 *-------------------------------------------------------------------------*/
void SerialDisNet::remove_nodes(std::vector<int> nodelist)
{
    if (nodelist.size() == 0) return;

    std::vector<int> remlinks;
    for (int i = 0; i < nodelist.size(); i++) {
        if (conn[nodelist[i]].num > 0) {
            for (int j = 0; j < conn[nodelist[i]].num; j++) {
                remlinks.push_back(conn[nodelist[i]].seg[j]);
            }
            printf(" Warning: removing %d connections for node %d\n",
                   conn[i].num, nodelist[i]);
        }
    }

    sort(nodelist.begin(), nodelist.end());
    nodelist.erase(unique(nodelist.begin(), nodelist.end()), nodelist.end());

    //Remove the nodes while not preserving the order (faster)
    std::vector<int> offset(number_of_nodes(), 0);
    std::vector<int> nid(number_of_nodes(), 0);
    for (int i = 0; i < number_of_nodes(); i++) offset[i] = nid[i] = i;

    for (int i = nodelist.size()-1; i >= 0; i--) {
        free_tag(nodes[nodelist[i]].tag);
        nodes[nodelist[i]] = nodes.back();
        nodes.pop_back();
        conn[nodelist[i]] = conn.back();
        conn.pop_back();
        nid[nodelist[i]] = nid[nodes.size()];
        offset[nid[nodelist[i]]] = nodelist[i];
    }

    for (int i = 0; i < number_of_segs(); i++) {
        segs[i].n1 = offset[segs[i].n1];
        segs[i].n2 = offset[segs[i].n2];
    }
    for (int i = 0; i < number_of_nodes(); i++)
        for (int j = 0; j < conn[i].num; j++)
            conn[i].node[j] = offset[conn[i].node[j]];

    remove_segs(remlinks);
    
    // Update network pointers in case std::vector
    // triggered a realloc under the hood
    update_ptr();
}

/*---------------------------------------------------------------------------
 *
 *    Function:     SerialDisNet::purge_network()
 *                  Purge/clean the dislocation network by removing 
 *                  1) segments with zero Burgers vector, and
 *                  2) isolated (unconnected) nodes
 *
 *-------------------------------------------------------------------------*/
void SerialDisNet::purge_network()
{
    // Remove links with zero Burgers vector
    std::vector<int> remlinks;
    for (int i = 0; i < number_of_segs(); i++)
        if (segs[i].burg.norm2() < 1e-10) remlinks.push_back(i);
    remove_segs(remlinks);

    // Remove isolated nodes
    std::vector<int> remnodes;
    for (int i = 0; i < number_of_nodes(); i++)
        if (conn[i].num == 0) remnodes.push_back(i);
    remove_nodes(remnodes);
    
    if (oprec)
        oprec->add_op(OpRec::PurgeNetwork());
}

/*---------------------------------------------------------------------------
 *
 *    Function:     SerialDisNet::physical_links()
 *                  Returns the list of segments belonging to each physical
 *                  dislocation link (lines connecting physical nodes).
 *                  The network is decomposed by using a hybrid depth /
 *                  breadth first search graph traversal algorithm.
 *
 *-------------------------------------------------------------------------*/
std::vector<std::vector<int> > SerialDisNet::physical_links()
{
    std::vector<std::vector<int> > seglinks;
    std::vector<int> visited(number_of_nodes(), -1);
    
    // Loop over the main component of the graph
    int nni, ilp;
    int np = 0;
    int nl = 0;
    std::vector<int> curlink;
    for (int n = 0; n < number_of_nodes(); n++) {
        if (discretization_node(n)) continue;
        int ni;
        if (visited[n] == -1) {
            np++;
            ni = np-1;
            visited[n] = ni;
        } else {
            ni = visited[n];
        }
        int selfconn = 0;
        for (int k = 0; k < conn[n].num; k++) {
            int nn = conn[n].node[k];
            int il = conn[n].seg[k];
            if (visited[nn] == -1) {
                curlink.push_back(il);
                int prev = n;
                if (!discretization_node(nn)) {
                    np++;
                    nni = np-1;
                    seglinks.push_back(curlink);
                    curlink.clear();
                    nl++;
                    visited[nn] = nni;
                } else {
                    visited[nn] = 1;
                }
                while (discretization_node(nn)) {
                    for (int l = 0; l < conn[nn].num; l++) {
                        if (conn[nn].node[l] != prev || l == conn[nn].num-1) {
                            prev = nn;
                            ilp = conn[nn].seg[l];
                            nn = conn[nn].node[l];
                            break;
                        }
                    }
                    curlink.push_back(ilp);
                    if (!discretization_node(nn)) {
                        if (visited[nn] == -1) {
                            np++;
                            nni = np-1;
                            visited[nn] = nni;
                        } else {
                            nni = visited[nn];
                        }
                        seglinks.push_back(curlink);
                        curlink.clear();
                        nl++;
                    } else {
                        visited[nn] = 1;
                    }
                }
            } else {
                nni = visited[nn];
                if (!discretization_node(nn) && nn > n) {
                    curlink.push_back(il);
                    seglinks.push_back(curlink);
                    curlink.clear();
                    nl++;
                } else if (nn == n) {
                    #if 0
                    if (!selfconn) nl++;
                    selfconn = 1-selfconn;
                    #endif
                }
            }
        }
        if (selfconn) {
            ExaDiS_log("Error: self connection error for node %d\n", n);
            return std::vector<std::vector<int> >();
        }
    }
    
    // Loop over infinite lines and closed loops (minor graph components)
    for (int n = 0; n < number_of_nodes(); n++) {
        if (visited[n] == -1 && discretization_node(n)) {
            np++;
            int ni = np-1;
            visited[n] = ni;
            int prev = n;
            int nn = conn[n].node[0];
            int il = conn[n].seg[0];
            curlink.push_back(il);
            while (nn != n) {
                visited[nn] = 1;
                for (int l = 0; l < conn[nn].num; l++) {
                    if (conn[nn].node[l] != prev || l == conn[nn].num-1) {
                        prev = nn;
                        ilp = conn[nn].seg[l];
                        nn = conn[nn].node[l];
                        break;
                    }
                }
                curlink.push_back(ilp);
            }
            seglinks.push_back(curlink);
            curlink.clear();
            nl++;
        }
    }
    
    return seglinks;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     SerialDisNet::write_data()
 *                  Calculate the dislocation density in 1/m^2
 *
 *-------------------------------------------------------------------------*/
double SerialDisNet::dislocation_density(double burgmag)
{
    double rho = 0.0;
    for (int i = 0; i < number_of_segs(); i++)
        rho += seg_length(i);
    return 1.0/cell.volume()/burgmag/burgmag * rho;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     SerialDisNet::write_data()
 *                  Export the network in ParaDiS data format
 *
 *-------------------------------------------------------------------------*/
void SerialDisNet::write_data(std::string filename)
{
    ExaDiS_log("Writing configuration in legacy data format\n");
    ExaDiS_log(" Output file: %s\n", filename.c_str());
    
    if (conn.empty())
        generate_connectivity();
    
    FILE *fp = fopen(filename.c_str(), "w");
    if (fp == NULL) {
        ExaDiS_fatal("Error: cannot open output file %s\n", filename.c_str());
    }
    
    if (cell.is_triclinic()) {
        ExaDiS_fatal("Error in write_data(): volume must be orthorombic\n");
    }
    
    int version_number = 4;
    int filesegments_number = 1;

    fprintf(fp, "dataFileVersion = %d\n", version_number);
    fprintf(fp, "numFileSegments = %d\n", filesegments_number);

    std::vector<Vec3> bounds = cell.get_bounds();
    fprintf(fp, "minCoordinates = [\n %f\n %f\n %f\n ]\n", bounds[0].x, bounds[0].y, bounds[0].z);
    fprintf(fp, "maxCoordinates = [\n %f\n %f\n %f\n ]\n", bounds[1].x, bounds[1].y, bounds[1].z);

    fprintf(fp, "nodeCount = %d\n", number_of_nodes());

    fprintf(fp, "dataDecompType = 2\n");
    fprintf(fp, "dataDecompGeometry = [\n 1\n 1\n 1\n ]\n\n");
    
    fprintf(fp, "#\n#  END OF DATA FILE PARAMETERS\n#\n\n");

    fprintf(fp, "domainDecomposition = \n");
    fprintf(fp, "# Dom_ID  Minimum XYZ bounds   Maximum XYZ bounds\n");
    fprintf(fp, "  %d  %f  %f  %f  %f  %f  %f\n",
            domain, bounds[0].x, bounds[0].y, bounds[0].z, bounds[1].x, bounds[1].y, bounds[1].z);

    fprintf(fp, "nodalData =\n");
    fprintf(fp, "# Primary lines: node_tag, x, y, z, num_arms, constraint\n");
    fprintf(fp, "# Secondary lines: arm_tag, burgx, burgy, burgz, nx, ny, nz\n");

    for (int i = 0; i < number_of_nodes(); i++) {

        Vec3 pos = cell.pbc_fold(nodes[i].pos);

        fprintf(fp, "%d, %4d %16.12f %16.12f %16.12f %4d %4d\n",
                domain, i, pos[0], pos[1], pos[2],
                conn[i].num, nodes[i].constraint);

        for (int j = 0; j < conn[i].num; j++) {
            int k = conn[i].node[j];
            int s = conn[i].seg[j];
            int o = conn[i].order[j];
            Vec3 b = o * segs[s].burg;
            Vec3 p = segs[s].plane;

            fprintf(fp, "%10d, %4d %16.12f %16.12f %16.12f\n",
                    domain, k, b[0], b[1], b[2]);
            fprintf(fp, "%33.12f %16.12f %16.12f\n", p[0], p[1], p[2]);
        }
    }

    fclose(fp);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     SerialDisNet::save_node()
 *                  Save a node and its neighbors
 *
 *-------------------------------------------------------------------------*/
SerialDisNet::SaveNode SerialDisNet::save_node(int i) {
    SaveNode saved_node;
    saved_node.id = i;
    int nconn = conn[i].num;
    saved_node.nconn = nconn;
    saved_node.nodes.resize(nconn+1);
    saved_node.segs.resize(nconn);
    saved_node.conn.resize(nconn+1);
    saved_node.nodes[0] = nodes[i];
    saved_node.conn[0] = conn[i];
    for (int j = 0; j < nconn; j++) {
        saved_node.nodes[j+1] = nodes[conn[i].node[j]];
        saved_node.segs[j] = segs[conn[i].seg[j]];
        saved_node.conn[j+1] = conn[conn[i].node[j]];
    }
    return saved_node;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     SerialDisNet::restore_node()
 *                  Restore a node and its neighbors
 *
 *-------------------------------------------------------------------------*/
void SerialDisNet::restore_node(SerialDisNet::SaveNode& saved_node) {
    int i = saved_node.id;
    nodes[i] = saved_node.nodes[0];
    conn[i] = saved_node.conn[0];
    for (int j = 0; j < saved_node.nconn; j++) {
        nodes[conn[i].node[j]] = saved_node.nodes[j+1];
        segs[conn[i].seg[j]] = saved_node.segs[j];
        conn[conn[i].node[j]] = saved_node.conn[j+1];
    }
}
    
} // namespace ExaDiS
