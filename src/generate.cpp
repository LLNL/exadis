/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "system.h"

#include <random>

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *	Function:	insert_frs
 *
 *-------------------------------------------------------------------------*/
void insert_frs(SerialDisNet *network, Vec3 burg, Vec3 plane, Vec3 ldir, 
                double L, Vec3 center, Mat33 R, int numnodes)
{
    ldir = R * ldir.normalized();
    plane = R * plane.normalized();
    burg = R * burg;
    
    if (fabs(dot(burg, plane)) >= 1e-5)
        ExaDiS_log("Warning: Burgers vector and plane normal are not orthogonal\n");

    int istart = network->number_of_nodes();
    for (int i = 0; i < numnodes; i++) {
        Vec3 p = center + -0.5*L*ldir + 1.0*i*L/(numnodes-1)*ldir;
        p = network->cell.pbc_fold(p);
        network->add_node(p, (i == 0 || i == numnodes-1) ? PINNED_NODE : UNCONSTRAINED);
    }
    for (int i = 0; i < numnodes-1; i++)
        network->add_seg(istart+i, istart+i+1, burg, plane);
}

/*---------------------------------------------------------------------------
 *
 *	Function:	insert_frs
 *
 *-------------------------------------------------------------------------*/
void insert_frs(SerialDisNet *network, Vec3 burg, Vec3 plane, 
                double thetadeg, double L, Vec3 center,
                Mat33 R, int numnodes)
{
    Vec3 b = burg.normalized();
    Vec3 p = plane.normalized();
    Vec3 y = cross(p, b).normalized();
    Vec3 ldir = cos(thetadeg*M_PI/180)*b+sin(thetadeg*M_PI/180)*y;
    insert_frs(network, burg, plane, ldir, L, center, R, numnodes);
}

/*---------------------------------------------------------------------------
 *
 *	Function:	insert_infinite_line
 *
 *-------------------------------------------------------------------------*/
double insert_infinite_line(SerialDisNet* network, Vec3 burg, Vec3 plane, Vec3 ldir, 
                            Vec3 origin, Mat33 R, double maxseg)
{
    ldir = R * ldir.normalized();
    plane = R * plane.normalized();
    burg = R * burg;
    
    if (fabs(dot(burg, plane)) >= 1e-5)
        ExaDiS_log("Warning: Burgers vector and plane normal are not orthogonal\n");
    
    Mat33 H = network->cell.H;
    double Lmin = fmin(fmin(H.xx(), H.yy()), H.zz());
    double seglength = 0.15*Lmin;
    if (maxseg > 0.0)
        seglength = fmin(seglength, maxseg);
        
    bool meet = 0;
    int maxnodes = 1000;
    int numnodes = 0;
    Vec3 p = 1.0*origin;
    Vec3 originpbc = 1.0*origin;
    while (!meet && (numnodes < maxnodes)) {
        p += seglength*ldir;
        Vec3 pp = network->cell.pbc_position(origin, p);
        double dist = (pp-origin).norm();
        if ((numnodes > 0) && (dist < seglength)) {
            originpbc = network->cell.pbc_position(p, origin);
            meet = 1;
        }
        numnodes++;
    }
    
    if (numnodes == maxnodes) {
        ExaDiS_log("Warning: infinite line is too long, aborting\n");
        return -1.0;
    }
    
    int istart = network->number_of_nodes();
    for (int i = 0; i < numnodes; i++) {
        Vec3 p = origin + 1.0*i/numnodes*(originpbc-origin);
        p = network->cell.pbc_fold(p);
        network->add_node(p);
        network->add_seg(istart+i, istart+(i+1)%numnodes, burg, plane);
    }
    
    return (originpbc-origin).norm();
}

/*---------------------------------------------------------------------------
 *
 *	Function:	insert_infinite_line
 *
 *-------------------------------------------------------------------------*/
double insert_infinite_line(SerialDisNet* network, Vec3 burg, Vec3 plane, double thetadeg, 
                            Vec3 origin, Mat33 R, double maxseg)
{
    Vec3 b = burg.normalized();
    Vec3 p = plane.normalized();
    Vec3 y = cross(p, b).normalized();
    Vec3 ldir = cos(thetadeg*M_PI/180)*b+sin(thetadeg*M_PI/180)*y;
    return insert_infinite_line(network, burg, plane, ldir, origin, R, maxseg);
}

/*---------------------------------------------------------------------------
 *
 *	Function:	insert_prismatic_loop
 *
 *-------------------------------------------------------------------------*/
void insert_prismatic_loop(Crystal& crystal, SerialDisNet *network, Vec3 burg, 
                           double radius, Vec3 center, double maxseg)
{
    int Nsides;
    Vec3 e[12], n[12];
    
    if (crystal.type == BCC_CRYSTAL) {
        burg = -1.0*burg;
        Nsides = 6;
        e[0] = Vec3(-2.0*burg.x, burg.y, burg.z);
        e[2] = Vec3(burg.x, -2.0*burg.y, burg.z);
        e[4] = Vec3(burg.x, burg.y, -2.0*burg.z);
        e[1] = -1.0*e[4];
        e[3] = -1.0*e[0];
        e[5] = -1.0*e[2];
        
        for (int i = 0; i < 6; i++)
            n[i] = cross(burg, e[(i+1)%Nsides]-e[i]).normalized();
        for (int i = 0; i < 6; i++)
            e[i] = crystal.R * e[i].normalized();
        
    } else if (crystal.type == FCC_CRYSTAL) {
        Nsides = 4;
        int bid = crystal.identify_closest_Burgers_index(crystal.R * burg);
        Vec3 p1 = crystal.ref_planes(bid*3+0);
        Vec3 p2 = crystal.ref_planes(bid*3+1);
        
        Vec3 l1 = cross(p1, burg).normalized();
        Vec3 l2 = cross(p2, burg).normalized();
        e[0] = -0.5*l1-0.5*l2; n[0] = p1;
        e[1] = +0.5*l1-0.5*l2; n[1] = p2;
        e[2] = +0.5*l1+0.5*l2; n[2] = p1;
        e[3] = -0.5*l1+0.5*l2; n[3] = p2;
        
        for (int i = 0; i < 4; i++)
            e[i] = crystal.R * e[i];
        
    } else {
        ExaDiS_fatal("Error: insert_prismatic_loop() not available for crystal type = %d\n", crystal.type);
    }

    int istart = network->number_of_nodes();
    int Nnodes = 0;
    for (int i = 0; i < Nsides; i++) {
        Vec3 l = radius*(e[(i+1)%Nsides]-e[i]);
        int Nseg = (maxseg > 0) ? (int)ceil(l.norm()/maxseg) : 1;
        for (int j = 0; j < Nseg; j++) {
            Vec3 p = radius*e[i]+1.0*j/Nseg*l+center;
            p = network->cell.pbc_fold(p);
            network->add_node(p);
            int n1 = istart+Nnodes;
            int n2 = (i == Nsides-1 && j == Nseg-1) ? istart : n1+1;
            network->add_seg(n1, n2, crystal.R*burg, crystal.R*n[i]);
            Nnodes++;
        }
    }
}

/*---------------------------------------------------------------------------
 *
 *	Function:		generate_frs_config
 *
 *-------------------------------------------------------------------------*/
SerialDisNet* generate_frs_config(Crystal crystal, Cell cell, int numsources,
                                  double Lsource, double maxseg, int seed)
{
    printf("generate_frs_config()\n");
    
    if (cell.volume() <= 0.0)
        ExaDiS_fatal("Error: undefined cell\n");
    if (Lsource <= 0.0)
        ExaDiS_fatal("Error: undefined source length\n");
    if (maxseg <= 0.0) {
        double Lmin = fmin(cell.H[0][0], fmin(cell.H[1][1], cell.H[2][2]));
        maxseg = 0.05 * Lmin;
    }
    
    SerialDisNet* network = new SerialDisNet(cell);
    
    double theta = 90.0; // character angle in degrees
    
    Mat33 R = crystal.R;
    
    if (crystal.num_sys == 0)
        ExaDiS_fatal("Error: generate_frs_config() not available for crystal type = %d\n", crystal.type);
    
    std::vector<Vec3> blist, nlist;
    for (int i = 0; i < crystal.num_sys; i++) {
        blist.push_back(crystal.ref_burgs(crystal.ref_sys(i,0)));
        nlist.push_back(crystal.ref_planes(crystal.ref_sys(i,1)));
    }
    
    // Sources positions
    std::vector<Vec3> pos;
    if (numsources > 1) {
        std::random_device rd;
    	auto rng = std::default_random_engine{rd()};
        if (seed < 0) seed = time(NULL);
        rng.seed((unsigned)seed);
        int ng = (int)ceil(cbrt(numsources));
        for (int i = 0; i < ng; i++) {
            for (int j = 0; j < ng; j++) {
                for (int k = 0; k < ng; k++) {
                    double px = 1.0*(i+0.5)/ng;
                    double py = 1.0*(j+0.5)/ng;
                    double pz = 1.0*(k+0.5)/ng;
                    pos.push_back(Vec3(px, py, pz));
                }
            }
        }
        std::shuffle(pos.begin(), pos.end(), rng);
    } else {
        pos.push_back(Vec3(0.5, 0.5, 0.5));
    }
    
    for (int i = 0; i < numsources; i++) {
        Vec3 b = blist[i % crystal.num_sys];
        Vec3 n = nlist[i % crystal.num_sys];
        Vec3 c = network->cell.real_position(pos[i]);
        int numnodes = (int)ceil(Lsource/maxseg);
        insert_frs(network, b, n, theta, Lsource, c, R, numnodes);
    }
    
    return network;
}

/*---------------------------------------------------------------------------
 *
 *	Function:		generate_frs_config
 *
 *-------------------------------------------------------------------------*/
SerialDisNet* generate_frs_config(Crystal crystal, double Lbox, int numsources,
                                  double Lsource, double maxseg, int seed)
{
    if (Lbox <= 0.0)
        ExaDiS_fatal("Error: undefined box size\n");
    
    return generate_frs_config(crystal, Cell(Lbox), numsources,
                               Lsource, maxseg, seed);
}

/*---------------------------------------------------------------------------
 *
 *	Function:		generate_prismatic_config
 *
 *-------------------------------------------------------------------------*/
SerialDisNet* generate_prismatic_config(Crystal crystal, Cell cell, int numsources,
                                        double radius, double maxseg, int seed, bool uniform)
{
    printf("generate_prismatic_config()\n");
    
    if (cell.volume() <= 0.0)
        ExaDiS_fatal("Error: undefined cell\n");
    if (radius <= 0.0)
        ExaDiS_fatal("Error: undefined loop radius\n");
    if (maxseg <= 0.0) {
        double Lmin = fmin(cell.H[0][0], fmin(cell.H[1][1], cell.H[2][2]));
        maxseg = 0.05 * Lmin;
    }
    
    SerialDisNet* network = new SerialDisNet(cell);
    
    if (crystal.num_glissile_burgs == 0)
        ExaDiS_fatal("Error: generate_prismatic_config() not available for crystal type = %d\n", crystal.type);
    
    std::vector<Vec3> blist;
    for (int i = 0; i < crystal.num_glissile_burgs; i++)
        blist.push_back(crystal.ref_burgs(i));
    
    // Sources positions
    std::random_device rd;
    auto rng = std::default_random_engine{rd()};
    if (seed < 0) seed = time(NULL);
    rng.seed((unsigned)seed);
    
    std::vector<Vec3> pos;
    if (uniform) {
        if (numsources > 1) {
            int ng = (int)ceil(cbrt(numsources));
            std::uniform_real_distribution<double> dp(-0.5, 0.5);
            for (int i = 0; i < ng; i++) {
                for (int j = 0; j < ng; j++) {
                    for (int k = 0; k < ng; k++) {
                        double px = 1.0*(i+0.5+0.5*dp(rng))/ng;
                        double py = 1.0*(j+0.5+0.5*dp(rng))/ng;
                        double pz = 1.0*(k+0.5+0.5*dp(rng))/ng;
                        pos.push_back(Vec3(px, py, pz));
                    }
                }
            }
            std::shuffle(pos.begin(), pos.end(), rng);
        } else {
            pos.push_back(Vec3(0.5, 0.5, 0.5));
        }
    } else {
        std::uniform_real_distribution<double> dp(0.0, 1.0);
        for (int i = 0; i < numsources; i++)
            pos.push_back(Vec3(dp(rng), dp(rng), dp(rng)));
    }
    
    for (int i = 0; i < numsources; i++) {
        Vec3 b = blist[i % crystal.num_glissile_burgs];
        Vec3 c = network->cell.real_position(pos[i]);
        insert_prismatic_loop(crystal, network, b, radius, c, maxseg);
    }
    
    return network;
}

/*---------------------------------------------------------------------------
 *
 *	Function:		generate_prismatic_config
 *
 *-------------------------------------------------------------------------*/
SerialDisNet* generate_prismatic_config(Crystal crystal, double Lbox, int numsources,
                                        double radius, double maxseg, int seed, bool uniform)
{
    if (Lbox <= 0.0)
        ExaDiS_fatal("Error: undefined box size\n");
    
    return generate_prismatic_config(crystal, Cell(Lbox), numsources,
                                     radius, maxseg, seed, uniform);
}

/*---------------------------------------------------------------------------
 *
 *	Function:		read_paradis
 *
 *-------------------------------------------------------------------------*/
struct ParaDisSeg {
    NodeTag nt1, nt2;
    int n1, n2;
    Vec3 burg, plane;
    ParaDisSeg(NodeTag _nt1, NodeTag _nt2, Vec3 _b, Vec3 _p) {
        nt1 = _nt1; nt2 = _nt2;
        burg = _b;
        plane = _p;
        n1 = -1; n2 = -1;
    }
};

SerialDisNet* read_paradis(const char* file, bool verbose)
{
    if (verbose) printf("Reading ParaDiS configuration\n");
    
    FILE *fp = fopen(file, "r");
    if (fp == NULL)
        ExaDiS_fatal("Error: cannot open ParaDiS file %s\n", file);
    else
        if (verbose) printf(" Input file: %s\n", file);
    
    char *line = NULL;
    size_t len = 0;
    
    // Read general information
    bool found_min = 0, found_max = 0, found_data = 0;
    Vec3 minBounds, maxBounds;
    int nodeCount = -1;
    
    while (getline(&line, &len, fp) != -1) {
        if (strncmp(line, "minCoordinates = [", 18) == 0) {
            found_min = 1;
            fscanf(fp, "%lf %lf %lf\n", &minBounds[0], &minBounds[1], &minBounds[2]);
            //minBounds.print(" minBounds");
        }
        if (strncmp(line, "maxCoordinates = [", 18) == 0) {
            found_max = 1;
            fscanf(fp, "%lf %lf %lf\n", &maxBounds[0], &maxBounds[1], &maxBounds[2]);
            //maxBounds.print(" maxBounds");
        }
        if (strncmp(line, "nodeCount", 9) == 0) {
            sscanf(line, "nodeCount = %d", &nodeCount);
            //printf(" nodeCount = %d\n",nodeCount);
        }
        if (strncmp(line, "nodalData =", 11) == 0) {
            found_data = 1;
            break;
        }
    }
    
    if (!found_min || !found_max || !found_data || nodeCount < 0)
        ExaDiS_fatal("Error: invalid ParaDiS file %s\n", file);
    
    Cell cell(minBounds, maxBounds);
    SerialDisNet *network = new SerialDisNet(cell);
    
    // Read nodal data
    std::map<NodeTag, int> nodeMap;
    std::vector<ParaDisSeg> segs;
    while (getline(&line, &len, fp) != -1) {
        if (strncmp(line, "#", 1) == 0) continue; // skip comment lines
        
        NodeTag ntag;
        Vec3 pos;
        int narms, flag;
        sscanf(line, "%d, %d %lf %lf %lf %d %d",
        &ntag.domain, &ntag.index, &pos[0], &pos[1], &pos[2], &narms, &flag);
        
        auto iter = nodeMap.find(ntag);
        if (iter == nodeMap.end()) {
            nodeMap.emplace(ntag, network->number_of_nodes());
            network->add_node(pos, flag);
        }
        
        for (int i = 0; i < narms; i++) {
            NodeTag nntag;
            Vec3 burg, plane;
            fscanf(fp, "%d, %d %lf %lf %lf\n",
            &nntag.domain, &nntag.index, &burg[0], &burg[1], &burg[2]);
            fscanf(fp, "%lf %lf %lf\n", &plane[0], &plane[1], &plane[2]);
            segs.emplace_back(ntag, nntag, burg, plane);
        }
    }
    free(line);
    fclose(fp);
    
    if (nodeCount != network->number_of_nodes())
        ExaDiS_fatal("Error: node inconsistency found in ParaDiS file\n");
    
    std::map<std::array<int,2>, int> segMap;
    for (int i = 0; i < segs.size(); i++) {
        auto nt1 = nodeMap.find(segs[i].nt1);
        auto nt2 = nodeMap.find(segs[i].nt2);
        if (nt1 == nodeMap.end() || nt2 == nodeMap.end())
            ExaDiS_fatal("Error: invalid segment found in ParaDiS file\n");
        segs[i].n1 = nt1->second;
        segs[i].n2 = nt2->second;
        std::array<int,2> s = {segs[i].n1, segs[i].n2};
        segMap.emplace(s, i);
    }
    // Check segment consistency
    for (int i = 0; i < segs.size(); i++) {
        int n1 = segs[i].n1;
        int n2 = segs[i].n2;
        std::array<int,2> s = {n2, n1};
        auto ns = segMap.find(s);
        if (ns == segMap.end())
            ExaDiS_fatal("Error: segment inconsistency found in ParaDiS file\n");
        if ((segs[i].burg + segs[ns->second].burg).norm2() > 1e-5)
            ExaDiS_fatal("Error: segment inconsistency found in ParaDiS file\n");
        if (n1 < n2)
            network->add_seg(n1, n2, segs[i].burg, segs[i].plane);
    }
    
    if (verbose) printf(" nodes: %d, segments: %d\n", 
    network->number_of_nodes(), network->number_of_segs());
    
    // Verify Burgers vector conservation
    network->generate_connectivity();
    int nb = 0;
    for (int i = 0; i < network->number_of_nodes(); i++) {
        Vec3 bsum(0.0);
        for (int j = 0; j < network->conn[i].num; j++)
            bsum += network->conn[i].order[j] * network->segs[network->conn[i].seg[j]].burg;
        if (bsum.norm2() > 1e-5) nb++;
    }
    if (nb > 0) printf(" Warning: Burgers vector is not conserved for %d node(s)\n", nb);
    else if (verbose) printf(" Burgers vector is conserved for all nodes\n");
    
    return network;
}

} // namespace ExaDiS
