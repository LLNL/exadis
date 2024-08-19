/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_DEBUG_H
#define EXADIS_DEBUG_H

#include "system.h"

namespace ExaDiS { namespace debug {
    
/*-----------------------------------------------------------------------
 *    Function:     verify_Burgers()
 *---------------------------------------------------------------------*/
inline int verify_Burgers(SerialDisNet* net)
{
    if (net->conn.empty())
        net->generate_connectivity();
    net->update_ptr();
	int nb = 0; int nc = 0; int nl = 0; int nd = 0;
	for (int i = 0; i < net->Nnodes_local; i++) {
        if (net->conn[i].num == 0) nc++;
		Vec3 bsum(0.0);
		for (int j = 0; j < net->conn[i].num; j++) {
			Vec3 b = net->segs[net->conn[i].seg[j]].burg;
			bsum += net->conn[i].order[j]*b;
            if (b.norm2() < 1e-5) nl++;
		}
		double b2 = bsum.norm2();
		if (b2 > 1e-5) {
			nb++;
			//printf(" Warning: Burgers vector is not conserved for node %d (bsum = %e, conn = %lu)\n",
            //i, sqrt(b2), net->conn[i].num);
            if (net->conn[i].num > 1) nd++;
		}
	}

	if (nb == 0) {
		printf(" Burgers vector is conserved for all nodes\n");
	} else {
        printf(" Warning: Burgers vector is not conserved for %d node(s)\n", nb);
        if (nd > 0) printf(" Warning: Burgers vector is not conserved for %d node(s) with more than 1 arm\n", nd);
    }
#if 1
    if (nc > 0) printf(" Warning: %d node(s) are unconnected\n", nc);
    if (nl > 0) printf(" Warning: %d link(s) have zero Burgers vector\n", nl);
#endif
    return nb;
}

/*-----------------------------------------------------------------------
 *    Function:     write_force()
 *---------------------------------------------------------------------*/
inline void write_force(System* system, std::string file) {
    printf(" write_force: %s\n", file.c_str());
    auto net = system->get_serial_network();
    FILE *fp = fopen(file.c_str(), "w");
    for (int i = 0; i < net->number_of_nodes(); i++)
        fprintf(fp, "%e %e %e\n", net->nodes[i].f.x, net->nodes[i].f.y, net->nodes[i].f.z);
    fclose(fp);
}

/*-----------------------------------------------------------------------
 *    Function:     write_velocity()
 *---------------------------------------------------------------------*/
inline void write_velocity(System* system, std::string file) {
    printf(" write_velocity: %s\n", file.c_str());
    auto net = system->get_serial_network();
    FILE *fp = fopen(file.c_str(), "w");
    for (int i = 0; i < net->number_of_nodes(); i++)
        fprintf(fp, "%e %e %e\n", net->nodes[i].v.x, net->nodes[i].v.y, net->nodes[i].v.z);
    fclose(fp);
}

/*-----------------------------------------------------------------------
 *    Function:     print_node()
 *---------------------------------------------------------------------*/
template<class N>
KOKKOS_INLINE_FUNCTION
void print_node(N* net, int i, int print_nei=0)
{
    auto nodes = net->get_nodes();
    auto segs = net->get_segs();
    auto conn = net->get_conn();
    
    printf("NODE %d (%d,%d): nconn = %d\n", i, nodes[i].tag.domain, nodes[i].tag.index, conn[i].num);
    printf("   pos = %e %e %e\n",nodes[i].pos.x,nodes[i].pos.y,nodes[i].pos.z);
    printf("   f = %e %e %e\n",nodes[i].f.x,nodes[i].f.y,nodes[i].f.z);
    printf("   v = %e %e %e\n",nodes[i].v.x,nodes[i].v.y,nodes[i].v.z);
    for(int j = 0; j < conn[i].num; j++) {
        int k = conn[i].node[j];
        int s = conn[i].seg[j];
        int o = conn[i].order[j];
        Vec3 b = o * segs[s].burg;
        NodeTag t = nodes[k].tag;
        printf("  conn[%d]: node = %d (%d,%d), seg = %d, order = %d\n",j,k,t.domain,t.index,s,o);
        printf("            burg = %e %e %e\n",b.x,b.y,b.z);
    }
    if (print_nei) {
        for(int j = 0; j < conn[i].num; j++) {
            int k = conn[i].node[j];
            print_node(net, k, 0);
        }
    }
}

/*-----------------------------------------------------------------------
 *    Function:     print_seg()
 *---------------------------------------------------------------------*/
template<class N>
KOKKOS_INLINE_FUNCTION
void print_seg(N* net, int i, int print_nodes=0)
{
    auto nodes = net->get_nodes();
    auto segs = net->get_segs();
    auto conn = net->get_conn();
    
    printf("SEG %d:\n", i);
    printf("   n1 = %d, n2 = %d\n",segs[i].n1,segs[i].n2);
    printf("   burg = %e %e %e\n",segs[i].burg.x,segs[i].burg.y,segs[i].burg.z);
    if (print_nodes) {
        print_node(net, segs[i].n1, 0);
        print_node(net, segs[i].n2, 0);
    }
}

/*-----------------------------------------------------------------------
 *    Function:     extract_nodes()
 *---------------------------------------------------------------------*/
inline void extract_nodes(SerialDisNet* net, std::vector<int> nodes, std::string filename)
{
    SerialDisNet* ncopy = new SerialDisNet(net->cell);
    ncopy->nodes = net->nodes;
    ncopy->segs = net->segs;
    ncopy->conn = net->conn;
    std::vector<int> sflag(net->number_of_segs(), 1);
    for (int i : nodes)
        for (int j = 0; j < ncopy->conn[i].num; j++)
            sflag[ncopy->conn[i].seg[j]] = 0;
    std::vector<int> remsegs;
    for (int i = 0; i < net->number_of_segs(); i++)
        if (sflag[i]) remsegs.push_back(i);
    ncopy->remove_segs(remsegs);
    ncopy->purge_network();
    ncopy->write_data(filename);
    delete ncopy;
}

/*-----------------------------------------------------------------------
 *    Function:     extract_segs()
 *---------------------------------------------------------------------*/
inline void extract_segs(SerialDisNet* net, std::vector<int> segs, std::string filename)
{
    SerialDisNet* ncopy = new SerialDisNet(net->cell);
    ncopy->nodes = net->nodes;
    ncopy->segs = net->segs;
    ncopy->conn = net->conn;
    std::vector<int> sflag(net->number_of_segs(), 1);
    for (int i : segs)
        sflag[i] = 0;
    std::vector<int> remsegs;
    for (int i = 0; i < net->number_of_segs(); i++)
        if (sflag[i]) remsegs.push_back(i);
    ncopy->remove_segs(remsegs);
    ncopy->purge_network();
    ncopy->write_data(filename);
    delete ncopy;
}

} } // namespace ExaDis::debug

#endif
