/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_NEIGHBOR_BOX_H
#define EXADIS_NEIGHBOR_BOX_H

#include "neighbor.h"

namespace ExaDiS {
    
#define MAX_BOX 50

/*---------------------------------------------------------------------------
 *
 *    Class:        NeighborBox
 *                  A neighbor list class for use on host/device. The
 *                  neighbors are found by first binning nodes into a set 
 *                  of regular boxes using a linked list.
 *
 *-------------------------------------------------------------------------*/
class NeighborBox : public Neighbor {
private:
    Cell cell;
    Mat33 cellHinv;
    Vec3 cbox[3];
    int pbc[3];
    int boxDim[3];
    Mat33 binHinv;
    
public:
    NeiType type;
    double cutoff;
    int Nbox_total;
    int N_local;
    
    typedef Kokkos::View<int*, T_memory_space> T_box;
    T_box boxes;
    T_box countBox;
    T_box nextInBox;
    
    NeighborBox(System* system, double _cutoff, NeiType _type)
    {
        DeviceDisNet* net = system->get_device_network();
        build(system, net, _cutoff, _type);
    }
    
    template<class N>
    NeighborBox(System* system, N* net, double _cutoff, NeiType _type) {
        build(system, net, _cutoff, _type);
    }
    
    KOKKOS_INLINE_FUNCTION T_box::pointer_type get_head() { return boxes.data(); }
    KOKKOS_INLINE_FUNCTION T_box::pointer_type get_count() { return countBox.data(); }
    KOKKOS_INLINE_FUNCTION T_box::pointer_type get_next() { return nextInBox.data(); }
    
    KOKKOS_INLINE_FUNCTION
    Vec3i find_box_coord(const Vec3& p) const {
        Vec3 rp = binHinv * (cell.pbc_fold(p) - cell.origin);
        Vec3i id;
        for (int k = 0; k < 3; k++) {
            id[k] = (int)floor(rp[k]);
            id[k] = MAX(MIN(id[k], boxDim[k]-1), 0);
        }
        return id;
    }
    
    KOKKOS_INLINE_FUNCTION
    int find_box_index(const Vec3& p) const {
        Vec3i id = find_box_coord(p);
        return (id[2] * (boxDim[0] * boxDim[1]) + id[1] * boxDim[0] + id[0]);
    }
    
    struct TagFindBoxNodes {};
    struct TagFindBoxSegs {};
    
    template<class N>
    struct FindBox {
        N* net;
        NeighborBox* neighbor;
        
        FindBox(N* _net, NeighborBox* _neighbor) : net(_net), neighbor(_neighbor) {
            Kokkos::resize(neighbor->nextInBox, neighbor->N_local);
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator()(TagFindBoxNodes, const int& i) const {
            auto nodes = net->get_nodes();
            
            Vec3 r = nodes[i].pos;
            int ibox = neighbor->find_box_index(r);
            // Insert node into linked list of boxes
            int nextInBox = Kokkos::atomic_exchange(&neighbor->boxes(ibox), i);
            Kokkos::atomic_increment(&neighbor->countBox(ibox));
            neighbor->nextInBox(i) = nextInBox;
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator()(TagFindBoxSegs, const int& i) const {
            auto nodes = net->get_nodes();
            auto segs = net->get_segs();
            auto cell = net->cell;
            
            // Use segment mid point
            int n1 = segs[i].n1;
            int n2 = segs[i].n2;
            Vec3 r1 = nodes[n1].pos;
            Vec3 r2 = cell.pbc_position(r1, nodes[n2].pos);
            Vec3 r = 0.5*(r1+r2);
            int ibox = neighbor->find_box_index(r);
            // Insert seg into linked list of boxes
            int nextInBox = Kokkos::atomic_exchange(&neighbor->boxes(ibox), i);
            Kokkos::atomic_increment(&neighbor->countBox(ibox));
            neighbor->nextInBox(i) = nextInBox;
        }
    };
    
    template<class N>
    void build(System* system, N* net, double _cutoff, NeiType _type)
    {    
        type = _type;
        cutoff = _cutoff;
        // For segs we use the mid point for binning, so we need to increase
        // the cutoff to make sure that all neighboring segments will be found
        if (type == NeiSeg)
            cutoff += system->params.maxseg;
        
        cell = net->cell;
        cellHinv = cell.Hinv;
        
        pbc[0] = cell.xpbc;
        pbc[1] = cell.ypbc;
        pbc[2] = cell.zpbc;
        
        cbox[0] = cell.H.colx();
        cbox[1] = cell.H.coly();
        cbox[2] = cell.H.colz();
        
        // Determine the dimensions of the 3d bin array
        Vec3 perpVecs[3];
        perpVecs[0] = cross(cbox[1], cbox[2]).normalized();
        perpVecs[1] = cross(cbox[2], cbox[0]).normalized();
        perpVecs[2] = cross(cbox[0], cbox[1]).normalized();

        Mat33 binH;
        for (int i = 0; i < 3; i++) {
            boxDim[i] = (int)floor(fabs(dot(cbox[i], perpVecs[i])) / cutoff);
            boxDim[i] = MIN(boxDim[i], MAX_BOX);
            boxDim[i] = MAX(boxDim[i], 1);
            for (int j = 0; j < 3; j++)
                binH[j][i] = cbox[i][j] / (double)boxDim[i];
        }
        binHinv = binH.inverse();
        
        Nbox_total = boxDim[0] * boxDim[1] * boxDim[2];
        Kokkos::resize(boxes, Nbox_total);
        Kokkos::deep_copy(boxes, -1);
        Kokkos::resize(countBox, Nbox_total);
        
        // Find box for each node/seg
        if (type == NeiNode) {
            N_local = net->Nnodes_local;
            using policy = Kokkos::RangePolicy<typename N::ExecutionSpace,TagFindBoxNodes>;
            Kokkos::parallel_for("NeighborBox::FindBoxNodes",
                policy(0, N_local), FindBox(net, this)
            );
        } else {
            N_local = net->Nsegs_local;
            using policy = Kokkos::RangePolicy<typename N::ExecutionSpace,TagFindBoxSegs>;
            Kokkos::parallel_for("NeighborBox::FindBoxSegs",
                policy(0, N_local), FindBox(net, this)
            );
        }
        Kokkos::fence();
    }
    
    KOKKOS_INLINE_FUNCTION
    Vec3i neighbor_box_coord(int ib)
    {
        int bz = ib / 9;
        int by = (ib - 9*bz) / 3;
        int bx = ib - 9*bz - 3*by;
        return Vec3i(bx-1, by-1, bz-1);
    }
    
    KOKKOS_INLINE_FUNCTION
    int neighbor_box_index(Vec3i c)
    {
        for (int k = 0; k < 3; k++) {
            if (pbc[k] == PBC_BOUND) {
                if (c[k] < 0) c[k] = boxDim[k]-1;
                if (c[k] >= boxDim[k]) c[k] = 0;
            } else {
                if (c[k] < 0) { continue; }
                else if (c[k] >= boxDim[k]) { continue; }
            }
        }
        int nei_box = c[2] * (boxDim[0] * boxDim[1]) + c[1] * boxDim[0] + c[0];
        if (nei_box < 0 || nei_box >= Nbox_total) nei_box = -1;
        return nei_box;
    }
    
    KOKKOS_INLINE_FUNCTION
    int neighbor_box_index(const Vec3i& id, int ib)
    {
        Vec3i b = neighbor_box_coord(ib);
        return neighbor_box_index(id + b);
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    Vec3 get_node_pos(N* net, int i, NeiType _type)
    {
        auto nodes = net->get_nodes();
        if (_type == NeiNode) {
            return nodes[i].pos;
        } else {
            auto segs = net->get_segs();
            int n1 = segs[i].n1;
            int n2 = segs[i].n2;
            Vec3 r1 = nodes[n1].pos;
            Vec3 r2 = cell.pbc_position(r1, nodes[n2].pos);
            return 0.5*(r1+r2);
        }
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    Vec3 get_node_pos(N* net, int i) {
        return get_node_pos(net, i, type);
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    void get_neighbor(N* net, const Vec3& p, int ib, int k, double& dist2)
    {
        Vec3 pnei = get_node_pos(net, k);
        if (boxDim[0] == 1 || boxDim[1] == 1 || boxDim[2] == 1) {
            // If the cutoff is so large that there is only one box in one 
            // of the dimensions, then we need to make sure that we only add 
            // the primary image neighbor to avoid doublons.
            Vec3i b = neighbor_box_coord(ib);
            Vec3 cellDelta(0.0);
            for (int k = 0; k < 3; k++) {
                if (pbc[k] == PBC_BOUND)
                    cellDelta += b[k] * cbox[k];
            }
            Vec3 delta = pnei - p + cellDelta;
            Vec3 rd = cellHinv * delta;
            bool add_nei = (fabs(rd.x) < 0.5 && fabs(rd.y) < 0.5 && fabs(rd.z) < 0.5);
            if (!add_nei) dist2 = 2.0 * cutoff * cutoff;
            else dist2 = delta.norm2();
        } else {
            pnei = cell.pbc_position(p, pnei);
            dist2 = (pnei-p).norm2();
        }
    }
    
    template<class N, class L>
    std::vector<L> query_list(N* net, const Vec3& p)
    {
        std::vector<L> neilist;
        Vec3i id = find_box_coord(p);
		for (int ib = 0; ib < 27; ib++) {
            int nei_box_id = neighbor_box_index(id, ib);
            if (nei_box_id < 0) continue;
            int k = boxes(nei_box_id);
            while (k >= 0) {
                double dist2;
                get_neighbor(net, p, ib, k, dist2);
                if (dist2 <= cutoff*cutoff) {
                    if constexpr (std::is_same<L, int>::value) {
                        neilist.push_back(k);
                    } else {
                        neilist.emplace_back(std::make_pair(k, sqrt(dist2)));
                    }
                }
                k = nextInBox(k);
            }
        }
        return neilist;
    }
    
    template<class N>
    std::vector<int> query(N* net, const Vec3& p)
    {
        return query_list<N,int>(net, p);
    }
    
    template<class N>
    std::vector<int> query(N* net, int i)
    {
        Vec3 p = get_node_pos(net, i);
        return query_list<N,int>(net, p);
    }
    
    template<class N>
    std::vector<std::pair<int,double> > query_distance(N* net, const Vec3& p)
    {
        return query_list<N,std::pair<int,double> >(net, p);
    }
    
    template<class N>
    std::vector<std::pair<int,double> > query_distance(N* net, int i)
    {
        Vec3 p = get_node_pos(net, i);
        return query_list<N,std::pair<int,double> >(net, p);
    }
    
    
    /*-----------------------------------------------------------------------
     *    Struct:     BuildNeighborList
     *                Structure to build a contiguous neighbor list from
     *                the NeighborBox data. The neighbor list is built by
     *                finding neighbors of input nodes/segs within the
     *                existing neighbors (nodes/segs) of the NeighborBox 
     *                instance. It also has the option to build the
     *                neighbor list by only considering a subset of the
     *                input nodes/segs provided with the ind array.
     *                When option strict=false, all nodes/segs in neighbor
     *                boxes are included in the neighbor list without checking
     *                their actual distance against the cutoff (faster).
     *---------------------------------------------------------------------*/
    template<class N>
    struct BuildNeighborList 
    {
        System* system;
        N* net;
        NeighborBox* neighbox;
        NeighborList* neilist;
        
        bool use_subset = false;
        NeiType input_type;
        Kokkos::View<int*, T_memory_space> ind;
        bool strict = true;
        
        int Nbox = 27;
        double cutoff2;
        
        BuildNeighborList(System* _system, N* _net, NeighborBox* _neighbox, 
                          NeighborList* _neilist, NeiType _type, bool _strict) :
        system(_system), net(_net), neighbox(_neighbox), neilist(_neilist), input_type(_type), strict(_strict) {
            use_subset = false;
            build();
        }
        
        BuildNeighborList(System* _system, N* _net, NeighborBox* _neighbox, NeighborList* _neilist,
                          NeiType _indtype, Kokkos::View<int*, T_memory_space>& _ind) :
        system(_system), net(_net), neighbox(_neighbox), neilist(_neilist), input_type(_indtype), ind(_ind) {
            use_subset = true;
            build();
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const team_handle& team) const {
            int tid = team.team_rank(); // returns a number between 0 and TEAM_SIZE
            int lid = team.league_rank(); // returns a number between 0 and N
            
            int i; // node/seg id
            if (use_subset) {
                i = ind(lid);
            } else {
                i = lid;
            }
            Vec3 p = neighbox->get_node_pos(net, i, input_type);
            Vec3i id = neighbox->find_box_coord(p);
            
            // Count the number of valid neighbors
            int Nneitot = 0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, Nbox), [&] (const int& ib, int& nsum) {
                int nloc = 0;
                double dist2;
                int nei_box_id = neighbox->neighbor_box_index(id, ib);
                if (nei_box_id >= 0) {
                    int Nnei = neighbox->countBox(nei_box_id);
                    if (!strict) {
                        nsum += Nnei;
                    } else {
                        int n = neighbox->boxes(nei_box_id);
                        for (int l = 0; l < Nnei; l++) {
                            neighbox->get_neighbor(net, p, ib, n, dist2);
                            if (dist2 <= cutoff2) nloc++;
                            n = neighbox->nextInBox(n);
                        }
                        nsum += nloc;
                    }
                }
            }, Nneitot);
            team.team_barrier();
            
            if (tid == 0)
                neilist->count(i) = Nneitot;
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (int i, int& psum, bool is_final) const {
            if (is_final) neilist->beg(i) = psum;
            psum += neilist->count(i);
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const int& t) const {
            int ib = t % Nbox; // box id
            int j = t / Nbox; // id
            
            int i; // node/seg id
            if (use_subset) {
                i = ind(j);
            } else {
                i = j;
            }
            int beg = neilist->beg(i);
            
            Vec3 p = neighbox->get_node_pos(net, i, input_type);
            Vec3i id = neighbox->find_box_coord(p);
            
            double dist2;
            int nei_box_id = neighbox->neighbor_box_index(id, ib);
            if (nei_box_id >= 0) {
                int Nnei = neighbox->countBox(nei_box_id);
                int n = neighbox->boxes(nei_box_id);
                for (int l = 0; l < Nnei; l++) {
                    bool add_nei = true;
                    if (strict) {
                        neighbox->get_neighbor(net, p, ib, n, dist2);
                        add_nei = (dist2 <= cutoff2);
                    }
                    if (add_nei) {
                        int idx = Kokkos::atomic_fetch_add(&neilist->count(i), 1);
                        neilist->list(beg+idx) = n;
                    }
                    n = neighbox->nextInBox(n);
                }
            }
        }
        
        void build()
        {
            cutoff2 = neighbox->cutoff * neighbox->cutoff;
            
            int Ntype_local = (input_type == NeiNode) ? net->Nnodes_local : net->Nsegs_local;
            int Nid = Ntype_local;
            if (use_subset) {
                Nid = ind.extent(0);
            }
            
            Kokkos::resize(neilist->count, Ntype_local);
            Kokkos::deep_copy(neilist->count, 0);
            
        #if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
            int TEAM_SIZE = 32;
        #else
            const Kokkos::AUTO_t TEAM_SIZE = Kokkos::AUTO;
        #endif
            Kokkos::parallel_for(Kokkos::TeamPolicy<>(Nid, TEAM_SIZE), *this);
            Kokkos::fence();
            
            Kokkos::resize(neilist->beg, Ntype_local);
            
            int Ntotnei = 0;
            Kokkos::parallel_scan(Ntype_local, *this, Ntotnei);
            Kokkos::fence();
            
            Kokkos::deep_copy(neilist->count, 0);
            Kokkos::resize(neilist->list, Ntotnei);
            
            Kokkos::parallel_for(Nid*Nbox, *this);
            Kokkos::fence();
        }
    };
    
    template<class N>
    NeighborList* build_neighbor_list(System* system, N* net, NeiType idtype, bool strict)
    {
        NeighborList* neilist = exadis_new<NeighborList>();
        BuildNeighborList(system, net, this, neilist, idtype, strict);
        return neilist;
    }
    
    template<class N>
    NeighborList* build_neighbor_list(System* system, N* net, NeiType idtype, 
                                      Kokkos::View<int*, T_memory_space>& id)
    {
        NeighborList* neilist = exadis_new<NeighborList>();
        BuildNeighborList(system, net, this, neilist, idtype, id);
        return neilist;
    }
    
    const char* name() { return "NeighborBox"; }
};

/*---------------------------------------------------------------------------
 *
 *    Function:     generate_neighbor_list()
 *
 *-------------------------------------------------------------------------*/
template<class N>
inline NeighborList* generate_neighbor_list(System* system, N* net, double cutoff, Neighbor::NeiType type, bool strict=true) {
    NeighborBox* neighbox = exadis_new<NeighborBox>(system, net, cutoff, type);
    NeighborList* neilist = neighbox->build_neighbor_list(system, net, type, strict);
    exadis_delete(neighbox);
    return neilist;
}

} // namespace ExaDiS

#endif
