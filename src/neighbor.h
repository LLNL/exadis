/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_NEIGHBOR_H
#define EXADIS_NEIGHBOR_H

#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        Neighbor
 *
 *-------------------------------------------------------------------------*/
class Neighbor {
public:
    enum NeiType {NeiNode, NeiSeg};
    
    Neighbor() {}
    Neighbor(System *system) {}
    virtual ~Neighbor() {}
    virtual const char* name() { return "NeighborNone"; }
};


/*---------------------------------------------------------------------------
 *
 *    Struct:       NeighborList
 *                  Data structure to hold a neighbor list with contiguous
 *                  neighbor access.
 *
 *-------------------------------------------------------------------------*/
struct NeiListAccessor;

struct NeighborList {
    typedef Kokkos::View<int*> T_list;
    T_list beg;
    T_list count;
    T_list list;
    int Ntotnei = 0;
    
    KOKKOS_INLINE_FUNCTION T_list::pointer_type get_count() const { return count.data(); }
    KOKKOS_INLINE_FUNCTION NeiListAccessor get_nei() const;
};

struct NeiListAccessor {
    NeighborList neilist;
    KOKKOS_INLINE_FUNCTION NeiListAccessor(const NeighborList& _neilist) : neilist(_neilist) {}
    KOKKOS_FORCEINLINE_FUNCTION
    int operator()(int i, const int& n) const {
        int beg = neilist.beg(i);
        return neilist.list(beg+n);
    }
};

KOKKOS_INLINE_FUNCTION NeiListAccessor NeighborList::get_nei() const {
    return NeiListAccessor(*this);
}

} // namespace ExaDiS


// Available neighbor types
#include "neighbor_bin.h"
#include "neighbor_box.h"

#endif
