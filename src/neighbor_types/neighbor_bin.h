/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_NEIGHBOR_BIN_H
#define EXADIS_NEIGHBOR_BIN_H

#include "neighbor.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        NeighborBin
 *
 *-------------------------------------------------------------------------*/
template<class Tnode>
class NeighborBin_t : public Neighbor {
private:
    Cell cell;
    Vec3 cbox[3];
    Vec3 binOrigin;
    int pbc[3];
    int binDim[3];
    int paddingDim[3];
    Mat33 binHinv;
    Mat33 cellHinv;
    std::vector<Tnode*> bins;
    double cutoff, cutoff2;
    bool unique_nei;
    bool free_nodes_on_delete;
    
    void initialize(const Cell &_cell, double _cutoff);
    
    class iterator {
    public:
        iterator(const NeighborBin_t& neighbor, Tnode* node);
        iterator(const NeighborBin_t& neighbor, const Vec3 &pos);
        bool atEnd() const { return dir[0] == _neighbor.paddingDim[0] + 1; }
        Tnode* current() { return _neighborNode; }
        const Vec3& delta() const { return _delta; }
        double dist() const { return sqrt(distsq); }
        Tnode* next();
    protected:
        const NeighborBin_t& _neighbor;
        Vec3 center;
        Tnode* _node;
        int dir[3];
        int centerbin[3];
        int currentbin[3];
        Tnode* binNode;
        Tnode* _neighborNode;
        Vec3 _delta;
        Vec3 cellDelta;
        double distsq;
        bool issamebin;
        bool checkAddNei;
    };
    
public:
    typedef Tnode node_type;
    
    NeighborBin_t(const Cell &cell, double _cutoff, 
                  bool _free_nodes_on_delete=false, bool _add_nei_once=true);
    
    ~NeighborBin_t() {
        if (free_nodes_on_delete) {
            for (int i = 0; i < binDim[0]*binDim[1]*binDim[2]; i++) {
                Tnode *binNode = bins[i];
                while (binNode) {
                    Tnode *delNode = binNode;
                    binNode = binNode->nextInBin;
                    delete delNode;
                }
            }
        }
    }
    
    void insert_node(Tnode* node);
    
    std::vector<int> query(Tnode* node);
    std::vector<std::pair<int,double> > query_distance(Tnode* node);

    std::vector<int> query(const Vec3& pos);
    std::vector<std::pair<int,double> > query_distance(const Vec3& pos);
    
    const char* name() { return "NeighborBin"; }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        NeighborBinNode_t
 *
 *-------------------------------------------------------------------------*/
struct NeighborBinNode_t {
    int index;
    Vec3 pos;
    NeighborBinNode_t* nextInBin;
    NeighborBinNode_t(int i, const Vec3& p) : index(i), pos(p) {}
};

typedef NeighborBin_t<NeighborBinNode_t> NeighborBin;

NeighborBin* generate_neighbor_nodes(SerialDisNet* network, double cutoff);
NeighborBin* generate_neighbor_segs(SerialDisNet* network, double cutoff, double maxseg);

} // namespace ExaDiS

#endif
