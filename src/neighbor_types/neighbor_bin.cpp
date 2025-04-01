/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "neighbor_bin.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Function:        NeighborBin_t::NeighborBin_t()
 *
 *-------------------------------------------------------------------------*/
template<class Tnode>
NeighborBin_t<Tnode>::NeighborBin_t(const Cell &_cell, double _cutoff, 
                                    bool _free_nodes_on_delete,
                                    bool _unique_nei)
{
    unique_nei = _unique_nei;
    initialize(_cell, _cutoff);
    free_nodes_on_delete = _free_nodes_on_delete;
}

/*---------------------------------------------------------------------------
 *
 *    Function:        NeighborBin_t::initialize()
 *
 *-------------------------------------------------------------------------*/
template<class Tnode>
void NeighborBin_t<Tnode>::initialize(const Cell &_cell, double _cutoff)
{
    cell = _cell;
    cutoff = _cutoff;
    cutoff2 = _cutoff * _cutoff;

    binOrigin = cell.origin;
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
    for (size_t i = 0; i < 3; i++) {
        binDim[i] = (int)floor(fabs(dot(cbox[i], perpVecs[i])) / cutoff);
        binDim[i] = MIN(binDim[i], 50);
        binDim[i] = MAX(binDim[i], 1);

        for (size_t j = 0; j < 3; j++)
            binH[j][i] = cbox[i][j] / (double)binDim[i];

        if (pbc[i] == PBC_BOUND) {
            paddingDim[i] = (int)ceil(cutoff / fabs(dot(cbox[i], perpVecs[i])));
            if (unique_nei) paddingDim[i] = 1;
        }
        else
            paddingDim[i] = 1;
    }
    binHinv = binH.inverse();

    // Resize bin array
    bins.resize(binDim[0] * binDim[1] * binDim[2]);
}

/*---------------------------------------------------------------------------
 *
 *    Function:        NeighborBin_t::insert_node()
 *
 *-------------------------------------------------------------------------*/
template<class Tnode>
void NeighborBin_t<Tnode>::insert_node(Tnode *node)
{
    // Determine the bin in which the node is located
    Vec3 rp = binHinv * (node->pos - binOrigin);
    int binCoord[3];
    for (int k = 0; k < 3; k++) {
        binCoord[k] = (int)floor(rp[k]);
        if (pbc[k] == PBC_BOUND) {
            while(binCoord[k] < 0) binCoord[k] += binDim[k];
            while(binCoord[k] >= binDim[k]) binCoord[k] -= binDim[k];
        }
        else {
            binCoord[k] = MAX(MIN(binCoord[k], binDim[k]-1), 0);
        }
    }
    int binIndex = binCoord[2] * binDim[0] * binDim[1] +
                   binCoord[1] * binDim[0] + binCoord[0];

    // Insert node into linked list of bin
    node->nextInBin = bins[binIndex];
    //node->binIndex = binIndex;
    bins[binIndex] = &*node;
}

/*---------------------------------------------------------------------------
 *
 *    Function:        NeighborBin_t::iterator::iterator()
 *
 *-------------------------------------------------------------------------*/
template<class Tnode>
NeighborBin_t<Tnode>::iterator::iterator(const NeighborBin_t& neighbor, Tnode* node) : _neighbor(neighbor), _node(node)
{
    dir[0] = -_neighbor.paddingDim[0] - 1;
    dir[1] = _neighbor.paddingDim[1];
    dir[2] = _neighbor.paddingDim[2];
    binNode = NULL;
    center = node->pos;
    _neighborNode = NULL;
    
    checkAddNei = _neighbor.unique_nei && (_neighbor.binDim[0] == 1 || _neighbor.binDim[1] == 1 || _neighbor.binDim[2] == 1);

    // Determine the bin the central node is located in
    Vec3 reducedp = _neighbor.binHinv * (center - _neighbor.binOrigin);
    for (int k = 0; k < 3; k++) {
        centerbin[k] = (int)floor(reducedp[k]);
        if (_neighbor.pbc[k] == PBC_BOUND) {
            while(centerbin[k] < 0) centerbin[k] += _neighbor.binDim[k];
            while(centerbin[k] >= _neighbor.binDim[k]) centerbin[k] -= _neighbor.binDim[k];
        } else {
            centerbin[k] = MAX(MIN(centerbin[k], _neighbor.binDim[k]-1), 0);
        }
    }
    next();
}

/*---------------------------------------------------------------------------
 *
 *    Function:        NeighborBin_t::iterator::iterator()
 *
 *-------------------------------------------------------------------------*/
template<class Tnode>
NeighborBin_t<Tnode>::iterator::iterator(const NeighborBin_t& neighbor, const Vec3 &pos) : _neighbor(neighbor), _node(NULL)
{
    dir[0] = -_neighbor.paddingDim[0] - 1;
    dir[1] = _neighbor.paddingDim[1];
    dir[2] = _neighbor.paddingDim[2];
    binNode = NULL;
    center = pos;
    _neighborNode = NULL;
    
    checkAddNei = _neighbor.unique_nei && (_neighbor.binDim[0] == 1 || _neighbor.binDim[1] == 1 || _neighbor.binDim[2] == 1);

    // Determine the bin the central node is located in
    Vec3 reducedp = _neighbor.binHinv * (center - _neighbor.binOrigin);
    for (int k = 0; k < 3; k++) {
        centerbin[k] = (int)floor(reducedp[k]);
        if (_neighbor.pbc[k] == PBC_BOUND) {
            while(centerbin[k] < 0) centerbin[k] += _neighbor.binDim[k];
            while(centerbin[k] >= _neighbor.binDim[k]) centerbin[k] -= _neighbor.binDim[k];
        } else {
            centerbin[k] = MAX(MIN(centerbin[k], _neighbor.binDim[k]-1), 0);
        }
    }
    next();
}

/*---------------------------------------------------------------------------
 *
 *    Function:        NeighborBin_t::iterator::next()
 *
 *-------------------------------------------------------------------------*/
template<class Tnode>
Tnode* NeighborBin_t<Tnode>::iterator::next()
{
    while (dir[0] != _neighbor.paddingDim[0] + 1) {
        while (binNode) {
            _neighborNode = binNode;
            binNode = binNode->nextInBin;
            _delta = _neighborNode->pos - center + cellDelta;
            distsq = _delta.norm2();
            bool addNeighbor = 1;
            if (checkAddNei) {
                Vec3 rd = _neighbor.cellHinv * _delta;
                addNeighbor = (fabs(rd.x) < 0.5 && fabs(rd.y) < 0.5 && fabs(rd.z) < 0.5);
            }
            if (distsq <= _neighbor.cutoff2 && addNeighbor /*&& (!issamebin || _neighborNode != _node)*/) // return self
                return _neighborNode;
        }
        if (dir[2] == _neighbor.paddingDim[2]) {
            dir[2] = -_neighbor.paddingDim[2];
            if (dir[1] == _neighbor.paddingDim[1]) {
                dir[1] = -_neighbor.paddingDim[1];
                if (dir[0] == _neighbor.paddingDim[0]) {
                    dir[0] = _neighbor.paddingDim[0] + 1;
                    _neighborNode = NULL;
                    return NULL;
                }
                else dir[0]++;
            }
            else dir[1]++;
        }
        else dir[2]++;

        int k;
        cellDelta = Vec3(0.0);
        issamebin = dir[0] == 0 && dir[1] == 0 && dir[2] == 0;
        for (k = 0; k < 3; k++) {
            currentbin[k] = centerbin[k] + dir[k];
            if (_neighbor.pbc[k] == PBC_BOUND) {
                if (currentbin[k] < 0) {
                    cellDelta += currentbin[k] * _neighbor.cbox[k];
                    currentbin[k] = _neighbor.binDim[k]-1;
                } else if (currentbin[k] >= _neighbor.binDim[k]) {
                    cellDelta += (currentbin[k] - _neighbor.binDim[k] + 1) * _neighbor.cbox[k];
                    currentbin[k] = 0;
                }
            } else {
                if (currentbin[k] < 0) { break; }
                else if (currentbin[k] >= _neighbor.binDim[k]) { break; }
            }
        }
        if(k != 3) continue;

        int binIndex = currentbin[2] * _neighbor.binDim[0] * _neighbor.binDim[1] + currentbin[1] * _neighbor.binDim[0] + currentbin[0];
        //assert(binIndex < _neighbor.bins.size() && binIndex >= 0);
        binNode = _neighbor.bins[binIndex];
    }
    _neighborNode = NULL;
    return NULL;
}

/*---------------------------------------------------------------------------
 *
 *    Function:        NeighborBin_t::query()
 *
 *-------------------------------------------------------------------------*/
template<class Tnode>
std::vector<int> NeighborBin_t<Tnode>::query(Tnode* node)
{
    std::vector<int> list;
    for (NeighborBin_t::iterator niter(*this, node); !niter.atEnd(); niter.next()) {
        list.push_back(niter.current()->index);
    }
    return list;
}

/*---------------------------------------------------------------------------
 *
 *    Function:        NeighborBin_t::query()
 *
 *-------------------------------------------------------------------------*/
template<class Tnode>
std::vector<int> NeighborBin_t<Tnode>::query(const Vec3& pos)
{
    std::vector<int> list;
    for (NeighborBin_t::iterator niter(*this, pos); !niter.atEnd(); niter.next()) {
        list.push_back(niter.current()->index);
    }
    return list;
}

/*---------------------------------------------------------------------------
 *
 *    Function:        NeighborBin_t::query_distance()
 *
 *-------------------------------------------------------------------------*/
template<class Tnode>
std::vector<std::pair<int,double> > NeighborBin_t<Tnode>::query_distance(Tnode* node)
{
    std::vector<std::pair<int,double> > list;
    for (NeighborBin_t::iterator niter(*this, node); !niter.atEnd(); niter.next()) {
        list.push_back(std::make_pair(niter.current()->index, niter.dist()));
    }
    return list;
}

/*---------------------------------------------------------------------------
 *
 *    Function:        NeighborBin_t::query_distance()
 *
 *-------------------------------------------------------------------------*/
template<class Tnode>
std::vector<std::pair<int,double> > NeighborBin_t<Tnode>::query_distance(const Vec3& pos)
{
    std::vector<std::pair<int,double> > list;
    for (NeighborBin_t::iterator niter(*this, pos); !niter.atEnd(); niter.next()) {
        list.push_back(std::make_pair(niter.current()->index, niter.dist()));
    }
    return list;
}

/*---------------------------------------------------------------------------
 *
 *    Function:        generate_neighbor_nodes
 *
 *-------------------------------------------------------------------------*/
NeighborBin* generate_neighbor_nodes(SerialDisNet* network, double cutoff)
{
    NeighborBin* neighbor = new NeighborBin(network->cell, cutoff, true);
    // Sort nodes into the 3d bin grid
    for (int i = 0; i < network->number_of_nodes(); i++) {
        Vec3 r = network->cell.pbc_fold(network->nodes[i].pos);
        NeighborBinNode_t* node = new NeighborBinNode_t(i, r);
        neighbor->insert_node(node);
    }
    return neighbor;
}

/*---------------------------------------------------------------------------
 *
 *    Function:        generate_neighbor_segs
 *
 *-------------------------------------------------------------------------*/
NeighborBin* generate_neighbor_segs(SerialDisNet* network, double cutoff, double maxseg)
{
    // For segs we use the mid point for binning, so we need to increase
    // the cutoff to make sure that all neighboring segments will be found
    cutoff = cutoff + maxseg;
    NeighborBin* neighbor = new NeighborBin(network->cell, cutoff, true);
    // Sort segs into the 3d bin grid
    for (int i = 0; i < network->number_of_segs(); i++) {
        int n1 = network->segs[i].n1;
        int n2 = network->segs[i].n2;
        Vec3 r1 = network->nodes[n1].pos;
        Vec3 r2 = network->cell.pbc_position(r1, network->nodes[n2].pos);
        Vec3 rmid = network->cell.pbc_fold(0.5 * (r1 + r2));
        NeighborBinNode_t* node = new NeighborBinNode_t(i, rmid);
        neighbor->insert_node(node);
    }
    return neighbor;
}

template std::vector<int> NeighborBin::query(const Vec3&);
template std::vector<std::pair<int,double> > NeighborBin::query_distance(const Vec3&);

} // namespace ExaDiS
