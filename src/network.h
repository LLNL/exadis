/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    Nicolas Bertin
 *    bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_NETWORK_H
#define EXADIS_NETWORK_H

#include <Kokkos_Core.hpp>
#include "vec.h"

namespace ExaDiS {
    
enum {FREE_BOUND, PBC_BOUND};
enum NodeConstraints {UNCONSTRAINED = 0, PINNED_NODE = 7, CORNER_NODE = 1};

/*---------------------------------------------------------------------------
 *
 *    Struct:        DisNode
 *
 *-------------------------------------------------------------------------*/
struct DisNode
{
    //uint8_t flag;
    int constraint;
    Vec3 pos;
    Vec3 f;
    Vec3 v;
    
    DisNode() = default;
    
    KOKKOS_FORCEINLINE_FUNCTION
    DisNode(const Vec3& _pos) {
        pos = _pos;
        //flag = 0;
        constraint = UNCONSTRAINED;
        f = v = Vec3(0.0);
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    DisNode(const Vec3& _pos, int _constraint) {
        pos = _pos;
        //flag = 0;
        constraint = _constraint;
        f = v = Vec3(0.0);
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    DisNode(const DisNode& node) {
        pos = node.pos;
        //flag = node.flag;
        constraint = node.constraint;
        f = node.f;
        v = node.v;
    }
};

/*---------------------------------------------------------------------------
 *
 *    Struct:        DisSeg
 *
 *-------------------------------------------------------------------------*/
struct DisSeg 
{
    int n1, n2;
    Vec3 burg;
    Vec3 plane;
    
    DisSeg() = default;
    
    KOKKOS_FORCEINLINE_FUNCTION
    DisSeg(int _n1, int _n2, const Vec3& _burg) {
        n1 = _n1;
        n2 = _n2;
        burg = _burg;
        plane = Vec3(0.0);
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    DisSeg(int _n1, int _n2, const Vec3& _burg, const Vec3& _plane) {
        n1 = _n1;
        n2 = _n2;
        burg = _burg;
        plane = _plane;
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    DisSeg(const DisSeg& seg) {
        n1 = seg.n1;
        n2 = seg.n2;
        burg = seg.burg;
        plane = seg.plane;
    }
};

/*---------------------------------------------------------------------------
 *
 *    Struct:        Conn
 *
 *-------------------------------------------------------------------------*/
#define MAX_CONN 10
struct Conn 
{
    int num;
    int node[MAX_CONN], seg[MAX_CONN], order[MAX_CONN];
    
    KOKKOS_FORCEINLINE_FUNCTION
    Conn() { num = 0; }
    
    KOKKOS_FORCEINLINE_FUNCTION
    bool add_connection(int n, int s, int o) {
        if (num == MAX_CONN) return true;
        node[num] = n;
        seg[num] = s;
        order[num] = o;
        num++;
        return false;
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    bool add_connection(const Conn &conn, int i) {
        if (num == MAX_CONN) return true;
        node[num] = conn.node[i];
        seg[num] = conn.seg[i];
        order[num] = conn.order[i];
        num++;
        return false;
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    void remove_connection(int i) {
        for (int j = i; j < num-1; j++) {
            node[j] = node[j+1];
            seg[j] = seg[j+1];
            order[j] = order[j+1];
        }
        num--;
    }
};

/*---------------------------------------------------------------------------
 *
 *    Struct:        Cell
 *
 *-------------------------------------------------------------------------*/
struct Cell 
{
    int xpbc, ypbc, zpbc;
    double xmin, ymin, zmin;
    double xmax, ymax, zmax;
    Mat33 H, Hinv;
    
    Cell() = default;
    
    KOKKOS_FORCEINLINE_FUNCTION
    Cell(double Lbox) {
        xpbc = ypbc = zpbc = PBC_BOUND;
        xmin = ymin = zmin = 0.0;
        xmax = ymax = zmax = Lbox;
        H = Mat33().diag(Lbox);
        Hinv = H.inverse();
    }
    
    Cell(const Vec3& bmin, const Vec3& bmax) {
        xpbc = ypbc = zpbc = PBC_BOUND;
        xmin = bmin.x; ymin = bmin.y; zmin = bmin.z;
        xmax = bmax.x; ymax = bmax.y; zmax = bmax.z;
        H = Mat33().diag(xmax-xmin, ymax-ymin, zmax-zmin);
        Hinv = H.inverse();
    }
    
    Cell(const Mat33& _H, const Vec3& origin, std::vector<int> pbc) {
        xpbc = pbc[0]; ypbc = pbc[1]; zpbc = pbc[2];
        H = _H;
        Hinv = H.inverse();
        if (is_triclinic()) {
            std::vector<Vec3> corners = {
                origin,
                origin + H.rowx,
                origin + H.rowy,
                origin + H.rowz,
                origin + H.rowx + H.rowy,
                origin + H.rowx + H.rowz,
                origin + H.rowy + H.rowz,
                origin + H.rowx + H.rowy + H.rowz
            };
            xmin = xmax = origin.x;
            ymin = ymax = origin.y;
            zmin = zmax = origin.z;
            for (int i = 1; i < 8; i++) {
                xmin = fmin(corners[i].x, xmin);
                ymin = fmin(corners[i].y, ymin);
                zmin = fmin(corners[i].z, zmin);
                xmax = fmax(corners[i].x, xmax);
                ymax = fmax(corners[i].y, ymax);
                zmax = fmax(corners[i].z, zmax);
            }
        } else {
            Vec3 c = origin + H.rowx + H.rowy + H.rowz;
            xmin = origin.x; ymin = origin.y; zmin = origin.z;
            xmax = c.x; ymax = c.y; zmax = c.z;
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    bool is_triclinic() const {
        return (H.xy() > 0.0 || H.xz() > 0.0 || H.yz() > 0.0 ||
                H.yx() > 0.0 || H.zx() > 0.0 || H.zy() > 0.0);
    }
    
    KOKKOS_INLINE_FUNCTION
    Vec3 origin() const {
        return Vec3(xmin, ymin, zmin);
    }
    
    KOKKOS_INLINE_FUNCTION
    Vec3 center() const {
        return real_position(Vec3(0.5, 0.5, 0.5));
    }
    
    KOKKOS_INLINE_FUNCTION
    Vec3 real_position(const Vec3 &s) const {
        return origin() + H * s;
    }
    
    KOKKOS_INLINE_FUNCTION
    Vec3 scaled_position(const Vec3 &p) const {
        return Hinv * (p - origin());
    }
    
    KOKKOS_INLINE_FUNCTION
    Vec3 pbc_position(const Vec3 &r0, const Vec3 &r) const {
        Vec3 rpbc = r;
        if (is_triclinic()) {
            Vec3 v = r-r0;
            Vec3 w = Hinv * v;
            w.x = (xpbc == PBC_BOUND) ? rint(w.x) : 0.0;
            w.y = (ypbc == PBC_BOUND) ? rint(w.y) : 0.0;
            w.z = (zpbc == PBC_BOUND) ? rint(w.z) : 0.0;
            v = H * w;
            rpbc -= v;
        } else {
            Vec3 Lbox(xmax-xmin, ymax-ymin, zmax-zmin);
            if (xpbc == PBC_BOUND) rpbc.x -= rint((rpbc.x-r0.x)/Lbox.x) * Lbox.x;
            if (ypbc == PBC_BOUND) rpbc.y -= rint((rpbc.y-r0.y)/Lbox.y) * Lbox.y;
            if (zpbc == PBC_BOUND) rpbc.z -= rint((rpbc.z-r0.z)/Lbox.z) * Lbox.z;
        }
        return rpbc;
    }
    
    KOKKOS_INLINE_FUNCTION
    Vec3 pbc_fold(const Vec3 &r) const {
        return pbc_position(center(), r);
    }
    
    KOKKOS_INLINE_FUNCTION
    double bounding_volume() {
        return (xmax-xmin)*(ymax-ymin)*(zmax-zmin);
    }

    KOKKOS_INLINE_FUNCTION
    double triclinic_volume() {
        return fabs(H.det());
    }

    KOKKOS_INLINE_FUNCTION
    double volume() {
        if (is_triclinic())
            return triclinic_volume();
        else
            return bounding_volume();
    }
    
    // Python binding
    std::vector<int> get_pbc();
    std::vector<Vec3> get_bounds();
    std::vector<Vec3> pbc_position_array(std::vector<Vec3>& r0, std::vector<Vec3>& r);
    std::vector<Vec3> pbc_position_array(Vec3& r0, std::vector<Vec3>& r);
    std::vector<Vec3> pbc_fold_array(std::vector<Vec3>& r);
};

/*---------------------------------------------------------------------------
 *
 *    Class:        SerialDisNet
 *                  This class implements a STL-based network data structure 
 *                  intented to be used in a serial execution space.
 *                  It implements basic network manipulation functions, e.g.
 *                  split_node() and merge_nodes(), that can be used to
 *                  perform topological operations easily.
 *
 *-------------------------------------------------------------------------*/
class SerialDisNet {
public:
    typedef typename Kokkos::Serial ExecutionSpace;
    static const char* name() { return "SerialDisNet"; }
    
    Cell cell;
    std::vector<DisNode> nodes;
    std::vector<DisSeg> segs;
    std::vector<Conn> conn;
    
    // We need these to avoid accessing STL functions directly from devices
    int Nnodes_local, Nsegs_local;
    DisNode *n_ptr;
    DisSeg *s_ptr;
    Conn *c_ptr;
    
    KOKKOS_INLINE_FUNCTION DisNode* get_nodes() { return n_ptr; }
    KOKKOS_INLINE_FUNCTION DisSeg* get_segs() { return s_ptr; }
    KOKKOS_INLINE_FUNCTION Conn* get_conn() { return c_ptr; }
    
    inline void update_ptr() {
        Nnodes_local = number_of_nodes();
        Nsegs_local = number_of_segs();
        n_ptr = nodes.data();
        s_ptr = segs.data();
        c_ptr = conn.data();
    }
    
    SerialDisNet() {}
    
    SerialDisNet(double Lbox) {
        cell = Cell(Lbox);
    }
    
    SerialDisNet(const Cell &_cell) {
        cell = _cell;
    }
    
    inline int number_of_nodes() { return nodes.size(); }
    inline int number_of_segs() { return segs.size(); }
    
    inline void add_node(const Vec3 &pos) { nodes.emplace_back(pos); }
    inline void add_node(const Vec3 &pos, int constraint) { nodes.emplace_back(pos, constraint); }
    inline void add_node(const DisNode &node) { nodes.emplace_back(node); }
    inline void add_seg(int n1, int n2, const Vec3 &b) { segs.emplace_back(n1, n2, b); }
    inline void add_seg(int n1, int n2, const Vec3 &b, const Vec3 &p) { segs.emplace_back(n1, n2, b, p); }
    inline void add_seg(const DisSeg &seg) { segs.emplace_back(seg); }
    
    inline int find_connection(int n1, int n2) {
        for (int i = 0; i < conn[n1].num; i++)
            if (conn[n1].node[i] == n2) return i;
        return -1;
    }
    
    inline void generate_connectivity() {
        conn = std::vector<Conn>(nodes.size(), Conn());
        for (int i = 0; i < segs.size(); i++) {
            conn[segs[i].n1].add_connection(segs[i].n2, i,  1);
            conn[segs[i].n2].add_connection(segs[i].n1, i, -1);
        }
    }
    
    inline double seg_length(int i) {
        Vec3 r1 = nodes[segs[i].n1].pos;
        Vec3 r2 = cell.pbc_position(r1, nodes[segs[i].n2].pos);
        return (r2-r1).norm();
    }
    
    bool constrained_node(int i);
    bool discretization_node(int i);
    
    void update_node_plastic_strain(int i, const Vec3& pold, const Vec3& pnew, Mat33& dEp);
    
    int split_seg(int i, const Vec3 &pos, bool update_conn=true);
    int split_node(int i, std::vector<int> arms);
    bool merge_nodes(int n1, int n2, Mat33& dEp);
    bool merge_nodes_position(int n1, int n2, const Vec3 &pos, Mat33& dEp);
    
    void remove_segs(std::vector<int> seglist);
    void remove_nodes(std::vector<int> nodelist);
    void purge_network();
    
    std::vector<std::vector<int> > physical_links();
    
    double dislocation_density(double burgmag);
    void write_data(std::string filename);
    
    struct SaveNode {
        int id;
        int nconn;
        std::vector<DisNode> nodes;
        std::vector<DisSeg> segs;
        std::vector<Conn> conn;
    };
    SaveNode save_node(int i);
    void restore_node(SaveNode& saved_node);
    
    // Python binding
    void set_nodes_array(std::vector<std::vector<double> >& nodes);
    void set_segs_array(std::vector<std::vector<double> >& segs);
    std::vector<std::vector<double> > get_nodes_array();
    std::vector<std::vector<double> > get_segs_array();
};

} // namespace ExaDiS

#endif
