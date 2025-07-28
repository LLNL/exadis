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
#include <stack>

namespace ExaDiS {
    
class OpRec; // forward declaration
    
enum {FREE_BOUND, PBC_BOUND};
enum NodeConstraints {UNCONSTRAINED = 0, PINNED_NODE = 7, CORNER_NODE = 1};

/*---------------------------------------------------------------------------
 *
 *    Struct:        NodeTag
 *
 *-------------------------------------------------------------------------*/
struct NodeTag
{
    int domain;
    int index;
    
    NodeTag() = default;
    
    KOKKOS_FORCEINLINE_FUNCTION
    NodeTag(int _domain, int _index) : domain(_domain), index(_index) {}
    
    KOKKOS_INLINE_FUNCTION
    bool operator==(const NodeTag& t) const {
        return (domain == t.domain && index == t.index);
    }
    
    KOKKOS_INLINE_FUNCTION
    bool operator<(const NodeTag& t) const {
        if (domain < t.domain) return 1;
        if (domain > t.domain) return 0;
        return (index < t.index);
    }
};

/*---------------------------------------------------------------------------
 *
 *    Struct:        DisNode
 *
 *-------------------------------------------------------------------------*/
struct DisNode
{
    NodeTag tag;
    //uint8_t flag;
    int constraint;
    Vec3 pos;
    Vec3 f;
    Vec3 v;
    
    DisNode() = default;
    
    KOKKOS_FORCEINLINE_FUNCTION
    DisNode(const NodeTag& _tag, const Vec3& _pos) {
        tag = _tag;
        //flag = 0;
        pos = _pos;
        constraint = UNCONSTRAINED;
        f = v = Vec3(0.0);
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    DisNode(const NodeTag& _tag, const Vec3& _pos, int _constraint) {
        tag = _tag;
        //flag = 0;
        pos = _pos;
        constraint = _constraint;
        f = v = Vec3(0.0);
    }
    
    DisNode(const Vec3& _pos, int _constraint) {
        //flag = 0;
        pos = _pos;
        constraint = _constraint;
        f = v = Vec3(0.0);
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
    Mat33 H, Hinv;
    Vec3 origin;
    
    Cell() = default;
    
    Cell(double Lbox, bool centered=false) {
        xpbc = ypbc = zpbc = PBC_BOUND;
        origin = centered ? Vec3(-0.5*Lbox) : Vec3(0.0);
        set_H(Mat33().diag(Lbox));
    }
    
    Cell(const Vec3& Lvecbox, bool centered=false) {
        xpbc = ypbc = zpbc = PBC_BOUND;
        origin = centered ? -0.5*Lvecbox : Vec3(0.0);
        set_H(Mat33().diag(Lvecbox));
    }
    
    Cell(const Vec3& bmin, const Vec3& bmax) {
        xpbc = ypbc = zpbc = PBC_BOUND;
        origin = Vec3(bmin.x, bmin.y, bmin.z);
        set_H(Mat33().diag(bmax.x-bmin.x, bmax.y-bmin.y, bmax.z-bmin.z));
    }
    
    Cell(const Mat33& _H, const Vec3& _origin, std::vector<int> pbc) {
        xpbc = pbc[0]; ypbc = pbc[1]; zpbc = pbc[2];
        origin = _origin;
        set_H(_H);
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    void set_H(const Mat33& _H) {
        H = _H; // cell column vectors H = [c1|c2|c3]
        Hinv = H.inverse();
    }
    
    KOKKOS_INLINE_FUNCTION
    bool is_triclinic() const {
        return (fabs(H.xy()) > 0.0 || fabs(H.xz()) > 0.0 || fabs(H.yz()) > 0.0 ||
                fabs(H.yx()) > 0.0 || fabs(H.zx()) > 0.0 || fabs(H.zy()) > 0.0);
    }
    
    KOKKOS_INLINE_FUNCTION
    Vec3 center() const {
        return real_position(Vec3(0.5, 0.5, 0.5));
    }
    
    KOKKOS_INLINE_FUNCTION
    Vec3 real_position(const Vec3 &s) const {
        return origin + H * s;
    }
    
    KOKKOS_INLINE_FUNCTION
    Vec3 scaled_position(const Vec3 &p) const {
        return Hinv * (p - origin);
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
            Vec3 Lbox(H.xx(), H.yy(), H.zz());
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
    bool is_inside(const Vec3& r) const {
        Vec3 s = scaled_position(r);
        if      (s.x < 0.0 || s.x > 1.0) return 0;
        else if (s.y < 0.0 || s.y > 1.0) return 0;
        else if (s.z < 0.0 || s.z > 1.0) return 0;
        return 1;
    }
    
    KOKKOS_INLINE_FUNCTION
    double volume() const {
        return fabs(H.det());
    }
    
    std::vector<Vec3> get_cell_vectors() const {
        std::vector<Vec3> vectors = {H.colx(), H.coly(), H.colz()};
        return vectors;
    }
    
    std::vector<Vec3> get_corners() const {
        std::vector<Vec3> c = get_cell_vectors();
        std::vector<Vec3> corners = {
            origin,
            origin + c[0],
            origin + c[1],
            origin + c[2],
            origin + c[0] + c[1],
            origin + c[0] + c[2],
            origin + c[1] + c[2],
            origin + c[0] + c[1] + c[2]
        };
        return corners;
    }
    
    std::vector<Vec3> get_bounds() const {
        std::vector<Vec3> bounds;
        if (is_triclinic()) {
            Vec3 bmin = origin;
            Vec3 bmax = origin;
            std::vector<Vec3> corners = get_corners();
            for (int i = 1; i < 8; i++) {
                bmin.x = fmin(corners[i].x, bmin.x);
                bmin.y = fmin(corners[i].y, bmin.y);
                bmin.z = fmin(corners[i].z, bmin.z);
                bmax.x = fmax(corners[i].x, bmax.x);
                bmax.y = fmax(corners[i].y, bmax.y);
                bmax.z = fmax(corners[i].z, bmax.z);
            }
            bounds = {bmin, bmax};
        } else {
            bounds = {origin, origin + Vec3(H.xx(), H.yy(), H.zz())};
        }
        return bounds;
    }
    
    // Python binding
    std::vector<int> get_pbc();
    std::vector<Vec3> pbc_position_array(std::vector<Vec3>& r0, std::vector<Vec3>& r);
    std::vector<Vec3> pbc_position_array(Vec3& r0, std::vector<Vec3>& r);
    std::vector<Vec3> pbc_fold_array(std::vector<Vec3>& r);
    std::vector<bool> is_inside_array(std::vector<Vec3>& r);
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
    int domain = 0;
    
    Cell cell;
    std::vector<DisNode> nodes;
    std::vector<DisSeg> segs;
    std::vector<Conn> conn;
    
    OpRec* oprec = nullptr;
    
    int maxindex = -1;
    bool recycle = true;
    std::stack<int> recycled_indices;
    inline void set_max_tag(const NodeTag& tag) {
        if (tag.index > maxindex) maxindex = tag.index;
    }
    inline NodeTag get_new_tag() {
        int index = maxindex+1;
        if (recycle && !recycled_indices.empty()) { 
            index = recycled_indices.top();
            recycled_indices.pop();
        }
        NodeTag tag(domain, index);
        set_max_tag(tag);
        return tag;
    }
    inline void free_tag(NodeTag& tag) {
        if (recycle) recycled_indices.push(tag.index);
    }
    void refresh_tags() {
        maxindex = -1;
        if (recycle) recycled_indices = std::stack<int>();
        for (int i = 0; i < number_of_nodes(); i++)
            set_max_tag(nodes[i].tag);
    }
    
    // We need these to avoid accessing STL functions directly from devices
    int Nnodes_local, Nsegs_local;
    DisNode* n_ptr;
    DisSeg* s_ptr;
    Conn* c_ptr;
    
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
    
    SerialDisNet(const Cell& _cell) {
        cell = _cell;
    }
    
    inline int number_of_nodes() { return nodes.size(); }
    inline int number_of_segs() { return segs.size(); }
    
    inline void add_node(const Vec3& pos) { add_node(get_new_tag(), pos); }
    inline void add_node(const Vec3& pos, int constraint) { add_node(get_new_tag(), pos, constraint); }
    inline void add_node(const NodeTag& tag, const Vec3& pos) {
        set_max_tag(tag);
        nodes.emplace_back(tag, pos);
    }
    inline void add_node(const NodeTag& tag, const Vec3& pos, int constraint) {
        set_max_tag(tag);
        nodes.emplace_back(tag, pos, constraint);
    }
    
    inline void add_seg(int n1, int n2, const Vec3& b) { segs.emplace_back(n1, n2, b); }
    inline void add_seg(int n1, int n2, const Vec3& b, const Vec3& p) { segs.emplace_back(n1, n2, b, p); }
    
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
    void move_node(int i, const Vec3& pos, Mat33& dEp);
    
    int split_seg(int i, const Vec3& pos, bool update_conn=true);
    int split_node(int i, std::vector<int>& arms);
    bool merge_nodes(int n1, int n2, Mat33& dEp);
    bool merge_nodes_position(int n1, int n2, const Vec3& pos, Mat33& dEp);
    
    void remove_segs(std::vector<int> seglist);
    void remove_nodes(std::vector<int> nodelist);
    void purge_network();
    
    void update() {
        generate_connectivity();
        update_ptr();
        purge_network();
        refresh_tags();
    }
    
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
    bool sanity_check();
};

} // namespace ExaDiS

#endif
