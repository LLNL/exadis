/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_CRYSTAL_H
#define EXADIS_CRYSTAL_H

#include "types.h"

#ifndef EXADIS_SYSTEM_H
namespace ExaDiS { class System; } // forward declaration
#else
#include "system.h"
#endif

namespace ExaDiS {

enum CrystalType {BCC_CRYSTAL, FCC_CRYSTAL, USER_CRYSTAL};

/*---------------------------------------------------------------------------
 *
 *    Struct:       CrystalParams
 *
 *-------------------------------------------------------------------------*/
struct CrystalParams
{
    int type = -1; // crystal type
    int num_bcc_plane_families = -1; // number of plane families for BCC
    Mat33 R = Mat33().eye(); // crystal orientation
    int use_glide_planes = -1;
    int enforce_glide_planes = -1;
    
    void set_crystal_type(std::string);
};

/*---------------------------------------------------------------------------
 *
 *    Struct:       Crystal
 *                  Base structrure to hold crystal structure information
 *                  and helper functions.
 *
 *-------------------------------------------------------------------------*/
struct Crystal : CrystalParams
{
    Mat33 Rinv = Mat33().eye();
    bool use_R = 0;
    
    int num_burgs = 0;
    int num_glissile_burgs = 0;
    int num_planes = 0;
    int num_sys = 0;
    Kokkos::View<Vec3*, T_memory_shared> ref_burgs;
    Kokkos::View<Vec3*, T_memory_shared> ref_planes;
    Kokkos::View<int**, T_memory_shared> ref_sys;
    Kokkos::View<int*, T_memory_shared> planes_per_burg;
    Kokkos::View<int*, T_memory_shared> burg_start_plane;
    
    RandomGenerator random_gen;
    
    Crystal() { 
        type = -1;
        initialize();
    }
    Crystal(int _type) {
        type = _type;
        initialize();
    }
    Crystal(int _type, Mat33 _R) {
        type = _type;
        initialize();
        set_orientation(_R);
    }
    Crystal(const CrystalParams& params) : CrystalParams(params) {
        initialize();
        set_orientation(R);
    }
    
    bool operator!=(const CrystalParams& p) {
        if (type != p.type || R != p.R ||
            (p.use_glide_planes != -1 && use_glide_planes != p.use_glide_planes) ||
            (p.enforce_glide_planes != -1 && enforce_glide_planes != p.enforce_glide_planes) ||
            (p.num_bcc_plane_families != -1 && num_bcc_plane_families != p.num_bcc_plane_families))
            return 1;
        return 0;
    }
    
    void set_orientation(Mat33 _R) {
        R = _R;
        R[0] = R[0].normalized();
        R[1] = R[1].normalized();
        R[2] = R[2].normalized();
        if (fabs(fabs(R.det()) - 1.0) > 1e-10)
            ExaDiS_fatal("Error: improper crystal orientation matrix (det = %e)\n", R.det());
        Rinv = R.inverse();
        use_R = 1;
    }
    
    void set_orientation(Vec3 euler) {
        float cose1 = cos(euler[0]);
        float sine1 = sin(euler[0]);
        float cose2 = cos(euler[1]);
        float sine2 = sin(euler[1]);
        float cose3 = cos(euler[2]);
        float sine3 = sin(euler[2]);
        R[0][0] = cose1*cose3-cose2*sine1*sine3;
        R[0][1] = -cose1*sine3-cose2*cose3*sine1;
        R[0][2] = sine1*sine2;
        R[1][0] = cose3*sine1+cose1*cose2*sine3;
        R[1][1] = cose1*cose2*cose3-sine1*sine3;
        R[1][2] = -cose1*sine2;
        R[2][0] = sine2*sine3;
        R[2][1] = cose3*sine2;
        R[2][2] = cose2;
        Rinv = R.inverse();
        use_R = 1;
    }
    
    void initialize()
    {
        if (type == -1) {
            use_glide_planes = enforce_glide_planes = 0;
            return;
        }
        
        if (type < BCC_CRYSTAL || type > USER_CRYSTAL)
            ExaDiS_fatal("Error: invalid crystal type %d\n", type);
        
        if (use_glide_planes < 0)
            use_glide_planes = (type == BCC_CRYSTAL) ? 0 : 1;
        if (enforce_glide_planes < 0)
            enforce_glide_planes = use_glide_planes;
        
        if (enforce_glide_planes && !use_glide_planes)
            ExaDiS_fatal("Error: cannot use option enforce_glide_planes = 1 with use_glide_planes = 0\n");
        
        if (type == BCC_CRYSTAL) {
            
            // Burgers vectors
            num_glissile_burgs = 4;
            num_burgs = num_glissile_burgs+3;
            Kokkos::resize(ref_burgs, num_burgs);
            // 1/2<111> glissile Burgers
            ref_burgs(0) = Vec3( 1.0, 1.0, 1.0).normalized();
            ref_burgs(1) = Vec3(-1.0, 1.0, 1.0).normalized();
            ref_burgs(2) = Vec3( 1.0,-1.0, 1.0).normalized();
            ref_burgs(3) = Vec3( 1.0, 1.0,-1.0).normalized();
            // <100> junction Burgers
            ref_burgs(4) = 2.0/sqrt(3.0) * Vec3(1.0, 0.0, 0.0);
            ref_burgs(5) = 2.0/sqrt(3.0) * Vec3(0.0, 1.0, 0.0);
            ref_burgs(6) = 2.0/sqrt(3.0) * Vec3(0.0, 0.0, 1.0);
            
            // Habit planes
            if (num_bcc_plane_families <= 0)
                num_bcc_plane_families = 2; // default
            int num_glissile_planes;
            if (num_bcc_plane_families == 1) num_glissile_planes = 3; // only {110} planes
            else if (num_bcc_plane_families == 2) num_glissile_planes = 3+3; // {110} and {112} planes
            else num_glissile_planes = 3+3+6; // {110}, {112}, and {123} planes
            
            num_planes = 4*num_glissile_planes+3*16;
            Kokkos::resize(ref_planes, num_planes);
            Kokkos::resize(planes_per_burg, num_burgs);
            Kokkos::resize(burg_start_plane, num_burgs);
            
            // 1/2<111> Burgers
            for (int i = 0; i < 4; i++) {
                Vec3 b = ref_burgs(i);
                // {110} planes
                ref_planes(i*num_glissile_planes+0) = Vec3(-1.0*b.x, b.y, 0.0).normalized();
                ref_planes(i*num_glissile_planes+1) = Vec3(0.0, -1.0*b.y, b.z).normalized();
                ref_planes(i*num_glissile_planes+2) = Vec3(b.x, 0.0, -1.0*b.z).normalized();
                if (num_glissile_planes > 3) {
                    // {112} planes
                    ref_planes(i*num_glissile_planes+3) = Vec3(-2.0*b.x, b.y, b.z).normalized();
                    ref_planes(i*num_glissile_planes+4) = Vec3(b.x, -2.0*b.y, b.z).normalized();
                    ref_planes(i*num_glissile_planes+5) = Vec3(b.x, b.y, -2.0*b.z).normalized();
                }
                if (num_glissile_planes > 6) {
                    // {123} planes
                    ref_planes(i*num_glissile_planes+6)  = Vec3(-3.0*b.x, 2.0*b.y, b.z).normalized();
                    ref_planes(i*num_glissile_planes+7)  = Vec3(-3.0*b.x, b.y, 2.0*b.z).normalized();
                    ref_planes(i*num_glissile_planes+8)  = Vec3(2.0*b.x, -3.0*b.y, b.z).normalized();
                    ref_planes(i*num_glissile_planes+9)  = Vec3(b.x, -3.0*b.y, 2.0*b.z).normalized();
                    ref_planes(i*num_glissile_planes+10) = Vec3(2.0*b.x, b.y, -3.0*b.z).normalized();
                    ref_planes(i*num_glissile_planes+11) = Vec3(b.x, 2.0*b.y, -3.0*b.z).normalized();
                }
                // Indexing
                planes_per_burg(i) = num_glissile_planes;
                burg_start_plane(i) = i*num_glissile_planes;
            }
            
            // <100> Burgers
            std::vector<Vec3> pref100 = {
                Vec3( 1.0, 0.0, 0.0), Vec3( 0.0, 1.0, 0.0), // {100}
                Vec3( 1.0, 1.0, 0.0), Vec3( 1.0,-1.0, 0.0), // {110}
                Vec3( 2.0, 1.0, 0.0), Vec3( 2.0,-1.0, 0.0), // {210}
                Vec3( 1.0, 2.0, 0.0), Vec3( 1.0,-2.0, 0.0),
                Vec3( 3.0, 1.0, 0.0), Vec3( 3.0,-1.0, 0.0), // {310}
                Vec3( 1.0, 3.0, 0.0), Vec3( 1.0,-3.0, 0.0),
                Vec3( 5.0, 1.0, 0.0), Vec3( 5.0,-1.0, 0.0), // {510}
                Vec3( 1.0, 5.0, 0.0), Vec3( 1.0,-5.0, 0.0),
            };
            for (int i = 0; i < 3; i++) {
                // Indexing
                planes_per_burg(4+i) = 16;
                burg_start_plane(4+i) = 4*num_glissile_planes+i*16;
                // <100> zonal planes
                for (int j = 0; j < 16; j++) {
                    Vec3 pj(0.0);
                    pj[(i+1)%3] = pref100[j].x;
                    pj[(i+2)%3] = pref100[j].y;
                    ref_planes(4*num_glissile_planes+i*16+j) = pj.normalized();
                }
            }
            
            // Slip systems: only register the 1/2<111>{110} systems for now
            // For BCC this is only used in network generatation functions
            num_sys = 12;
            Kokkos::resize(ref_sys, num_sys, 2);
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 3; j++) {
                    ref_sys(i*3+j,0) = i; // Burgers index
                    ref_sys(i*3+j,1) = burg_start_plane(i)+j; // Plane index
                }
            }
            
        } else if (type == FCC_CRYSTAL) {
            
            // Burgers vectors
            num_glissile_burgs = 6;
            num_burgs = num_glissile_burgs+3;
            Kokkos::resize(ref_burgs, num_burgs);
            // 1/2<110> Burgers
            ref_burgs(0) = Vec3( 1.0, 1.0, 0.0).normalized();
            ref_burgs(1) = Vec3( 1.0,-1.0, 0.0).normalized();
            ref_burgs(2) = Vec3( 1.0, 0.0, 1.0).normalized();
            ref_burgs(3) = Vec3( 1.0, 0.0,-1.0).normalized();
            ref_burgs(4) = Vec3( 0.0, 1.0, 1.0).normalized();
            ref_burgs(5) = Vec3( 0.0, 1.0,-1.0).normalized();
            // <100> junction Burgers
            ref_burgs(6) = sqrt(2.0) * Vec3(1.0, 0.0, 0.0);
            ref_burgs(7) = sqrt(2.0) * Vec3(0.0, 1.0, 0.0);
            ref_burgs(8) = sqrt(2.0) * Vec3(0.0, 0.0, 1.0);
            
            // Habit planes
            num_planes = 6*3+3*4;
            Kokkos::resize(ref_planes, num_planes);
            Kokkos::resize(planes_per_burg, num_burgs);
            Kokkos::resize(burg_start_plane, num_burgs);
            
            // 1/2<110> Burgers
            for (int i = 0; i < 6; i++) {
                // Indexing
                planes_per_burg(i) = 3;
                burg_start_plane(i) = i*3;
            }
            // for ref_burgs(0)
            ref_planes(0*3+0) = Vec3( 1.0,-1.0, 1.0).normalized(); /* common glide plane */
            ref_planes(0*3+1) = Vec3(-1.0, 1.0, 1.0).normalized(); /* common glide plane */
            ref_planes(0*3+2) = Vec3( 0.0, 0.0, 1.0).normalized(); /* junction plane */
            // for ref_burgs(1)
            ref_planes(1*3+0) = Vec3( 1.0, 1.0, 1.0).normalized(); /* common glide plane */
            ref_planes(1*3+1) = Vec3( 1.0, 1.0,-1.0).normalized(); /* common glide plane */
            ref_planes(1*3+2) = Vec3( 0.0, 0.0, 1.0).normalized(); /* junction plane */
            // for ref_burgs(2)
            ref_planes(2*3+0) = Vec3( 1.0, 1.0,-1.0).normalized(); /* common glide plane */
            ref_planes(2*3+1) = Vec3(-1.0, 1.0, 1.0).normalized(); /* common glide plane */
            ref_planes(2*3+2) = Vec3( 0.0, 1.0, 0.0).normalized(); /* junction plane */
            // for ref_burgs(3)
            ref_planes(3*3+0) = Vec3( 1.0, 1.0, 1.0).normalized(); /* common glide plane */
            ref_planes(3*3+1) = Vec3( 1.0,-1.0, 1.0).normalized(); /* common glide plane */
            ref_planes(3*3+2) = Vec3( 0.0, 1.0, 0.0).normalized(); /* junction plane */
            // for ref_burgs(4)
            ref_planes(4*3+0) = Vec3( 1.0, 1.0,-1.0).normalized(); /* common glide plane */
            ref_planes(4*3+1) = Vec3( 1.0,-1.0, 1.0).normalized(); /* common glide plane */
            ref_planes(4*3+2) = Vec3( 1.0, 0.0, 0.0).normalized(); /* junction plane */
            // for ref_burgs(5)
            ref_planes(5*3+0) = Vec3( 1.0, 1.0, 1.0).normalized(); /* common glide plane */
            ref_planes(5*3+1) = Vec3(-1.0, 1.0, 1.0).normalized(); /* common glide plane */
            ref_planes(5*3+2) = Vec3( 1.0, 0.0, 0.0).normalized(); /* junction plane */ 
            
            // <100> Burgers
            for (int i = 0; i < 3; i++) {
                // Indexing
                planes_per_burg(6+i) = 4;
                burg_start_plane(6+i) = 6*3+i*4;
            }
            // for ref_burgs(6)
            ref_planes(6*3+0*4+0) = Vec3( 0.0, 1.0, 1.0).normalized(); /* junction plane */
            ref_planes(6*3+0*4+1) = Vec3( 0.0, 1.0,-1.0).normalized(); /* junction plane */
            ref_planes(6*3+0*4+2) = Vec3( 0.0, 1.0, 0.0).normalized(); /* junction plane */
            ref_planes(6*3+0*4+3) = Vec3( 0.0, 0.0, 1.0).normalized(); /* junction plane */
            // for ref_burgs(7)
            ref_planes(6*3+1*4+0) = Vec3( 1.0, 0.0, 1.0).normalized(); /* junction plane */
            ref_planes(6*3+1*4+1) = Vec3( 1.0, 0.0,-1.0).normalized(); /* junction plane */
            ref_planes(6*3+1*4+2) = Vec3( 1.0, 0.0, 0.0).normalized(); /* junction plane */
            ref_planes(6*3+1*4+3) = Vec3( 0.0, 0.0, 1.0).normalized(); /* junction plane */
            // for ref_burgs(8)
            ref_planes(6*3+2*4+0) = Vec3( 1.0, 1.0, 0.0).normalized(); /* junction plane */
            ref_planes(6*3+2*4+1) = Vec3( 1.0,-1.0, 0.0).normalized(); /* junction plane */
            ref_planes(6*3+2*4+2) = Vec3( 0.0, 1.0, 0.0).normalized(); /* junction plane */
            ref_planes(6*3+2*4+3) = Vec3( 1.0, 0.0, 0.0).normalized(); /* junction plane */
            
            // 12 FCC 1/2<110>{111} systems
            num_sys = 12;
            Kokkos::resize(ref_sys, num_sys, 2);
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 2; j++) {
                    ref_sys(i*2+j,0) = i; // Burgers index
                    ref_sys(i*2+j,1) = burg_start_plane(i)+j; // Plane index
                }
            }
            
        } else if (type == USER_CRYSTAL) {
            // Read or import file...
            ExaDiS_fatal("Error: user defined crystal type not implemented yet\n");
        }
        
        // Checks
        for (int i = 0; i < num_burgs; i++) {
            Vec3 b = ref_burgs(i);
            for (int j = 0; j < planes_per_burg(i); j++) {
                int s = burg_start_plane(i);
                Vec3 n = ref_planes(s+j);
                if (dot(b, n) > 1e-5)
                    ExaDiS_fatal("Error: Burgers and plane normals are not orthogonal for crystal type = %d\n", type);
            }
        }
        for (int i = 0; i < num_sys; i++) {
            Vec3 b = ref_burgs(ref_sys(i,0));
            Vec3 n = ref_planes(ref_sys(i,1));
            if (dot(b, n) > 1e-5)
                ExaDiS_fatal("Error: Burgers and plane normals are not orthogonal in crystal type = %d\n", type);
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    int identify_closest_Burgers_index(const Vec3& b) {
        int bid = 0;
        double smax = 0.0;
        Vec3 bn = b.normalized();
        if (use_R) bn = Rinv * bn;
        for (int i = 0; i < num_burgs; i++) {
            double s = fabs(dot(ref_burgs(i).normalized(), bn)); // cosine similarity
            //s -= fabs(ref_burgs(i).norm2()-bn.norm2()); // penalize length difference
            if (s > smax) {
                bid = i;
                smax = s;
            }
        }
        return bid;
    }
    
    KOKKOS_INLINE_FUNCTION
    bool is_crystallographic_plane(Vec3 plane) {
        if (use_R) plane = Rinv * plane;
        for (int i = 0; i < num_planes; i++) {
            Vec3 p = ref_planes(i);
            if (fabs(fabs(dot(p, plane))-1.0) < 1e-5) return 1;
        }
        return 0;
    }
    
    KOKKOS_INLINE_FUNCTION
    Vec3 find_precise_glide_plane(const Vec3& b, const Vec3& t)
    {
        Vec3 plane = cross(b, t).normalized();
        if (plane.norm2() < 1e-2 || fabs(dot(b, t))/b.norm()/t.norm() > 0.995)
            return Vec3(0.0); // screw segment
        
        if (use_glide_planes) {
            int bid = identify_closest_Burgers_index(b);
            if (use_R) plane = Rinv * plane;
            int nid = 0;
            double smax = 0.0;
            for (int i = 0; i < planes_per_burg(bid); i++) {
                Vec3 p = ref_planes(burg_start_plane(bid)+i);
                double s = fabs(dot(p, plane));
                if (s > smax) {
                    nid = i;
                    smax = s;
                }
            }
            Vec3 p = ref_planes(burg_start_plane(bid)+nid);
            if (enforce_glide_planes || fabs(dot(plane, p)) > 0.99) plane = p;
            if (use_R) plane = R * plane;
        }
        return plane;
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    Vec3 pick_screw_glide_plane(N* net, const Vec3& b)
    {
        Vec3 plane(0.0);
        if (use_glide_planes) {
            if (type == FCC_CRYSTAL) {
                double val = random_gen.drand<typename N::ExecutionSpace>(0.0, 1.0);
                if (fabs(b.x) < 1e-10) {
                    plane.x = plane.y = 1.0 / sqrt(3.0);
                    plane.z = -SIGN(b.y * b.z) * plane.y;
                    if (val < 0.5) plane.x = -plane.y;
                } else if (fabs(b.y) < 1e-10) {
                    plane.x = plane.y = 1.0 / sqrt(3.0);
                    plane.z = -SIGN(b.x * b.z) * plane.x;
                    if (val < 0.5) plane.y = -plane.x;
                } else {
                    plane.x = plane.z = 1.0 / sqrt(3.0);
                    plane.y = -SIGN(b.x * b.y) * plane.x;
                    if (val < 0.5) plane.z = -plane.x;
                }
            } else {
                int bid = identify_closest_Burgers_index(b);
                int nid = random_gen.rand<typename N::ExecutionSpace>(0, planes_per_burg(bid));
                plane = ref_planes(burg_start_plane(bid)+nid);
            }
        }
        plane = plane.normalized();
        if (use_R) plane = R * plane;
        return plane;
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    Vec3 find_seg_glide_plane(N* net, int i)
    {
        auto nodes = net->get_nodes();
        auto segs = net->get_segs();
        auto cell = net->cell;
        
        int n1 = segs[i].n1;
        int n2 = segs[i].n2;
        Vec3 b = segs[i].burg;
        Vec3 r1 = nodes[n1].pos;
        Vec3 r2 = cell.pbc_position(r1, nodes[n2].pos);
        Vec3 l = r2-r1;
        return find_precise_glide_plane(b, l);
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    void reset_node_glide_planes(N* net, int i)
    {
        if (!use_glide_planes) return;
        auto segs = net->get_segs();
        auto conn = net->get_conn();
        for (int j = 0; j < conn[i].num; j++) {
            int s = conn[i].seg[j];
            Vec3 p = find_seg_glide_plane(net, s);
            if (p.norm2() > 1e-5) segs[s].plane = p;
        }
    }
};

/*---------------------------------------------------------------------------
 *
 *      Function:   BCC_binary_junction_node
 *                  Returns -1 if this is not a binary junction,
 *                  returns the index of the junction arm otherwise.
 *
 *-------------------------------------------------------------------------*/
template<class N>
KOKKOS_INLINE_FUNCTION
int BCC_binary_junction_node(System* system, N* net, const int& i, 
                             Vec3& tjunc, int* planarjunc)
{
    auto nodes = net->get_nodes();
    auto segs = net->get_segs();
    auto conn = net->get_conn();
    auto cell = net->cell;
    
    int binaryjunc = -1;
    tjunc = Vec3(0.0);
    *planarjunc = 0;

    if (conn[i].num != 3) return -1;

    int numarmglide = 0;
    int numarmedge = 0;
    int numarmjunc = 0;
    Vec3 ndir[3];
    
    double eps = 1e-12;
    double angletol = 5.0*M_PI/180.0; // 5 degrees
    
    Vec3 r1 = nodes[i].pos;
    
    for (int j = 0; j < conn[i].num; j++) {
        
        int k = conn[i].node[j];
        Vec3 r2 = cell.pbc_position(r1, nodes[k].pos);
        Vec3 t = r2-r1;
        double mag2 = t.norm2();
        if (mag2 < eps) continue;
        t = 1.0/sqrt(mag2) * t;
        
        int s = conn[i].seg[j];
        int order = conn[i].order[j];
        Vec3 burg = order*segs[s].burg;
        double bmag = burg.norm();

        if (bmag > 1.0+eps) {
            tjunc = t;
            binaryjunc = j;
            numarmjunc++;
        } else {
            Vec3 n = cross(burg, t);
            if (fabs(dot(n, n)) > 1.0e-2)
                ndir[numarmedge++] = n;
            numarmglide++;
        }
    }
    
    if (numarmjunc == 1 && numarmglide == 2) {
        // Determine if the junction is planar, i.e. if it
        // belongs to the intersection of its parent planes.
        if (numarmedge == 2) {
            // If both arms are edge, check if the junction
            // is contained in the intersection of both planes
            double tol = 1.0-cos(angletol); // 5 degrees
            Vec3 tdir = cross(ndir[0], ndir[1]).normalized();
            *planarjunc = is_collinear(tdir, tjunc, tol);
        } else if (numarmedge == 1) {
            // If only one of the arm is edge, check that
            // the junction is contained in the plane
            *planarjunc = (fabs(dot(ndir[0], tjunc)) < sin(angletol));
        } else {
            // Both arms are screw, so the junction is
            // co-planar by convention
            *planarjunc = 1;
        }
    } else {
        binaryjunc = -1;
    }
    
    return binaryjunc;
}

} // namespace ExaDiS

#endif
