/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_FORCE_LT_H
#define EXADIS_FORCE_LT_H

#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Struct:   ForceSegLT: 
 *              Line-tension force kernel that includes:
 *              1) core force coming from a core model <C>
 *              2) PK force from applied stress
 *              WARNING: it does not account for the non-singular self-force
 *              unless it is explicitely instantiated with selfforce = true
 *
 *-------------------------------------------------------------------------*/
template<class C, bool selfforce>
struct ForceSegLT
{
    static const bool has_pre_compute = false;
    static const bool has_compute_team = false;
    static const bool has_node_force = false;
    
    typedef typename C::Params Params;
    
    C *core;
    double MU, NU, a;
    
    ForceSegLT(System *system, Params &params)
    {
        MU = system->params.MU;
        NU = system->params.NU;
        a = system->params.a;
        
        core = exadis_new<C>(system, params);
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    SegForce segment_force(System *system, N *net, const int &i) 
    {
        auto nodes = net->get_nodes();
        auto segs = net->get_segs();
        auto cell = net->cell;
        
        int n1 = segs[i].n1;
        int n2 = segs[i].n2;
        Vec3 b = segs[i].burg;
        Vec3 r1 = nodes[n1].pos;
        Vec3 r2 = cell.pbc_position(r1, nodes[n2].pos);
        
        Vec3 t = r2-r1;
        double L = t.norm();
        if (L < 1e-10) return SegForce();
        t = t.normalized();

        // Core-force
        Vec3 fsf = core->core_force(b, t);
        
        // Self-force
        if (selfforce)
            fsf += self_force(b, t, L, MU, NU, a);
        
        // External PK force
        Vec3 fpk = pk_force(b, r1, r2, system->extstress);
        
        Vec3 f1 = +1.0*fsf + fpk;
        Vec3 f2 = -1.0*fsf + fpk;
        
        return SegForce(f1, f2);
    }
    
    ~ForceSegLT() {
        exadis_delete(core);
    }
    
    static constexpr const char* name = "ForceSegLT";
};

namespace ForceType {
    typedef ForceSeg<ForceSegLT<CoreDefault,false> > LINE_TENSION_MODEL;
    typedef ForceSeg<ForceSegLT<CoreDefault,true> > CORE_SELF_PKEXT;
    typedef ForceSeg<ForceSegLT<CoreConstant,true> > CORECONST_SELF_PKEXT;
    typedef ForceSeg<ForceSegLT<CoreMD,true> > COREMD_SELF_PKEXT;
}

} // namespace ExaDiS

#endif
