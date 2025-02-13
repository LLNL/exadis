/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  OpRec module
 *
 *  This module implements functions to record all network / topolgical
 *  operations that are performed during the course of a simulation.
 *  Using this data, a player implemented in the driver class can re-play 
 *  a simulation from previously saved OpRec files, bypassing all
 *  computations.
 *
 *  Note: The OpRec output option (from the driver) uses 3 frequencies:
 *      - oprecwritefreq: frequency at which the list of operations stored
 *        in memory is dumped to the current oprec file (oprec.XXX.exadis).
 *      - oprecfilefreq: frequency at which new OpRec files are being
 *        created to dump further operations.
 *      - oprecposfreq: frequency at which nodal motion is recorded.
 *        A value of 1 means that the exact trajectory is saved.
 *        A value > 1 means that node positions are only saved every 
 *        oprecposfreq frame.
 *        A value of 0 means that only topological operations are recorded.
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_OPREC_H
#define EXADIS_OPREC_H

#include "vec.h"

#define OPREC_VERSION "1.0"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Struct:       OpRec
 *
 *-------------------------------------------------------------------------*/
struct OpRec {
    
    enum OpTypes {
        TIME_INTEGRATE,
        PLASTIC_STRAIN,
        MOVE_NODE,
        SPLIT_SEG,
        MERGE_NODES,
        SPLIT_MULTI_NODE,
        UPDATE_SEG_PLANE,
        PURGE_NETWORK,
        UPDATE_OUTPUT
    };
    
    struct TimeIntegrate {};
    struct PlasticStrain {};
    struct MoveNode {};
    struct SplitSeg {};
    struct MergeNodes {};
    struct SplitMultiNode {};
    struct UpdateSegPlane {};
    struct PurgeNetwork {};
    struct UpdateOutput {};
    
    struct Op {
        int optype = -1;
        int i1, i2;
        double d1;
        Vec3 v1, v2;
        
        inline Op(int _optype) : optype(_optype) {}
        Op(int _optype, char* line);
        void write(FILE* fp);
    };
    
    bool record = false;
    std::vector<Op> ops;
    int iop = -1;
    int filecounter = 0;
    
    void activate() { record = true; }
    void deactivate() { record = false; }
    
    inline void add_op(const TimeIntegrate, bool rec_pos, double dt) {
        if (!record) return;
        Op op(TIME_INTEGRATE);
        op.i1 = (int)rec_pos;
        op.d1 = dt;
        ops.push_back(op);
    }
    
    inline void add_op(const PlasticStrain, const Mat33& dEp, const Mat33& dWp, double density) {
        if (!record) return;
        Op op(PLASTIC_STRAIN);
        op.d1 = density;
        op.v1 = Vec3(dEp.xx(), dEp.yy(), dEp.zz());
        op.v2 = Vec3(dEp.xy(), dEp.xz(), dEp.yz());
        ops.push_back(op);
        op.d1 = 0.0;
        op.v1 = Vec3(dWp.xx(), dWp.yy(), dWp.zz());
        op.v2 = Vec3(dWp.xy(), dWp.xz(), dWp.yz());
        ops.push_back(op);
    }
    
    inline void add_op(const MoveNode, int i, const Vec3& pos) {
        if (!record) return;
        Op op(MOVE_NODE);
        op.i1 = i;
        op.v1 = pos;
        ops.push_back(op);
    }
    
    inline void add_op(const SplitSeg, int i, const Vec3& pos) {
        if (!record) return;
        Op op(SPLIT_SEG);
        op.i1 = i;
        op.v1 = pos;
        ops.push_back(op);
    }
    
    inline void add_op(const MergeNodes, int n1, int n2, const Vec3& pos) {
        if (!record) return;
        Op op(MERGE_NODES);
        op.i1 = n1;
        op.i2 = n2;
        op.v1 = pos;
        ops.push_back(op);
    }
    
    inline void add_op(const SplitMultiNode, int i, int kmax, const Vec3& p0, const Vec3& p1) {
        if (!record) return;
        Op op(SPLIT_MULTI_NODE);
        op.i1 = i;
        op.i2 = kmax;
        op.v1 = p0;
        op.v2 = p1;
        ops.push_back(op);
    }
    
    inline void add_op(const UpdateSegPlane, int i, const Vec3& plane) {
        if (!record) return;
        Op op(UPDATE_SEG_PLANE);
        op.i1 = i;
        op.v1 = plane;
        ops.push_back(op);
    }
    
    inline void add_op(const PurgeNetwork) {
        if (!record) return;
        Op op(PURGE_NETWORK);
        ops.push_back(op);
    }
    
    inline void add_op(const UpdateOutput) {
        if (!record) return;
        Op op(UPDATE_OUTPUT);
        ops.push_back(op);
    }
    
    void clear() { 
        ops.clear();
        iop = -1;
    }
    
    bool step() {
        return (++iop < ops.size());
    }
    
    Op* current() {
        return &ops[iop];
    }
    
    Op* iterate() {
        if (!step()) {
            printf("Error: oprec reached end of op list\n");
            exit(1);
        }
        return &ops[iop];
    }
    
    void write_file(std::string oprec_file);
    void read_file(std::string oprec_file);
};

} // namespace ExaDiS

#endif
