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

#include "network.h"
#include <any>
#include <sstream>

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
    
    typedef std::any OpAny;
    
    struct TimeIntegrate {
        int rec_pos; double dt;
        TimeIntegrate(int _rec_pos, double _dt) : rec_pos(_rec_pos), dt(_dt) {}
        TimeIntegrate(char* line) {
            int type;
            sscanf(line, "%d %d %lf",
                   &type, &rec_pos, &dt);
        }
        inline void write(FILE* fp) {
            fprintf(fp, "%d %d %e\n",
                    TIME_INTEGRATE, rec_pos, dt);
        }
    };
    
    struct PlasticStrain {
        double density; Mat33 dEp; Mat33 dWp;
        PlasticStrain(const Mat33& _dEp, const Mat33& _dWp, double _density) :
        dEp(_dEp), dWp(_dWp), density(_density) {}
        PlasticStrain(char* line) {
            int type;
            sscanf(line, "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &type, &density,
                   &dEp[0][0], &dEp[0][1], &dEp[0][2],
                   &dEp[1][0], &dEp[1][1], &dEp[1][2],
                   &dEp[2][0], &dEp[2][1], &dEp[2][2],
                   &dWp[0][0], &dWp[0][1], &dWp[0][2],
                   &dWp[1][0], &dWp[1][1], &dWp[1][2],
                   &dWp[2][0], &dWp[2][1], &dEp[2][2]);
        }
        inline void write(FILE* fp) {
            fprintf(fp, "%d %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n",
                    PLASTIC_STRAIN, density,
                    dEp.xx(), dEp.xy(), dEp.xz(),
                    dEp.yx(), dEp.yy(), dEp.yz(),
                    dEp.zx(), dEp.zy(), dEp.zz(),
                    dWp.xx(), dWp.xy(), dWp.xz(),
                    dWp.yx(), dWp.yy(), dWp.yz(),
                    dWp.zx(), dWp.zy(), dWp.zz());
        }
    };
    
    struct MoveNode {
        NodeTag tag; Vec3 pos;
        MoveNode(const NodeTag& _tag, const Vec3& _pos) : tag(_tag), pos(_pos) {}
        MoveNode(char* line) {
            int type;
            sscanf(line, "%d %d %d %lf %lf %lf",
                   &type, &tag.domain, &tag.index, &pos.x, &pos.y, &pos.z);
        }
        inline void write(FILE* fp) {
            fprintf(fp, "%d %d %d %e %e %e\n",
                    MOVE_NODE, tag.domain, tag.index, pos.x, pos.y, pos.z);
        }
    };
    
    struct SplitSeg {
        NodeTag tag1; NodeTag tag2; Vec3 pos; NodeTag tagnew;
        SplitSeg(const NodeTag& _tag1, const NodeTag& _tag2, const Vec3& _pos, const NodeTag& _tagnew) :
        tag1(_tag1), tag2(_tag2), pos(_pos), tagnew(_tagnew) {}
        SplitSeg(char* line) {
            int type;
            sscanf(line, "%d %d %d %d %d %lf %lf %lf %d %d",
                   &type, &tag1.domain, &tag1.index, &tag2.domain, &tag2.index,
                   &pos.x, &pos.y, &pos.z, &tagnew.domain, &tagnew.index);
        }
        inline void write(FILE* fp) {
            fprintf(fp, "%d %d %d %d %d %e %e %e %d %d\n",
                    SPLIT_SEG, tag1.domain, tag1.index, tag2.domain, tag2.index,
                    pos.x, pos.y, pos.z, tagnew.domain, tagnew.index);
        }
    };
    
    struct MergeNodes {
        NodeTag tag1; NodeTag tag2; Vec3 pos;
        MergeNodes(const NodeTag& _tag1, const NodeTag& _tag2, const Vec3& _pos) :
        tag1(_tag1), tag2(_tag2), pos(_pos) {}
        MergeNodes(char* line) {
            int type;
            sscanf(line, "%d %d %d %d %d %lf %lf %lf",
                   &type, &tag1.domain, &tag1.index, &tag2.domain, &tag2.index,
                   &pos.x, &pos.y, &pos.z);
        }
        inline void write(FILE* fp) {
            fprintf(fp, "%d %d %d %d %d %e %e %e\n",
                    MERGE_NODES, tag1.domain, tag1.index, tag2.domain, tag2.index,
                    pos.x, pos.y, pos.z);
        }
    };
    
    struct SplitMultiNode {
        NodeTag tag; std::vector<NodeTag> tagarms; Vec3 p0; Vec3 p1; NodeTag tagnew;
        SplitMultiNode(const NodeTag& _tag, const std::vector<NodeTag>& _tagarms,
                       const Vec3& _p0, const Vec3& _p1, const NodeTag& _tagnew) :
        tag(_tag), tagarms(_tagarms), p0(_p0), p1(_p1), tagnew(_tagnew) {}
        SplitMultiNode(char* line) {
            int type, size;
            std::istringstream sline(line);
            sline >> type >> tag.domain >> tag.index >> size;
            for (int i = 0; i < size; i++) {
                NodeTag t;
                sline >> t.domain >> t.index;
                tagarms.push_back(t);
            }
            sline >> p0.x >> p0.y >> p0.z >> p1.x >> p1.y >> p1.z >> tagnew.domain >> tagnew.index;
        }
        inline void write(FILE* fp) {
            fprintf(fp, "%d %d %d %d",
                    SPLIT_MULTI_NODE, tag.domain, tag.index, (int)tagarms.size());
            for (const auto& t : tagarms)
                fprintf(fp, " %d %d", t.domain, t.index);
            fprintf(fp, " %e %e %e %e %e %e %d %d\n",
                    p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, tagnew.domain, tagnew.index);
        }
    };
    
    struct UpdateSegPlane {
        NodeTag tag1; NodeTag tag2; Vec3 plane;
        UpdateSegPlane(const NodeTag& _tag1, const NodeTag& _tag2, const Vec3& _plane) :
        tag1(_tag1), tag2(_tag2), plane(_plane) {}
        UpdateSegPlane(char* line) {
            int type;
            sscanf(line, "%d %d %d %d %d %lf %lf %lf",
                   &type, &tag1.domain, &tag1.index, &tag2.domain, &tag2.index,
                   &plane.x, &plane.y, &plane.z);
        }
        inline void write(FILE* fp) {
            fprintf(fp, "%d %d %d %d %d %e %e %e\n",
                    UPDATE_SEG_PLANE, tag1.domain, tag1.index, tag2.domain, tag2.index,
                    plane.x, plane.y, plane.z);
        }
    };
    
    struct PurgeNetwork {
        PurgeNetwork() {}
        PurgeNetwork(char* line) {}
        inline void write(FILE* fp) {
            fprintf(fp, "%d\n",
                    PURGE_NETWORK);
        }
    };
    
    struct UpdateOutput {
        UpdateOutput() {}
        UpdateOutput(char* line) {}
        inline void write(FILE* fp) {
            fprintf(fp, "%d\n",
                    UPDATE_OUTPUT);
        }
    };
    
    
    std::vector<OpAny> ops;
    int iop = -1;
    int filecounter = 0;
    bool record = false;
    
    void activate() { record = true; }
    void deactivate() { record = false; }
    
    template<typename T>
    inline void add_op(const T& op) {
        if (!record) return;
        ops.push_back(op);
    }
    
    void clear() { 
        ops.clear();
        iop = -1;
    }
    
    bool step() {
        return (++iop < ops.size());
    }
    
    OpAny* current() {
        return &ops[iop];
    }
    
    OpAny* iterate() {
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
