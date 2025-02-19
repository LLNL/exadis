/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  cliParser: utility class to parse command lines inputs
 *
 *  Syntax:
 *        ExaDiS::tools::cliParser parser(argc, argv);
 *         parser.add_argument(TYPE, &ptr, size, "name", "Description");
 *         parser.add_option(OPTIONAL, TYPE, &ptr, size, "-n", "--name", "Description");
 *        parser.parse(VERBOSE);
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_CLI_PARSER_H
#define EXADIS_CLI_PARSER_H

namespace ExaDiS { namespace tools {

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <vector>
    
/*---------------------------------------------------------------------------
 *
 *    Class:        cliParser
 *                  Command line input parser
 *
 *-------------------------------------------------------------------------*/
class cliParser
{
public:
    enum OptionReq {REQUIRED, OPTIONAL};
    enum OptionType {BOOL, INT, FLOAT, DOUBLE, STRING};
    const static inline std::vector<std::string> types = {"<bool>", "<int>", "<float>", "<double>", "<string>"};
    static const int VERBOSE = 1;

private:
    struct Option {
        OptionReq req;
        OptionType type;
        void* ptr;
        int size;
        std::string shortcmd, longcmd;
        std::string description;
        int found = 0;

        Option(OptionType _type, void* _ptr, int _size,
               std::string _name, std::string _description) {
            req = REQUIRED;
            type = _type;
            ptr = _ptr;
            size = _size;
            shortcmd = _name;
            longcmd = _name;
            description = _description;
        }

        Option(OptionReq _req, OptionType _type, void* _ptr, int _size,
               std::string _shortcmd, std::string _longcmd,
               std::string _description) {
            req = _req;
            type = _type;
            ptr = _ptr;
            size = _size;
            shortcmd = _shortcmd;
            longcmd = _longcmd;
            description = _description;
        }
        
        std::string string() {
            std::string m = (size > 1) ? (std::to_string(size)+"*") : "";
            std::string opt = shortcmd + "," + longcmd + " " + m + types[type];
            if (req == OPTIONAL) opt = "[" + opt + "]";
            return opt;
        }
    };

    int argc;
    char** argv;
    std::vector<Option> args;
    std::vector<Option> opts;
    int nreq = 0;

public:
    cliParser(int _argc, char *_argv[]) {
        argc = _argc;
        argv = _argv;
    }

    void add_argument(OptionType type, void* ptr, int size,
                      std::string name, std::string description) {
        args.emplace_back(type, ptr, size, name, description);
        nreq++;
    }

    void add_option(OptionReq req, OptionType type, void* ptr, int size,
                    std::string shortcmd, std::string longcmd,
                    std::string description) {
        opts.emplace_back(req, type, ptr, size, shortcmd, longcmd, description);
        check_option();
        if (req == REQUIRED) nreq++;
    }

    void check_option() {
        if (opts.back().shortcmd.substr(0, 6) == "--help" || 
            opts.back().longcmd.substr(0, 6) == "--help") {
            printf("Error: option --help reserved\n");
            exit(1);
        }
        if (opts.back().shortcmd.substr(0, 1) != "-" || 
            opts.back().longcmd.substr(0, 1) != "-") {
            printf("Error: option %s,%s must start with '-'\n",
            opts.back().shortcmd.c_str(), opts.back().longcmd.c_str());
            exit(1);
        }
        for (int i = 0; i < opts.size()-1; i++) {
            if (opts[i].shortcmd == opts.back().shortcmd ||
                opts[i].longcmd == opts.back().longcmd) {
                printf("Error: ambiguous option %s,%s (already in use)\n", 
                opts.back().shortcmd.c_str(), opts.back().longcmd.c_str());
                exit(1);
            }
        }
    }
    
    void parse(int verbose=0) {
        
        char* prog = argv[0];
        
        if (argc-1 < nreq)
            parser_error(prog, "Error: Not enough input arguments\n");
        int ireq = 0;
        
        for (int i = 0; i < argc-1; i++) {
            
            char *arg = argv[i+1];
            //printf("parsing token %s\n", arg);
            
            if (!strcmp(arg, "--help")) {
                help(prog);
                exit(1);
            }
            
            // First look for options
            int j;
            for (j = 0; j < opts.size(); j++) {
                if (!strcmp(arg, opts[j].shortcmd.c_str()) ||
                    !strcmp(arg, opts[j].longcmd.c_str())) {
                    break;
                }
            }
            if (j < opts.size()) {
                parse_token(false, &opts[j], ++i, prog, verbose);
                continue;
            }
            
            if (!strncmp(arg, "-", 1) || ireq >= args.size())
                parser_error(prog, "Error: Unknown option %s\n", arg);
            
            // Or get the arguments
            parse_token(true, &args[ireq], i, prog, verbose);
            ireq++;
        }
            
        // Check required
        for (int i = 0; i < args.size(); i++) {
            if (!args[i].found)
                parser_error(prog, "Error: argument %s is required\n", args[i].longcmd.c_str());
        }
        for (int i = 0; i < opts.size(); i++) {
            if (opts[i].req == REQUIRED && !opts[i].found)
                parser_error(prog, "Error: option %s is required\n", opts[i].longcmd.c_str());
        }
    }
    
    void parse_token(bool isarg, Option* cmd, int& i, char* prog, int verbose) {
        
        if (i+cmd->size > argc-1) {
            parser_error(prog, "Error: argument %s requires %d value(s)\n", 
            cmd->shortcmd.c_str(), cmd->size);
        }
        
        if (cmd->type == BOOL) {
            try {
                for (int j = 0; j < cmd->size; j++)
                    *((bool*)(cmd->ptr)+j) = (bool)atoi(argv[i+1+j]);
            } catch (std::exception& e) {
                parser_error(prog, "Error: argument %s requires %d boolean value(s)\n", 
                             cmd->longcmd.c_str(), cmd->size);
            }
        } else if (cmd->type == INT) {
            try {
                for (int j = 0; j < cmd->size; j++)
                    *((int*)(cmd->ptr)+j) = atoi(argv[i+1+j]);
            } catch (std::exception& e) {
                parser_error(prog, "Error: argument %s requires %d integer value(s)\n", 
                             cmd->longcmd.c_str(), cmd->size);
            }
        } else if (cmd->type == FLOAT) {
            try {
                for (int j = 0; j < cmd->size; j++)
                    *((float*)(cmd->ptr)+j) = atof(argv[i+1+j]);
            } catch (std::exception& e) {
                parser_error(prog, "Error: argument %s requires %d floating-point value(s)\n", 
                             cmd->longcmd.c_str(), cmd->size);
            }
        } else if (cmd->type == DOUBLE) {
            try {
                for (int j = 0; j < cmd->size; j++)
                    *((double*)(cmd->ptr)+j) = atof(argv[i+1+j]);
            } catch (std::exception& e) {
                parser_error(prog, "Error: argument %s requires %d floating-point value(s)\n", 
                             cmd->longcmd.c_str(), cmd->size);
            }
        } else if (cmd->type == STRING) {
            try {
                for (int j = 0; j < cmd->size; j++)
                    *((std::string*)(cmd->ptr)+j) = std::string(argv[i+1+j]);
            } catch (std::exception& e) {
                parser_error(prog, "Error: argument %s requires %d string value(s)\n", 
                             cmd->longcmd.c_str(), cmd->size);
            }
        } else {
            parser_error(prog, "Error: unknown argument type %d\n", cmd->type);
        }
            
        if (verbose) {
            if (isarg)
                printf("Found argument %s:", cmd->longcmd.c_str());
            else
                printf("Found argument %s,%s:", cmd->shortcmd.c_str(), cmd->longcmd.c_str());
            for (int j = 0; j < cmd->size; j++)
                printf(" %s", argv[i+1+j]);
            printf("\n");
        }
        
        i += cmd->size-1;
        cmd->found = 1;
    }
    
    void list_parsed() {
        
        for (int i = 0; i < args.size(); i++) {
            printf("%s:", args[i].longcmd.c_str());
            if (args[i].type == BOOL) {
                for (int j = 0; j < args[i].size; j++)
                    printf(" %d", *((bool*)(args[i].ptr)+j));
                printf("\n");
            } else if (args[i].type == INT) {
                for (int j = 0; j < args[i].size; j++)
                    printf(" %d", *((int*)(args[i].ptr)+j));
                printf("\n");
            } else if (args[i].type == FLOAT) {
                for (int j = 0; j < args[i].size; j++)
                    printf(" %e", *((float*)(args[i].ptr)+j));
                printf("\n");
            } else if (args[i].type == DOUBLE) {
                for (int j = 0; j < args[i].size; j++)
                    printf(" %e", *((double*)(args[i].ptr)+j));
                printf("\n");
            } else if (args[i].type == STRING) {
                for (int j = 0; j < args[i].size; j++) {
                    std::string s = *((std::string*)(args[i].ptr)+j);
                    printf(" %s", s.c_str());
                }
                printf("\n");
            }
        }
        
        for (int i = 0; i < opts.size(); i++) {
            if (opts[i].req == OPTIONAL && !opts[i].found) continue;
            printf("%s:", opts[i].longcmd.c_str());
            if (opts[i].type == BOOL) {
                for (int j = 0; j < opts[i].size; j++)
                    printf(" %d", *((bool*)(opts[i].ptr)+j));
                printf("\n");
            } else if (opts[i].type == INT) {
                for (int j = 0; j < opts[i].size; j++)
                    printf(" %d", *((int*)(opts[i].ptr)+j));
                printf("\n");
            } else if (opts[i].type == FLOAT) {
                for (int j = 0; j < opts[i].size; j++)
                    printf(" %e", *((float*)(opts[i].ptr)+j));
                printf("\n");
            } else if (opts[i].type == DOUBLE) {
                for (int j = 0; j < opts[i].size; j++)
                    printf(" %e", *((double*)(opts[i].ptr)+j));
                printf("\n");
            } else if (opts[i].type == STRING) {
                for (int j = 0; j < opts[i].size; j++) {
                    std::string s = *((std::string*)(opts[i].ptr)+j);
                    printf(" %s", s.c_str());
                }
                printf("\n");
            }
        }
        
    }
    
    void help(char* prog) {
        
        fprintf(stderr, "\n");
        fprintf(stderr, "Usage: %s", prog);
        for (int i = 0; i < args.size(); i++)
            fprintf(stderr, " %s", args[i].shortcmd.c_str());
        for (int i = 0; i < opts.size(); i++)
            fprintf(stderr, " %s", opts[i].string().c_str());
        fprintf(stderr, "\n\n");
        
        for (int i = 0; i < args.size(); i++) {
            fprintf(stderr, "   %s: %s\n\n", 
            args[i].shortcmd.c_str(), args[i].description.c_str());
        }
        for (int i = 0; i < opts.size(); i++) {
            fprintf(stderr, "   %s: %s\n\n",
            opts[i].string().c_str(), opts[i].description.c_str());
        }
    }
    
    void parser_error(char* prog, const char* format, ...) {
        char msg[512];
        va_list args;
        va_start(args, format);
        vsnprintf(msg, sizeof(msg)-1, format, args);
        msg[sizeof(msg)-1] = 0;
        va_end(args);
        help(prog);
        printf("%s", msg);
        exit(1);
    }
};

} } // namespace ExaDiS::tools

#endif
