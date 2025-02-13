/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "system.h"

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *	Function:	fatal
 *
 *-------------------------------------------------------------------------*/
template <unsigned int error>
void print_(const char *format, ...)
{
    char msg[512];
    va_list args;
    va_start(args, format);
    vsnprintf(msg, sizeof(msg)-1, format, args);
    msg[sizeof(msg)-1] = 0;
    va_end(args);
    printf("%s", msg); fflush(stdout);
    if (flog) fprintf(flog, "%s", msg);
    if (error) exit(1);
}

/*---------------------------------------------------------------------------
 *
 *	Function:	get_file_extension
 *
 *-------------------------------------------------------------------------*/
std::string get_file_extension(const std::string &filename)
{
    if (filename.find_last_of(".") != std::string::npos)
        return filename.substr(filename.find_last_of(".") + 1);
    return "";
}

/*---------------------------------------------------------------------------
 *
 *	Function:	get_file_directory
 *
 *-------------------------------------------------------------------------*/
std::string get_file_directory(const std::string &filename)
{
    std::string::size_type p = filename.find_last_of("/");
    if (p == std::string::npos) return ".";
    return filename.substr(0, p);
}

/*---------------------------------------------------------------------------
 *
 *	Function:	get_filename_base
 *
 *-------------------------------------------------------------------------*/
std::string get_filename_base(const std::string &filename)
{
    std::string::size_type p = filename.find_last_of("/");
    if (p == std::string::npos) return filename;
    return filename.substr(p + 1);
}

/*---------------------------------------------------------------------------
 *
 *	Function:	get_step_number
 *
 *-------------------------------------------------------------------------*/
int get_step_number(const std::string &filename)
{
    std::string str = get_filename_base(filename);
    size_t i = 0;
    for (; i < str.length(); i++)
        if (isdigit(str[i])) break;
    str = str.substr(i, str.length() - i);
    return std::atoi(str.c_str());
}

/*---------------------------------------------------------------------------
 *
 *	Function:	replace_string
 *
 *-------------------------------------------------------------------------*/
std::string replace_string(std::string& str, const std::string& from, const std::string& to) {
    std::string new_str = str;
    size_t start_pos = new_str.find(from);
    if (start_pos == std::string::npos) return new_str;
    new_str.replace(start_pos, from.length(), to);
    return new_str;
}

/*---------------------------------------------------------------------------
 *
 *	Function:	create_directory
 *
 *-------------------------------------------------------------------------*/
int create_directory(std::string dirname)
{
    if (mkdir(dirname.c_str(), S_IRWXU) != 0) {
        if (errno == EEXIST) {
            ExaDiS_log("Warning: directory %s already exists\n", dirname.c_str());
        } else {
            ExaDiS_fatal("Error: Open error %d on directory %s", errno, dirname.c_str());
        }
    }
    return 0;
}

/*---------------------------------------------------------------------------
 *
 *	Function:	remove_directory
 *
 *-------------------------------------------------------------------------*/
void remove_directory(std::string dirname)
{
    // Clean directory
    std::string rm = "rm -rf " + dirname;
    system(rm.c_str());
}

/*---------------------------------------------------------------------------
 *
 *      Function:    get_time_base
 *
 *-------------------------------------------------------------------------*/
double get_time_base(void)
{
    const unsigned long long int million = 1000000;
    struct timeval tv;
    unsigned long long int s, us;

    gettimeofday(&tv, NULL);

    s = tv.tv_sec;
    us = tv.tv_usec;
    return 1.0*(s*million + us)/1.e6;
}

} // namespace ExaDiS
