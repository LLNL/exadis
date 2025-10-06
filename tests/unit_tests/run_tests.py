"""
ExaDiS unit tests driver
Run with:
    python run_tests.py
    python run_tests.py --noplot
    python run_tests.py --build_path=/path/to/build/folder/

Nicolas Bertin
bertin1@llnl.gov
"""

import os, sys, time
import subprocess
import numpy as np

# ANSI font codes
FONT_GREEN = "\033[92m"
FONT_RED = "\033[91m"
FONT_BOLD = "\033[1m"
FONT_RESET = "\033[0m"

class CheckTestOutput():
    """Compare the console output to the expected output or output file content
    """
    def diff_count(self, s1, s2):
        differences = 0
        min_len = min(len(s1), len(s2))
        # Compare characters up to the length of the shorter string
        for i in range(min_len):
            if s1[i] != s2[i]: differences += 1
        # Add the difference in length if strings are not of equal length
        differences += abs(len(s1) - len(s2))
        return differences
    
    def __init__(self, expected_output):
        self.expected_output = expected_output
        
    def __call__(self, test_name, actual_output, error_output):
        if isinstance(self.expected_output, str):
            expected_output_file = self.expected_output
            if not os.path.exists(expected_output_file):
                print(f"[{test_name}] {FONT_RED}ERROR: Expected output file '{expected_output_file}' not found.{FONT_RESET}")
                return False
            
            expected_output = np.loadtxt(expected_output_file)
            try:
                actual_output = np.fromstring(actual_output, sep=' ').reshape(expected_output.shape)
            except Exception as e:
                print(f"[{test_name}] {FONT_RED}ERROR: {e}{FONT_RESET}")
                return False
            if np.allclose(actual_output, expected_output, rtol=1e-4):
                print(f"[{test_name}] {FONT_GREEN}{FONT_BOLD}PASS{FONT_RESET}")
                return True
            else:
                print(f"[{test_name}] {FONT_RED}{FONT_BOLD}FAIL{FONT_RESET}")
                print(f"  Error magnitude: {np.linalg.norm(actual_output-expected_output):e}")
                return False
        else:
            expected_output = '\n'.join([str(item) for item in self.expected_output])
        
        if actual_output == expected_output:
            print(f"[{test_name}] {FONT_GREEN}{FONT_BOLD}PASS{FONT_RESET}")
            return True
        else:
            print(f"[{test_name}] {FONT_RED}{FONT_BOLD}FAIL{FONT_RESET}")
            if error_output:
                print(f"  Error output: {error_output}")
            else:
                if len(expected_output) < 100 and len(actual_output) < 100:
                    print(f"  Expected: {expected_output}")
                    print(f"  Actual:   {actual_output}")
                else:
                    print(f"  Expected / Actual diff count: {self.diff_count(expected_output, actual_output)}")
            return False

class CheckRunError(CheckTestOutput):
    """Check that a command has run and did not produce errors
    """
    def __init__(self):
        pass
    def __call__(self, test_name, actual_output, error_output):
        if error_output:
            print(f"[{test_name}] {FONT_RED}{FONT_BOLD}FAIL{FONT_RESET}")
            print(f"  Error output: {error_output}")
            return False
        elif "Error:" in actual_output:
            print(f"[{test_name}] {FONT_RED}{FONT_BOLD}FAIL{FONT_RESET}")
            print(f"  Error output: {actual_output}")
            return False
        else:
            print(f"[{test_name}] {FONT_GREEN}{FONT_BOLD}PASS{FONT_RESET}")
            return True
            
class CheckRunFile(CheckTestOutput):
    """Check that a given output file was generated
    """
    def __init__(self, file_path):
        self.file_path = file_path
    def __call__(self, test_name, actual_output, error_output):
        if not os.path.exists(self.file_path):
            print(f"[{test_name}] {FONT_RED}{FONT_BOLD}FAIL{FONT_RESET}")
            print(f"  Expected output file '{self.file_path}' not found.{FONT_RESET}")
            return False
        else:
            print(f"[{test_name}] {FONT_GREEN}{FONT_BOLD}PASS{FONT_RESET}")
            return True
            
class CheckRunStep(CheckRunFile):
    """Check that a given simulation step was reached
    """
    def __init__(self, step):
        self.step = step
    def __call__(self, test_name, actual_output, error_output):
        if not os.path.exists(f"output/config.{self.step}.data"):
            print(f"[{test_name}] {FONT_RED}{FONT_BOLD}FAIL{FONT_RESET}")
            print(f"  Expected simulation step '{self.step}' was not reached.{FONT_RESET}")
            return False
        else:
            print(f"[{test_name}] {FONT_GREEN}{FONT_BOLD}PASS{FONT_RESET}")
            return True

def cleanup():
    directory_to_remove = "output"
    if os.path.exists(directory_to_remove):
        try:
            import shutil
            shutil.rmtree(directory_to_remove)
            os.remove('tmp.py')
        except OSError as e:
            pass

class PreProcessTest():
    """Pre-processing base class
    """
    def __call__(self, test_name, command, expected_output):
        cleanup() # clean-up before running
        return command

class VisualizeExample(PreProcessTest):
    """Toggle to disable matplotlib visualizer
    """
    def __init__(self, visualizer=True):
        self.visualizer = visualizer
    def __call__(self, test_name, command, expected_output):
        super().__call__(test_name, command, expected_output)
        if not self.visualizer:
            file_path, new_path = command[1], "tmp.py"
            try:
                with open(file_path, 'r') as f:
                    file_content = f.read()
                import re
                file_content = re.sub(r"vis = VisualizeNetwork\(.*\)", "vis = None", file_content)
                with open(new_path, 'w') as f:
                    f.write(file_content)
            except Exception as e:
                print(f"[{test_name}] {FONT_RED}ERROR: {e}{FONT_RESET}")
            command[1] = new_path
        return command

def run_test(test_name, command, expected_output, preprocess):
    
    if preprocess is None:
        preprocess = PreProcessTest()
    try:
        command = preprocess(test_name, command, expected_output)
    except Exception as e:
        print(f"[{test_name}] {FONT_RED}ERROR: Preprocessing failed: {e}{FONT_RESET}")
        return False
    
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        actual_output = result.stdout.strip()
        error_output = result.stderr.strip()
    except Exception as e:
        print(f"[{test_name}] {FONT_RED}ERROR: Failed to run command: {e}{FONT_RESET}")
        return False

    if isinstance(expected_output, CheckTestOutput):
        return expected_output(test_name, actual_output, error_output)
    else:
        return CheckTestOutput(expected_output)(test_name, actual_output, error_output)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ExaDiS unit tests")
    parser.add_argument('--noplot', action='store_true', help='Disable matplotlib visualizer')
    parser.add_argument('--build_path', type=str, default="../../build", help='Build folder path')
    args = parser.parse_args()
    
    build_path = args.build_path
    if build_path == "../../build" and not os.path.isdir(build_path):
        build_path = "../../../../build/core/exadis"
    if not os.path.isdir(build_path):
        print(f"{FONT_RED}ERROR: exadis build path ({build_path}) does not exist. Some tests will fail. {FONT_RESET}")
    tests_path = os.path.join(build_path, 'tests/unit_tests')
    if not os.path.isdir(tests_path):
        print(f"{FONT_RED}ERROR: exadis tests path ({tests_path}) does not exist. Some tests will fail. " \
        f"Make sure the code was compiled with option '-DEXADIS_BUILD_TESTS=On'. {FONT_RESET}")
    
    vis = VisualizeExample(not args.noplot)
    
    # Define test cases (test_name, command, expected_output)
    test_cases = [
        ("Test_Kokkos_Init", [f"{build_path}/tests/test_kokkos", "test_initialize"], ["test_initialize()\n PASS"], None),
        ("Test_Kokkos_Memory", [f"{build_path}/tests/test_kokkos", "test_memory"], ["test_memory()\n PASS"], None),
        ("Test_Kokkos_Unified_Memory", [f"{build_path}/tests/test_kokkos", "test_unified_memory"], ["test_unified_memory()\n PASS"], None),
        ("Test_System", [f"{build_path}/tests/test_system", "test_system"], ["test_system()\n PASS"], None),
        ("Test_System_Unified", [f"{build_path}/tests/test_system", "test_system_unified"], ["test_system_unified()\n PASS"], None),
        ("Test_Exadis", [f"{build_path}/tests/test_exadis"], CheckRunStep(100), None),
        
        ("Test_Pyexadis_Import", [sys.executable, "test_pyexadis.py", "test_import"], ["pass"], None),
        ("Test_Pyexadis_Init", [sys.executable, "test_pyexadis.py", "test_init"], ["pass"], None),
        
        ("Test_Force_LT_Python", [sys.executable, "test_force.py", "lt"], "expected_output/test_force_lt.dat", None),
        ("Test_Force_CUTOFF_Python", [sys.executable, "test_force.py", "cutoff"], "expected_output/test_force_cutoff.dat", None),
        ("Test_Force_DDD-FFT_Python", [sys.executable, "test_force.py", "ddd_fft"], "expected_output/test_force_ddd_fft.dat", None),
        ("Test_Force_FFT_Python", [sys.executable, "test_force.py", "fft"], "expected_output/test_force_fft.dat", None),
        
        ("Test_Force_LT_CPP", [f"{tests_path}/test_force", "lt"], "expected_output/test_force_lt.dat", None),
        ("Test_Force_CUTOFF_CPP", [f"{tests_path}/test_force", "cutoff"], "expected_output/test_force_cutoff.dat", None),
        ("Test_Force_DDD-FFT_CPP", [f"{tests_path}/test_force", "ddd_fft"], "expected_output/test_force_ddd_fft.dat", None),
        ("Test_Force_FFT_CPP", [f"{tests_path}/test_force", "fft"], "expected_output/test_force_fft.dat", None),
        ("Test_Force_FFT_SerialDisNet_CPP", [f"{tests_path}/test_force", "fft_serialdisnet"], "expected_output/test_force_fft.dat", None),
        
        ("Test_Neighbor_NeighborList_CPP", [f"{tests_path}/test_neighbor", "test_neighborlist"], [22330, 44362, 78780, 298786, 673446, 3609540], None),
        ("Test_Neighbor_SegSegList_CPP", [f"{tests_path}/test_neighbor", "test_segseglist"], [4171, 11436, 22158, 105147, 259015, 1549038], None),
        ("Test_Neighbor_SubGroups_CPP", [f"{tests_path}/test_neighbor", "test_subgroups"], [247266, 0, 752, 3221, 10696, 232597], None),
        
        ("Test_Example2_Python", [sys.executable, "../../examples/02_frank_read_src/test_frank_read_src.py"], CheckRunStep(200), vis),
        ("Test_Example3_Python", [sys.executable, "../../examples/03_collision/test_collision.py"], CheckRunStep(200), vis),
        ("Test_Example4_Python", [sys.executable, "../../examples/04_bcc_junction/test_bcc_junction.py"], CheckRunStep(200), vis),
        ("Test_Example5_Python", [sys.executable, "../../examples/05_fcc_junctions/test_fcc_junctions.py"], CheckRunStep(200), vis),
        ("Test_Example7-FCC_Python", [sys.executable, "../../examples/07_cross_slip/test_cross_slip.py"], CheckRunStep(1000), vis),
        #("Test_Example7-BCC_Python", [sys.executable, "../../examples/07_cross_slip/test_cross_slip_bcc.py"], CheckRunStep(3000), vis),
    ]
    
    # Run test cases
    tstart = time.perf_counter()
    num_passed = 0
    for test_name, command, expected_output, preprocess in test_cases:
        num_passed += run_test(test_name, command, expected_output, preprocess)
    tend = time.perf_counter()
    
    # Clean directory
    cleanup()

    # Print global results
    print(f"-------------------------")
    print(f"{len(test_cases)} tests executed in {(tend-tstart):.2f} seconds")
    if num_passed == len(test_cases):
        print(f"{FONT_GREEN}All ({num_passed}/{len(test_cases)}) tests PASSED.{FONT_RESET}")
        sys.exit(0)
    else:
        print(f"{FONT_GREEN}{num_passed} tests PASSED.{FONT_RESET}")
        print(f"{FONT_RED}{len(test_cases)-num_passed} tests FAILED.{FONT_RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()
