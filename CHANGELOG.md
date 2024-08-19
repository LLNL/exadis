
# Change Log
All notable changes to ExaDiS will be documented in this file.


## Global update 3 - Aug 19, 2024

### Added
- Added force cutoff model and corresponding bindings
- Extended ForceFFT to support different resolution in each direction
- Added compute_team option to ForceSeg and team kernel to ForceN2
- Added binding to the simulation stepping options
- Added binding to the simulation restart option
- Added option to automatically select the minseg param
- Added option strict to NeighborBox class
- Added CHANGELOG file


### Changed
- Added function register_neighbor_cutoff()
- Modified binding to accept independent FFT grid dimensions
- Switched to using the Voigt convention for stress in the python binding to be consistent with OpenDiS


### Fixed
- Checking for positive pair distances when building seg/seg lists


## Update Aug 9, 2024
### Added
- Added binding to enable calling python-based modules from within ExaDiS


## Update Jul 6, 2024

### Added
- Added benchmark tests


## Global update 2 - Jun 22, 2024

### Changed
- Made modifications to allow for hybrid use of unified memory

### Fixed
- Fixed bug to use shared memory for armsets array in TopologyParallel
- Fixed bugs with the use of triclinic cells and modified Cell to use origin instead of min-max coords


## Global update 1 - Jun 4, 2024

### Added
- Added use of node tags internally and in python binding
- Added tests for kokkos and cuda
- Added non-linear BCC mobility law Mobility_BCC_nl
- Added helper functions make_system() make_network_manager()
- Added remesh rule to not refine between pinned nodes

### Changed
- Modified test cases to work with updated DisNetManager
