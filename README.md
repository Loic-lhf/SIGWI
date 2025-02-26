# SIGWI
Identification of Signature whistles in the DOLPHINFREE dataset.

Using an original mehtod (DTW + HDBSCAN + manual verification), this repo analyses the whistle contour types (WCTs) and Signature Whistle Types (SWTs) of short-beaked common dolphins during the DOLPHINFREE experiments.

See article (in preparation) for more details.



## Description
```bash
.
├── README.md
├── WCT_analysis_utils.py		# Functions/classes for analyses
├── dtw_hdbscan_categorisation.py	# Categorisation of contours into WCTs
├── WCT_analysis.ipynb			# Overview and statistics on WCTs and SWTs
├── artwarp-comparison			# Categorisation of contours with ARTwarp
│	├── artwarp-custom				# ARTwarp modified for octave
│	├── ARTwarp_automation.py			# Python script to run ARTwarp on all contours
│	├── WCT_verification.py				# Python script for manual verification
│	└── artwarp_hdbscan_comparison			# Overview and comparison of WCTs obtained from 2 methods
├── resources				# Inputs and outputs of python scripts
│	├── ARTwarp_outputs				# octave and json files
│	├── DF-whistles					# Whistle features and json files
│	└── dtw_resources				# hdbscan categorisation results
└── drafts				# Folder containing unused scripts
```

## What to use
Main results : see `WCT_analysis.ipynb`.

Comparison with [ARTwarp](https://github.com/dolphin-acoustics-vip/artwarp/releases): see `artwarp-comparison/artwarp_hdbscan_comparison.ipynb`.

## Contact
For any additional requests, please contact me at [loic.lehnhoff@gmail.com](mailto:loic.lehnhoff@gmail.com)
