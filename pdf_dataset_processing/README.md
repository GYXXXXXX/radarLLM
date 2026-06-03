# PDF Dataset Processing

This folder is reserved for future code that processes datasets described by
PDF manuals, especially the MTDSP marine-target multi-source observation
dataset.

Suggested modules to add later:

- `extract_pdf_metadata.py`: extract dataset names, ship classes, and file
  structure from the PDF manuals.
- `parse_mtdsp_zip.py`: scan downloaded MTDSP ZIP packages and build a unified
  manifest.
- `convert_radar_dat.py`: convert radar IF/video `.dat` slices to MAT, NumPy,
  or tensor cache formats.
- `align_multisource.py`: align radar slices, visible/infrared images, AIS
  tracks, and meteorological/hydrological records by timestamp and target ID.
- `build_tasks.py`: create train/validation splits for classification,
  regression, detection, tracking, or trajectory-prediction tasks.

