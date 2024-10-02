merquaco
====================================
**merquaco** is a tool for assessing the quality of spatial transcriptomics experiments on the Vizgen MERSCOPE platform.
merquaco computes a range of metrics to assess transcript detection efficiency in space, tissue integrity, and
deviations from ideal perfusion flow rate using the transcripts table, codebook, DAPI image, and perfusion log file.
All analyses ignore cell segmentation boundaries and only use information pertaining to transcript locations.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

**merquaco**'s key applications
-------------------------------
- Calculate total tissue area and quantify extent of tissue damage or detachment from coverslip.
- Measure transcript density for on-tissue regions; easily comparable metric between experiments.
- Quantify nonuniform transcript detection in x, y, z axes.
- Detect fields-of-view with missing transcript species.

Contents
--------
.. toctree::

	usage
	api
