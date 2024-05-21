MERQuaCo
========

**MERQuaCo** is a tool for assessing the quality of spatial transcriptomics experiments on the Vizgen MERSCOPE platform. **MERQuaCo** computes a range of metrics to assess transcript detection efficiency in space, tissue integrity, and deviations from ideal perfusion flow rate using the transcripts table, codebook, DAPI image, and perfusion log file. All analyses ignore cell segmentation boundaries and only use information pertaining to transcript locations.

Key Metrics
-----------

- Transcripts per on-tissue square micrometer

  - Measure of transcript density, easily comparable between experiments

- Data loss

  - Individual field-of-view tiles with missing transcripts for ~10-100 genes

- Periodicity

  - Nonuniform transcript detection efficiency in x-, y-axes within FOV tiles

- 9/0 Ratio

  - Nonuniform transcript detection efficiency in z-axis throughout entire tissue

- Perfusion flow rate

  - Deviations from ideal fluidics can subtly impact transcript detection for a subset of genes

- Tissue integrity

  - Assessment of missing tissue from damage or gel detachment.

Installation
------------

.. code-block:: bash

    $ git clone https://github.com/AllenInstitute/merquaco.git
    $ cd merquaco
    $ pip install .
