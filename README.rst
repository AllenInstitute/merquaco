MERQuaCo
========

**MERQuaCo** is a tool for assessing the quality of spatial transcriptomics experiments on the Vizgen MERSCOPE platform. **MERQuaCo** computes a range of metrics to assess transcript detection efficiency in space, tissue integrity, and data loss using the transcripts table, codebook, and DAPI image. All analyses ignore cell segmentation boundaries and only use information pertaining to transcript locations.

Installation
------------

.. code-block:: bash

    $ git clone https://github.com/AllenInstitute/merquaco.git
    $ cd merquaco
    $ pip install .

Usage
-----
Refer to the `demo notebook <https://github.com/AllenInstitute/merquaco/blob/main/demo_notebook.ipynb>`_ and `documentation <https://merquaco.readthedocs.io/en/latest/api.html>`_ for example usage

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


Contact
-------
paul.olsen@alleninstitute.org and naomi.martin@alleninstitute.org
