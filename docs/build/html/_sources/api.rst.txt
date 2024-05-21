API
===
Import MERQuaCo as::

	import merquaco as mqc

Experiment
~~~~~~~~~~

.. module:: merquaco.experiment.Experiment
.. currentmodule:: merquaco.experiment.Experiment

.. autosummary::
	:toctree: api
	
	read_transcripts
	remove_blanks
	read_codebook
	scale_transcripts_xy
	find_fovs
	get_fov_neighbors
	get_fovs_dataframe
	get_transcript_density
	run_dropout_pipeline
	run_full_pixel_classification
	run_all_qc

Data Loss
~~~~~~~~~

.. module:: merquaco.data_loss
.. currentmodule:: merquaco.data_loss

.. autosummary::
	:toctree: api
	
	FOVDropout.find_on_tissue_fovs
	FOVDropout.detect_dropouts
	FOVDropout.compare_codebook_fov_genes
	FOVDropout.detect_false_positives
	FOVDropout.save_fov_tsv	

	DropoutResult.get_dropout_count
	DropoutResult.get_dropped_genes
	DropoutResult.get_dropped_gene_counts
	DropoutResult.get_dropped_fovs
	DropoutResult.get_dropped_fov_counts
	DropoutResult.get_considered_genes
	DropoutResult.get_considered_gene_counts
	DropoutResult.get_considered_fovs
	DropoutResult.get_considered_fov_counts
	DropoutResult.get_false_positive_fovs
	DropoutResult.get_false_positive_fov_counts
	DropoutResult.dropout_summary
	DropoutResult.draw_genes_dropped_per_fov


Periodicity
~~~~~~~~~~~

.. module:: merquaco.periodicity
.. currentmodule:: merquaco.periodicity

.. autosummary::
	:toctree: api
	
	get_periodicity_list
	get_chunk_values
	get_image_dimensions
	get_periodicity_vals_all_z

Z-Axis Nonuniformity
~~~~~~~~~~~~~~~~~~~~

.. module:: merquaco.z_plane_detection
.. currentmodule:: merquaco.z_plane_detection

.. autosummary::
	:toctree: api

	get_transcripts_per_z
	compute_z_ratio

Perfusion
~~~~~~~~~

.. module:: merquaco.perfusion
.. currentmodule:: merquaco.perfusion

.. autosummary::
	:toctree: api

	analyze_flow

Figures
~~~~~~~

.. module:: merquaco.figures
.. currentmodule:: merquaco.figures

.. autosummary::
	:toctree: api
	
	transcripts_overview
	plot_periodicity_hist
	plot_every_z_plane
	plot_transcripts_per_z
	plot_perfusion_figure
	plot_pixel_percentages
	plot_pixel_classification
	plot_full_pixel_fig
	plot_mask
	plot_masks

Pixel Classification
~~~~~~~~~~~~~~~~~~~~

.. module:: merquaco.pixel_classification
.. currentmodule:: merquaco.pixel_classification

.. autosummary::
	:toctree: api

	generate_mask
	create_transcripts_image
	generate_transcripts_mask
	create_dapi_image
	generate_dapi_mask
	generate_detachment_mask
	create_ventricle_genes_image
	generate_ventricle_mask
	generate_damage_mask
	classify_pixels
	calculate_class_areas
	calculate_class_percentages
