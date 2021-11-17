PoreMapper
==========

:author: Andrew Tarzia

Inflate a balloon inside a cavity to get the ``pore`` and ``windows``.

Built for molecules with a single, central cavity.

Please contact me with any questions (<andrew.tarzia@gmail.com>) or submit an issue!

Installation
------------

Clone this repository and ``python setup.py develop`` in this directory, or using pip::

    $ pip install PoreMapper


Algorithm
---------

Very simple algorithm:

1. Define a sphere of radius 0.1 Angstrom at the centroid of the host with equally placed beads on the sphere. The number of beads is defined by the ``host.get_maximum_diameter()``. Beads have ``sigma``, which define their radius, and the resolution of the calculation. Hosts have atoms, which have radii defined by Streussel atomic radii [citation].

2. Define steps of inflation (simply moving each bead in the blob along a vector emanating from the centroid) at even step size from 0.1 Angstrom to maximum host radii.

3. For each step, check if a bead will collide with the host (based on distance-(bead radii + atom radii)). If it collides, it becomes immovable and a pore bead. Else, continue on.

A pore, and blob, have a series of analysis methods, including:

* Measures of pore shape based on the inertia tensor.

* Measure of pore radii (based on distance to host) and volume (based on its convex hull).

* Calculation of windows based on the blob (a Pore contains a Blob), where movable beads are clustered using ``sklearn.cluster.MeanShift`` [this may change and be improved] to calculate the number and size of windows.

Examples
--------

Two examples in ``examples/`` take ``.xyz`` files and either run the step-wise inflation (``inflate_blob``) or the single-step inflation (``get_inflated_blob``).
The step-wise process will produce a plot and ``.xyz`` structures, monitoring the pore and blob, while the single-step will run the full calculation and produce just the final pore and blob.

Contributors and Acknowledgements
---------------------------------

I developed this code as a post doc in the Jelfs research group at Imperial College London (<http://www.jelfs-group.org/>, <https://github.com/JelfsMaterialsGroup>).

License
-------

This project is licensed under the MIT license.
