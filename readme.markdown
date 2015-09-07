Step 12a
========

Overview
--------
This is a reimplementation of `step-12` without `MeshWorker` and with a Trilinos
solver (and distributed functionality). I think `MeshWorker` is quite nice and a
used a lot of its ideas, but I am not very familiar with DG, so it was a good
exercise to reimplement all of the face integration logic myself (and there is
not very much of it; perhaps a score of lines).

Files
-----
* `step-12a/step-12a.cc`: the original reimplementation.
* `distributed/step-12a-distributed.cc`: the distributed copy.
* `dg_tools/dg_tools.h`: some functions for calculating face and subface
  positions between neighboring cells.
* `dg_tools/test.cc`: A *very crude* test for some of the `DGTools` functions.

Pictures
--------
Here are two pictures of the distributed version in action, across `16`
processes. The first picture is of the solution itself:
![Solution](https://raw.githubusercontent.com/drwells/dealii-step-12a/master/distributed/solution.png)
and the second shows the sixteen subdomains:
![Subdomains](https://raw.githubusercontent.com/drwells/dealii-step-12a/master/distributed/subdomains.png)
