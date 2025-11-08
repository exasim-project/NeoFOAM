Continuous Integration
======================
The **FoamAdapter** project uses a two-level Continuous Integration (CI) system
to ensure correct builds, GPU compatibility, and automated benchmarking.

The main repository is hosted on **GitHub**, and GPU-based workflows are delegated
to **LRZ GitLab**, where jobs are executed on both **NVIDIA** and **AMD** GPUs.
The CI architecture for FoamAdapter is illustrated below.

.. figure:: _static/ci/ci_overview.png
   :align: center
   :alt: Overview of the CI architecture for FoamAdapter
   :width: 90%

--------------------------------
Continuous Integration on GitHub
--------------------------------
GitHub CI is responsible for managing the overall FoamAdapter CI workflow.

**Responsibilities:**

* Build and test FoamAdapter on **CPU** across different platforms (Linux, macOS, Windows).
* Push the source code and commit metadata to **LRZ GitLab**.
* Cancel outdated pipelines on LRZ GitLab for the same branch.
* Trigger new LRZ GitLab pipelines for GPU builds and benchmarks.

.. note::
   The GitHub CI acts as the *control layer* for all FoamAdapter CI operations.
   Developers interact only with GitHub — all LRZ GitLab pipelines are triggered automatically.

------------------------------------
Continuous Integration on LRZ GitLab
------------------------------------
The LRZ GitLab CI handles GPU-related operations.

**Responsibilities:**

* Build and test FoamAdapter on **NVIDIA** and **AMD** GPUs on Linux.
* Run benchmark jobs after successful build and test stages.
* Report the status and results back to GitHub for unified monitoring.

.. _ci-foamadapter-workflow:

--------------------
Development Workflow
--------------------
The development workflow for FoamAdapter proceeds as follows:

#. A developer opens a pull request (PR) or pushes a commit to an existing PR on GitHub.
#. GitHub CI builds and tests FoamAdapter on CPUs.
#. GitHub CI pushes the same branch to LRZ GitLab.
#. GitHub CI cancels all pending or running LRZ GitLab pipelines for that branch.
#. GitHub CI triggers a **new LRZ GitLab pipeline**.
#. LRZ GitLab CI builds and tests FoamAdapter on GPUs.
#. *(Optional)* Benchmark jobs are executed after successful testing.
#. The developer monitors all results directly on GitHub.

.. _ci-foamadapter-labels:

-------------------
Pull Request Labels
-------------------
FoamAdapter’s GitHub repository uses labels to control the CI behavior.

**Relevant Labels:**

* ``Skip-build`` — Skip all build-and-test jobs on both GitHub and LRZ GitLab.
* ``benchmark`` — Enable GPU benchmarking jobs after successful build-and-test jobs.

These labels allow developers to customize the CI process according to their needs.

.. _ci-foamadapter-summary:

-------
Summary
-------
The FoamAdapter CI system provides:

* Unified GitHub-driven CI management.
* Transparent CPU and GPU build workflows.
* Automatic synchronization between GitHub and LRZ GitLab.
* Branch-aware pipeline handling and cancellation.
* On-demand GPU benchmarking via PR labels.

.. seealso::

   * :ref:`ci-foamadapter-workflow`
   * :ref:`ci-foamadapter-labels`
