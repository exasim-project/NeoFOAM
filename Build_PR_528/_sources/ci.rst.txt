Continuous Integration
======================
The **NeoFOAM** project uses a two-level Continuous Integration (CI) system
to ensure correct builds, GPU compatibility, and automated benchmarking.

The main repository is hosted on **GitHub**, and GPU-based workflows are delegated
to **LRZ GitLab**, where jobs are executed on both **NVIDIA** and **AMD** GPUs.
The CI architecture for NeoFOAM is illustrated below.

.. figure:: _static/ci/ci_overview.png
   :align: center
   :alt: Overview of the CI architecture for NeoFOAM
   :width: 90%

--------------------------------
Continuous Integration on GitHub
--------------------------------
GitHub CI is responsible for managing the overall NeoFOAM CI workflow.

**Responsibilities:**

* Build and test NeoFOAM on **CPU** on Linux (clang, gcc) and macOS, and the Python package.
* Push the source code and commit metadata to **LRZ GitLab**.
* Cancel outdated pipelines on LRZ GitLab for the same branch.
* Trigger new LRZ GitLab pipelines for GPU builds, and weekly for benchmarks.

.. note::
   The GitHub CI acts as the *control layer* for all NeoFOAM CI operations.
   Developers interact only with GitHub — all LRZ GitLab pipelines are triggered automatically.

All of it lives in one workflow, ``.github/workflows/ci.yaml``, which runs in stages so that a
broken commit occupies as few of the organization's shared runners as possible:

#. **Gate**: evaluates the event, the PR labels and the NeoN branch to build against.
#. **Cheap checks** (no build, a few minutes): pre-commit (without mypy) and the changelog check.
   In parallel, the **first build**: Linux clang build and unit tests. It does not wait for the
   cheap checks, so a formatting issue does not hold back the build results.
#. Only once that passes: Linux gcc, macOS, IWYU followed by clang-tidy, the LRZ GPU pipeline, and
   the Python job. The Python job builds the package once and runs pytest with coverage, mypy and,
   on PRs, the documentation preview.
#. After the Python job: the Python tests on 3.11 and 3.13, on every PR and push that builds.

The Linux builds (clang and gcc), IWYU and the Python job do not build NeoN themselves when they
test against NeoN ``develop``: they install the Open MPI flavour of the nightly ``neon-pde`` conda
package from `exasim-project/nightlies <https://prefix.dev/channels/exasim-project/nightlies>`_ and
configure NeoFOAM with ``-DNEOFOAM_USE_INSTALLED_NEON=ON``. The clang and IWYU builds install the clang-built
package (``cpu_openmpi_clang_py312_*``), gcc and the Python job the GCC-built one: GCC and clang 18+ mangle C++20
requires-clauses differently, so clang-compiled code cannot link a GCC-built NeoN. clang-tidy installs the same package as
the IWYU build, whose compilation database it uses. The Python job runs on the package
environment's Python 3.12 (the Python version of the Open MPI build), which also provides the
``neon`` module; NeoFOAM's bindings then take nanobind from the Python build environment, the same
nanobind ABI the package's ``_neon`` was built with. If the package cannot be installed, these
builds fall back to building NeoN and warn.

That NeoN is a Release build of ``develop`` from the last night. To test against the current
``develop`` with the ``develop`` preset's Kokkos debug and bounds checks, set the
``neon_from_source`` label: every build of the PR then builds NeoN from source. A NeoN branch with
the same name as the PR's branch is always built from source.

``build_doc.yaml`` deploys the documentation of merged PRs and release tags, and
``gpu_benchmark.yaml`` runs the GPU benchmarks of ``develop`` once a week. Their results are on the
``NeoFOAM_develop`` branch of
`NeoFOAM-BenchmarkData <https://github.com/exasim-project/NeoFOAM-BenchmarkData>`_.

------------------------------------
Continuous Integration on LRZ GitLab
------------------------------------
The LRZ GitLab CI handles GPU-related operations.

**Responsibilities:**

* Build and test NeoFOAM on **NVIDIA** and **AMD** GPUs on Linux.
* Run the weekly benchmark jobs of ``develop``.
* Report the status and results back to GitHub for unified monitoring.

.. _ci-neofoam-workflow:

--------------------
Development Workflow
--------------------
The development workflow for NeoFOAM proceeds as follows:

#. A developer opens a pull request (PR) or pushes a commit to an existing PR on GitHub.
#. GitHub CI runs the cheap checks and, in parallel, builds and tests NeoFOAM on CPUs.
#. Once the first CPU build passes, GitHub CI pushes the same branch to LRZ GitLab.
#. GitHub CI cancels all pending or running LRZ GitLab pipelines for that branch.
#. GitHub CI triggers a **new LRZ GitLab pipeline**.
#. LRZ GitLab CI builds and tests NeoFOAM on GPUs.
#. The developer monitors all results directly on GitHub.

.. _ci-neofoam-labels:

-------------------
Pull Request Labels
-------------------
NeoFOAM’s GitHub repository uses labels to control the CI behavior.

**Relevant Labels:**

* ``skip-build`` — Skip all build-and-test jobs on both GitHub and LRZ GitLab.

Pull requests that only change documentation (``doc/``, ``LICENSES/``, Markdown and
reStructuredText files or ``CITATION.cff``) skip the build-and-test jobs automatically, here and on
LRZ GitLab. The Python 3.12 job still installs the package and builds the documentation preview,
without pytest and mypy.

When a pull request is closed, merged or not, ``.github/workflows/cleanup-pr-branches.yaml`` cancels
its running GitHub Actions runs and LRZ GitLab pipelines and deletes the copy of its branch that the
GPU job pushed to the LRZ GitLab mirror. GitHub deletes the branch itself on merge.
* ``neon_from_source`` — Build NeoN from source with debug options instead of installing the nightly
  conda package.
* ``skip-changelog`` — Skip the changelog check.
* ``skip-validation`` — Skip the solver validation runs in the LRZ GitLab build-and-test job.
* ``skip-nvidia``, ``skip-amd``, ``skip-intel`` — Skip the GPU jobs on the respective vendor's hardware.

These labels allow developers to customize the CI process according to their needs.

Adding or removing ``skip-build`` or ``neon_from_source`` restarts CI, and adding or removing ``skip-changelog`` reruns the
changelog check. Any other label, including the ones above, takes effect on the next push, so
labelling a PR does not restart its builds.

.. _ci-neofoam-summary:

-------
Summary
-------
The NeoFOAM CI system provides:

* Unified GitHub-driven CI management.
* Transparent CPU and GPU build workflows.
* Automatic synchronization between GitHub and LRZ GitLab.
* Branch-aware pipeline handling and cancellation.
* Weekly GPU benchmarking of ``develop``.

.. seealso::

   * :ref:`ci-neofoam-workflow`
   * :ref:`ci-neofoam-labels`
