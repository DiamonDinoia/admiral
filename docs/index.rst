Admiral
=======

Admiral is a C++20 FFT library: complex and real transforms, any size, 1-D and
N-D, float or double, with optional multithreading. One compiled engine behind
three interfaces: C++, C, and a drop-in FFTW subset.

Project links
-------------

- `GitHub repository <https://github.com/DiamonDinoia/admiral>`_
- `Issue tracker <https://github.com/DiamonDinoia/admiral/issues>`_
- `License (BSD 3-Clause Attribution) <https://github.com/DiamonDinoia/admiral/blob/master/LICENSE>`_

Quick start
-----------

.. code-block:: cpp

   #include <admiral/admiral.hpp>

   std::vector<std::complex<double>> x(1024);

   admiral::plan<double> p(x.size());
   p.forward(x);                  // in place
   p.inverse(x);                  // divides by 1024

   admiral::plan<double> p2d({64, 64});              // N-D
   admiral::plan<double> p8(1 << 20, {.nthreads = 8});

   admiral::forward<double>(x, x);   // one-shot, no plan

The snippet is `examples/quickstart.cpp
<https://github.com/DiamonDinoia/admiral/blob/master/examples/quickstart.cpp>`_,
built and run by ctest; `examples/
<https://github.com/DiamonDinoia/admiral/tree/master/examples>`_ also covers real
transforms, DCT/DST, per-axis transforms, C, and the FFTW drop-in.

Next reads
----------

- :doc:`install`
- :doc:`cpp-api`
- :doc:`usage`
- :doc:`build-options`

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install

.. toctree::
   :maxdepth: 2
   :caption: Guides

   cpp-api
   usage
   build-options

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/library_root
