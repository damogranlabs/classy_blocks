.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/FranzBangar/classy_blocks/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

classy_blocks could always use more documentation, whether as part of the
official classy_blocks docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/FranzBangar/classy_blocks/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `classy_blocks` for local development.

1. Fork the `classy_blocks` repo on GitHub
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/classy_blocks.git

3. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

4. Prepare and activate virtual environment, update pip::

    $ python -m venv venv
    $ "venv/bin/activate"
    $ python -m pip install -U pip

5. Install required dependencies (based on what you want to contribute).
- Small fixes like documentation, typo or similar: no need to install anything. Do your change and submit PR.
- Code/test/examples fixes: install local classy_block pkg and development requirements::

    $ python -m pip install -r requirements_dev.txt
    $ python -m pip install -e .

6. If code changes were made: check that your changes pass tests, typing, format and lint rules::

    $ pytest tests
    $ mypy typing
    $ black src
    $ pylint src

    Note, make sure you test on all python versions. Help yourself with `tox` configurations.

7. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python >= 3.6. Check
   results of GitHub actions and make sure that nothing was broken.

Tips
----

To run a subset of tests::

$ pytest tests.test_classy_blocks
