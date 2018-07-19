
"""Check that the required software is available."""

from __future__ import print_function, division #Python 2 compatibility
import subprocess
import sys

#check required python modules
import pathlib #standard library in python 3 but not in python 2
import yaml
import jinja2
import numpy
import scipy
import matplotlib.pyplot as plt
import pandas
import fenics as fem

#fenics version check
target_fenics_versions=[2016, 2017]
fenics_version_msg_template="This code was written for FEniCS major versions '%s'. Detected major version '%d'."
assert fem.DOLFIN_VERSION_MAJOR in target_fenics_versions, fenics_version_msg_template%(target_fenics_version,fem.DOLFIN_VERSION_MAJOR)

#check that gmsh runs
gmsh_cmd="gmsh --version"
retcode=subprocess.call(gmsh_cmd,stderr=subprocess.DEVNULL,shell=True)
assert retcode==0, "gmsh failed to execute correctly"

#tests passed
print("OK")