#!/usr/bin/env python

import sys

from setuptools import setup

assert sys.version_info.major == 3 and sys.version_info.minor >= 6

setup(
    name='la_mbda',
    packages=['la_mbda', 'experiments'],
    )
