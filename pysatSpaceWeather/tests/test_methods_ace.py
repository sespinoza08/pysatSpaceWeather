#!/usr/bin/env python
# Full license can be found in License.md
# Full author list can be found in .zenodo.json file
# DOI:10.5281/zenodo.3986138
# ----------------------------------------------------------------------------
"""Integration and unit test suite for ACE methods."""

import pysat
import pytest

from pysatSpaceWeather.instruments.methods import ace as mm_ace

pysat_version_minor = int(pysat.__version__.split('.')[1])


class TestACEMethods(object):
    """Test class for ACE methods."""

    def setup(self):
        """Create a clean testing setup."""
        self.out = None
        return

    def teardown(self):
        """Clean up previous testing setup."""
        del self.out
        return

    def test_acknowledgements(self):
        """Test the ACE acknowledgements."""
        self.out = mm_ace.acknowledgements()
        assert self.out.find('ACE') >= 0
        return

    @pytest.mark.parametrize('name', ['mag', 'epam', 'swepam', 'sis'])
    def test_references(self, name):
        """Test the references for an ACE instrument."""
        self.out = mm_ace.references(name)
        assert self.out.find('Space Sci. Rev.') > 0
        return

    def test_references_bad_name(self):
        """Test the references raise an informative error for bad instrument."""
        with pytest.raises(KeyError) as kerr:
            mm_ace.references('ace')

        assert str(kerr.value).find('unknown ACE instrument') >= 0
        return


@pytest.mark.skipif(pysat_version_minor < 1)
class TestACESWEPAMMethods(object):
    """Test class for ACE SWEPAM methods."""

    def setup(self):
        """Create a clean testing setup."""
        self.testInst = pysat.Instrument('pysat', 'testing')
        self.testInst.load(date=self.testInst.inst_module._test_dates[''][''])

        self.omni_keys = ['sw_proton_dens_norm', 'sw_ion_temp_norm']
        return

    def teardown(self):
        """Clean up previous testing setup."""
        del self.testInst
        return

    def test_ace_swepam_hourly_omni_norm(self):
        """Test ACE SWEPAM conversion to OMNI hourly normalized standards."""

        self.testInst['slt'] *= 400.0
        mm_ace.ace_swepam_hourly_omni_norm(self.testInst, speed_key='slt',
                                           dens_key='mlt', temp_key='dummy3')

        # Test that normalized data and metadata are present and realistic
        for okey in self.omni_keys:
            assert okey in self.testInst.variables
            assert okey in self.testInst.meta.keys()
            assert (self.testInst[okey].values >= 0.0).all()

        return

    def test_ace_swepam_hourly_omni_norm_bad_keys(self):
        """Test ACE SWEPAM conversion to OMNI hourly normalized standards."""

        with pytest.raises(ValueError) as verr:
            mm_ace.ace_swepam_hourly_omni_norm(self.testInst)

        # Test the error message for missing data variables
        assert str(verr).find("instrument missing variable") >= 0

        return
