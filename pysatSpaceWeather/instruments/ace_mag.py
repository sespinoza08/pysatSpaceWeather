# -*- coding: utf-8 -*-
"""Supports ACE Magnetometer data

Properties
----------
platform
    'ace' Advanced Composition Explorer
name
    'mag' Magnetometer
tag
    - 'realtime' Real-time data from the Space Weather Prediction Center (SWPC)
    - 'historic' Historic data from the SWPC
inst_id
    - ''

Note
----
This is not the ACE scientific data set, which will be available at pysatNASA

The real-time data is stored by generation date, where each file contains the
data for the current day.  If you leave download dates empty, though, it will
grab today's file three times and assign dates from yesterday, today, and
tomorrow.
::

    mag = pysat.Instrument('ace', 'mag', tag='realtime')
    mag.download(start=mag.today())
    mag.load(date=mag.today())



Warnings
--------
The real-time data should not be used with the data padding option available
from pysat.Instrument objects.

"""

import datetime as dt
import functools
import numpy as np

from pysat import logger

from pysatSpaceWeather.instruments.methods import ace as mm_ace

# ----------------------------------------------------------------------------
# Instrument attributes

platform = 'ace'
name = 'mag'
tags = {'realtime': 'Real-time data from the SWPC',
        'historic': ' Historic data from the SWPC'}
inst_ids = {inst_id: [tag for tag in tags.keys()] for inst_id in ['']}

# Define today's date
now = dt.datetime.utcnow()

# ----------------------------------------------------------------------------
# Instrument test attributes

# Set test dates (first level: inst_id, second level: tag)
_test_dates = {inst_id: {'realtime': dt.datetime(now.year, now.month, now.day),
                         'historic': dt.datetime(2009, 1, 1)}
               for inst_id in inst_ids.keys()}

# ----------------------------------------------------------------------------
# Instrument methods

preprocess = mm_ace.preprocess


def init(self):
    """Initializes the Instrument object with instrument specific values.

    Runs once upon instantiation.

    """

    # Set the appropraite acknowledgements and references
    self.acknowledgements = mm_ace.acknowledgements()
    self.references = mm_ace.references(self.name)

    logger.info(self.acknowledgements)

    return


def clean(self):
    """Routine to clean real-time ACE data using the status flag

    Note
    ----
    Supports 'clean' and 'dirty'.  Replaces all fill values with NaN.
    Clean - status flag of zero (nominal data)
    Dirty - status flag < 9 (accepts bad data record, removes no data record)

    """
    # Perform the standard ACE cleaning
    max_status = mm_ace.clean(self)

    # Replace bad values with NaN and remove times with no valid data
    self.data = self.data[self.data['status'] <= max_status]

    return


# ----------------------------------------------------------------------------
# Instrument functions

download = functools.partial(mm_ace.download, name=name, now=now)
list_files = functools.partial(mm_ace.list_files, name=name)


def load(fnames, tag=None, inst_id=None):
    """Load the ACE space weather prediction data

    Parameters
    ----------
    fnames : array-like
        Series, list, or array of filenames
    tag : str or NoneType
        tag or None (default=None)
    inst_id : str or NoneType
        ACE instrument or None (default=None)

    Returns
    -------
    data : pandas.DataFrame
        Object containing instrument data
    meta : pysat.Meta
        Object containing metadata such as column names and units

    Raises
    ------
    ValueError
        When unknown inst_id is supplied.

    Note
    ----
    Called by pysat. Not intended for direct use by user.

    """

    # Save each file to the output DataFrame
    data = mm_ace.load_csv_data(fnames, read_csv_kwargs={'index_col': 0,
                                                         'parse_dates': True})

    # Assign the meta data
    meta, status_desc = mm_ace.common_metadata()

    meta['status'] = {meta.labels.units: '',
                      meta.labels.name: 'Status',
                      meta.labels.notes: '',
                      meta.labels.desc: status_desc,
                      meta.labels.fill_val: np.nan,
                      meta.labels.min_val: 0,
                      meta.labels.max_val: 9}
    meta['bx_gsm'] = {meta.labels.units: 'nT',
                      meta.labels.name: 'Bx GSM',
                      meta.labels.notes: '',
                      meta.labels.desc: '1-min averaged IMF Bx',
                      meta.labels.fill_val: -999.9,
                      meta.labels.min_val: -np.inf,
                      meta.labels.max_val: np.inf}
    meta['by_gsm'] = {meta.labels.units: 'nT',
                      meta.labels.name: 'By GSM',
                      meta.labels.notes: '',
                      meta.labels.desc: '1-min averaged IMF By',
                      meta.labels.fill_val: -999.9,
                      meta.labels.min_val: -np.inf,
                      meta.labels.max_val: np.inf}
    meta['bz_gsm'] = {meta.labels.units: 'nT',
                      meta.labels.notes: '',
                      meta.labels.name: 'Bz GSM',
                      meta.labels.desc: '1-min averaged IMF Bz',
                      meta.labels.fill_val: -999.9,
                      meta.labels.min_val: -np.inf,
                      meta.labels.max_val: np.inf}
    meta['bt_gsm'] = {meta.labels.units: 'nT',
                      meta.labels.name: 'Bt GSM',
                      meta.labels.notes: '',
                      meta.labels.desc: '1-min averaged IMF Bt',
                      meta.labels.fill_val: -999.9,
                      meta.labels.min_val: -np.inf,
                      meta.labels.max_val: np.inf}
    meta['lat_gsm'] = {meta.labels.units: 'degrees',
                       meta.labels.name: 'GSM Lat',
                       meta.labels.notes: '',
                       meta.labels.desc: 'GSM Latitude',
                       meta.labels.fill_val: -999.9,
                       meta.labels.min_val: -90.0,
                       meta.labels.max_val: 90.0}
    meta['lon_gsm'] = {meta.labels.units: 'degrees',
                       meta.labels.name: 'GSM Lon',
                       meta.labels.notes: '',
                       meta.labels.desc: 'GSM Longitude',
                       meta.labels.fill_val: -999.9,
                       meta.labels.min_val: 0.0,
                       meta.labels.max_val: 360.0}

    return data, meta
