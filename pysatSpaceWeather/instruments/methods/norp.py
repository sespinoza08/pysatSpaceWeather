#!/usr/bin/env python
# -*- coding: utf-8 -*-.
# Full license can be found in License.md
# Full author list can be found in .zenodo.json file
# DOI:10.5281/zenodo.3986138
# ----------------------------------------------------------------------------

"""Routines for the norp solar index."""

import datetime as dt
import numpy as np
from packaging.version import Version

import pandas as pds
import pysat

import pysatSpaceWeather as pysat_sw


def acknowledgements(tag):
    """Define the acknowledgements for the norp data.

    Parameters
    ----------
    tag : str
        Tag of the space waether index

    Returns
    -------
    ackn : str
        Acknowledgements string associated with the appropriate norp tag.

    """
    norp = ''.join(['Prepared by the National Astronomical Observatory of ',
                    'Japan (NAO), Solar Science Observatory (SSO), Nobeyama ',
                    'Radio Polarimeters (NoRP)'])

    ackn = {'fits': norp, 'calibrated': norp, 'historic': norp}

    return ackn[tag]


def references(tag):
    """Define the references for the norp data.

    Parameters
    ----------
    tag : str
        Instrument tag for the norp data.

    Returns
    -------
    refs : str
        Reference string associated with the appropriate norp tag.

    """
    norp_desc = ''.join(['Dataset description: ',
                         'https://solar.nro.nao.ac.jp/norp/html/',
                         'fits_png_ql/ReadMe.pdf, accessed June 2023'])
    orig_ref = ''.join(['TD de Wit (2017), The 30 cm radio flux as a solar ',
                        'proxy for thermosphere density modelling, '
                        'J. Space Weather Space Clim. 7 A9 (2017), '
                        'DOI: 10.1051/swsc/2017008'])

    refs = {'fits': "\n".join([norp_desc, orig_ref]),
            'calibrated': "\n".join([norp_desc, orig_ref]),
            'historic': "\n".join([norp_desc, orig_ref])}

    return refs[tag]


def combine_norp(standard_inst, forecast_inst, start=None, stop=None):
    """Combine the output from the measured and forecasted norp sources.

    Parameters
    ----------
    standard_inst : pysat.Instrument or NoneType
        Instrument object containing data for the 'sw' platform, 'norp' name,
        and 'historic', 'prelim', or 'daily' tag
    forecast_inst : pysat.Instrument or NoneType
        Instrument object containing data for the 'sw' platform, 'norp' name,
        and 'prelim', '45day' or 'forecast' tag
    start : dt.datetime or NoneType
        Starting time for combining data, or None to use earliest loaded
        date from the pysat Instruments (default=None)
    stop : dt.datetime or NoneType
        Ending time for combining data, or None to use the latest loaded date
        from the pysat Instruments (default=None)

    Returns
    -------
    norp_inst : pysat.Instrument
        Instrument object containing norp observations for the desired period
        of time, merging the standard, 45day, and forecasted values based on
        their reliability

    Raises
    ------
    ValueError
        If appropriate time data is not supplied, or if the date range is badly
        formed.

    Notes
    -----
    Merging prioritizes the standard data, then the 45day data, and finally
    the forecast data

    Will not attempt to download any missing data, but will load data

    """

    # Initialize metadata and flags
    notes = "Combines data from"
    stag = standard_inst.tag if len(standard_inst.tag) > 0 else 'default'
    tag = 'combined_{:s}_{:s}'.format(stag, forecast_inst.tag)
    inst_flag = 'standard'

    # If the start or stop times are not defined, get them from the Instruments
    if start is None:
        stimes = [inst.index.min() for inst in [standard_inst, forecast_inst]
                  if len(inst.index) > 0]
        start = min(stimes) if len(stimes) > 0 else None

    if stop is None:
        stimes = [inst.index.max() for inst in [standard_inst, forecast_inst]
                  if len(inst.index) > 0]
        stop = max(stimes) + pds.DateOffset(days=1) if len(stimes) > 0 else None

    if start is None or stop is None:
        raise ValueError(' '.join(("must either load in Instrument objects or",
                                   "provide starting and ending times")))

    if start >= stop:
        raise ValueError("date range is zero or negative")

    # Initialize the output instrument
    norp_inst = pysat.Instrument()
    norp_inst.inst_module = pysat_sw.instruments.sw_norp
    norp_inst.tag = tag
    norp_inst.date = start
    norp_inst.doy = np.int64(start.strftime("%j"))
    fill_val = None

    norp_times = list()
    norp_values = list()

    # Cycle through the desired time range
    itime = start

    while itime < stop and inst_flag is not None:
        # Load and save the standard data for as many times as possible
        if inst_flag == 'standard':
            # Test to see if data loading is needed
            if not np.any(standard_inst.index == itime):
                # Set the load kwargs, which vary by pysat version and tag
                load_kwargs = {'date': itime}

                if Version(pysat.__version__) > Version('3.0.1'):
                    load_kwargs['use_header'] = True

                if standard_inst.tag == 'daily':
                    # Add 30 days
                    load_kwargs['date'] += dt.timedelta(days=30)

                standard_inst.load(**load_kwargs)

            if standard_inst.empty:
                good_times = [False]
            else:
                good_times = ((standard_inst.index >= itime)
                              & (standard_inst.index < stop))

            if notes.find("standard") < 0:
                notes += " the {:} source ({:} to ".format(inst_flag,
                                                           itime.date())

            if np.any(good_times):
                if fill_val is None:
                    norp_inst.meta = standard_inst.meta
                    fill_val = norp_inst.meta['norp'][
                        norp_inst.meta.labels.fill_val]

                good_vals = standard_inst['norp'][good_times] != fill_val
                new_times = list(standard_inst.index[good_times][good_vals])
                norp_times.extend(new_times)
                new_vals = list(standard_inst['norp'][good_times][good_vals])
                norp_values.extend(new_vals)
                itime = norp_times[-1] + pds.DateOffset(days=1)
            else:
                inst_flag = 'forecast'
                notes += "{:})".format(itime.date())

        # Load and save the forecast data for as many times as possible
        if inst_flag == "forecast":
            # Determine which files should be loaded
            if len(forecast_inst.index) == 0:
                files = np.unique(forecast_inst.files.files[itime:stop])
            else:
                files = [None]  # No load needed, if already initialized

            # Cycle through all possible files of interest, saving relevant
            # data
            for filename in files:
                if filename is not None:
                    load_kwargs = {'fname': filename}
                    if Version(pysat.__version__) > Version('3.0.1'):
                        load_kwargs['use_header'] = True

                    forecast_inst.load(**load_kwargs)

                if notes.find("forecast") < 0:
                    notes += " the {:} source ({:} to ".format(inst_flag,
                                                               itime.date())

                # Check in case there was a problem with the standard data
                if fill_val is None:
                    norp_inst.meta = forecast_inst.meta
                    fill_val = norp_inst.meta['norp'][
                        norp_inst.meta.labels.fill_val]

                # Determine which times to save
                if forecast_inst.empty:
                    good_vals = []
                else:
                    good_times = ((forecast_inst.index >= itime)
                                  & (forecast_inst.index < stop))
                    good_vals = forecast_inst['norp'][good_times] != fill_val

                # Save desired data and cycle time
                if len(good_vals) > 0:
                    new_times = list(
                        forecast_inst.index[good_times][good_vals])
                    norp_times.extend(new_times)
                    new_vals = list(
                        forecast_inst['norp'][good_times][good_vals])
                    norp_values.extend(new_vals)
                    itime = norp_times[-1] + pds.DateOffset(days=1)

            notes += "{:})".format(itime.date())

            inst_flag = None

    if inst_flag is not None:
        notes += "{:})".format(itime.date())

    # Determine if the beginning or end of the time series needs to be padded
    if len(norp_times) >= 2:
        freq = pysat.utils.time.calc_freq(norp_times)
    else:
        freq = None
    end_date = stop - pds.DateOffset(days=1)
    date_range = pds.date_range(start=start, end=end_date, freq=freq)

    if len(norp_times) == 0:
        norp_times = date_range

    date_range = pds.date_range(start=start, end=end_date, freq=freq)

    if date_range[0] < norp_times[0]:
        # Extend the time and value arrays from their beginning with fill
        # values
        itime = abs(date_range - norp_times[0]).argmin()
        norp_times.reverse()
        norp_values.reverse()
        extend_times = list(date_range[:itime])
        extend_times.reverse()
        norp_times.extend(extend_times)
        norp_values.extend([fill_val for kk in extend_times])
        norp_times.reverse()
        norp_values.reverse()

    if date_range[-1] > norp_times[-1]:
        # Extend the time and value arrays from their end with fill values
        itime = abs(date_range - norp_times[-1]).argmin() + 1
        extend_times = list(date_range[itime:])
        norp_times.extend(extend_times)
        norp_values.extend([fill_val for kk in extend_times])

    # Save output data
    norp_inst.data = pds.DataFrame(norp_values, columns=['norp'],
                                   index=norp_times)

    # Resample the output data, filling missing values
    if (date_range.shape != norp_inst.index.shape
            or abs(date_range - norp_inst.index).max().total_seconds() > 0.0):
        norp_inst.data = norp_inst.data.resample(freq).fillna(method=None)
        if np.isfinite(fill_val):
            norp_inst.data[np.isnan(norp_inst.data)] = fill_val

    # Update the metadata notes for this procedure
    notes += ", in that order"
    norp_inst.meta['norp'] = {norp_inst.meta.labels.notes: notes}

    return norp_inst


def parse_45day_block(block_lines):
    """Parse the data blocks used in the 45-day Ap and norp Flux Forecast file.

    Parameters
    ----------
    block_lines : list
        List of lines containing data in this data block

    Returns
    -------
    dates : list
        List of dates for each date/data pair in this block
    values : list
        List of values for each date/data pair in this block

    """

    # Initialize the output
    dates = list()
    values = list()

    # Cycle through each line in this block
    for line in block_lines:
        # Split the line on whitespace
        split_line = line.split()

        # Format the dates
        dates.extend([dt.datetime.strptime(tt, "%d%b%y")
                      for tt in split_line[::2]])

        # Format the data values
        values.extend([np.int64(vv) for vv in split_line[1::2]])

    return dates, values


def rewrite_daily_file(year, outfile, lines):
    """Rewrite the SWPC Daily Solar Data files.

    Parameters
    ----------
    year : int
        Year of data file (format changes based on date)
    outfile : str
        Output filename
    lines : str
        String containing all output data (result of 'read')

    """

    # Get to the solar index data
    if year > 2000:
        raw_data = lines.split('#---------------------------------')[-1]
        raw_data = raw_data.split('\n')[1:-1]
        optical = True
    else:
        raw_data = lines.split('# ')[-1]
        raw_data = raw_data.split('\n')
        optical = False if raw_data[0].find('Not Available') or year == 1994 \
            else True
        istart = 7 if year < 2000 else 1
        raw_data = raw_data[istart:-1]

    # Parse the data
    solar_times, data_dict = parse_daily_solar_data(raw_data, year, optical)

    # Collect into DataFrame
    data = pds.DataFrame(data_dict, index=solar_times,
                         columns=data_dict.keys())

    # Write out as a file
    data.to_csv(outfile, header=True)

    return


def parse_daily_solar_data(data_lines, year, optical):
    """Parse the data in the SWPC daily solar index file.

    Parameters
    ----------
    data_lines : list
        List of lines containing data
    year : list
        Year of file
    optical : boolean
        Flag denoting whether or not optical data is available

    Returns
    -------
    dates : list
        List of dates for each date/data pair in this block
    values : dict
        Dict of lists of values, where each key is the value name

    """

    # Initialize the output
    dates = list()
    val_keys = ['norp', 'ssn', 'ss_area', 'new_reg', 'smf', 'goes_bgd_flux',
                'c_flare', 'm_flare', 'x_flare', 'o1_flare', 'o2_flare',
                'o3_flare']
    optical_keys = ['o1_flare', 'o2_flare', 'o3_flare']
    xray_keys = ['c_flare', 'm_flare', 'x_flare']
    values = {kk: list() for kk in val_keys}

    # Cycle through each line in this file
    for line in data_lines:
        # Split the line on whitespace
        split_line = line.split()

        # Format the date
        dfmt = "%Y %m %d" if year > 1996 else "%d %b %y"
        dates.append(dt.datetime.strptime(" ".join(split_line[0:3]), dfmt))

        # Format the data values
        j = 0
        for i, kk in enumerate(val_keys):
            if year == 1994 and kk == 'new_reg':
                # New regions only in files after 1994
                val = -999
            elif((year == 1994 and kk in xray_keys)
                 or (not optical and kk in optical_keys)):
                # X-ray flares in files after 1994, optical flares come later
                val = -1
            else:
                val = split_line[j + 3]
                j += 1

            if kk != 'goes_bgd_flux':
                if val == "*":
                    val = -999 if i < 5 else -1
                else:
                    val = np.int64(val)
            values[kk].append(val)

    return dates, values


def calc_norpa(norp_inst, norp_name='norp', norpa_name='norpa', min_pnts=41):
    """Calculate the 81 day mean norp.

    Parameters
    ----------
    norp_inst : pysat.Instrument
        pysat Instrument holding the norp data
    norp_name : str
        Data column name for the norp data (default='norp')
    norpa_name : str
        Data column name for the norpa data (default='norpa')
    min_pnts : int
        Minimum number of points required to calculate an average (default=41)

    Note
    ----
    Will not pad data on its own

    """

    # Test to see that the input data is present
    if norp_name not in norp_inst.data.columns:
        raise ValueError("unknown input data column: " + norp_name)

    # Test to see that the output data does not already exist
    if norpa_name in norp_inst.data.columns:
        raise ValueError("output data column already exists: " + norpa_name)

    if norp_name in norp_inst.meta:
        fill_val = norp_inst.meta[norp_name][norp_inst.meta.labels.fill_val]
    else:
        fill_val = np.nan

    # Calculate the rolling mean.  Since these values are centered but rolling
    # function doesn't allow temporal windows to be calculated this way, create
    # a hack for this.
    #
    # Ensure the data are evenly sampled at a daily frequency, since this is
    # how often norp is calculated.
    norp_fill = norp_inst.data.resample('1D').fillna(method=None)

    # Replace the time index with an ordinal
    time_ind = norp_fill.index
    norp_fill['ord'] = pds.Series([tt.toordinal() for tt in time_ind],
                                  index=time_ind)
    norp_fill.set_index('ord', inplace=True)

    # Calculate the mean
    norp_fill[norpa_name] = norp_fill[norp_name].rolling(window=81,
                                                         min_periods=min_pnts,
                                                         center=True).mean()

    # Replace the ordinal index with the time
    norp_fill['time'] = pds.Series(time_ind, index=norp_fill.index)
    norp_fill.set_index('time', inplace=True)

    # Resample to the original frequency, if it is not equal to 1 day
    freq = pysat.utils.time.calc_freq(norp_inst.index)
    if freq != "86400S":
        # Resample to the desired frequency
        norp_fill = norp_fill.resample(freq).ffill()

        # Save the output in a list
        norpa = list(norp_fill[norpa_name])

        # Fill any dates that fall
        time_ind = pds.date_range(norp_fill.index[0], norp_inst.index[-1],
                                  freq=freq)
        for itime in time_ind[norp_fill.index.shape[0]:]:
            if (itime - norp_fill.index[-1]).total_seconds() < 86400.0:
                norpa.append(norpa[-1])
            else:
                norpa.append(fill_val)

        # Redefine the Series
        norp_fill = pds.DataFrame({norpa_name: norpa}, index=time_ind)

    # There may be missing days in the output data, remove these
    if norp_inst.index.shape < norp_fill.index.shape:
        norp_fill = norp_fill.loc[norp_inst.index]

    # Save the data
    norp_inst[norpa_name] = norp_fill[norpa_name]

    # Update the metadata
    meta_dict = {norp_inst.meta.labels.units: 'SFU',
                 norp_inst.meta.labels.name: 'norpa',
                 norp_inst.meta.labels.desc: "81-day centered average of norp",
                 norp_inst.meta.labels.min_val: 0.0,
                 norp_inst.meta.labels.max_val: np.nan,
                 norp_inst.meta.labels.fill_val: fill_val,
                 norp_inst.meta.labels.notes:
                 ' '.join(('Calculated using data between',
                           '{:} and {:}'.format(norp_inst.index[0],
                                                norp_inst.index[-1])))}

    norp_inst.meta[norpa_name] = meta_dict

    return

def rewrite_historic_file(outfile, lines):
    """Rewrite the historic norp file"""

    # File contains two header lines
    raw = lines.split('\n')[2:-1]
    data = pds.DataFrame([x.split(',')[1:] for x in raw],
                        index=[x.split(',')[0][1:-1] for x in raw])
    # Have pandas parse the dates
    data.index = pds.to_datetime(data.index)
    data.index.name = lines.split('\n')[1].split(',')[0]
    data.columns = lines.split('\n')[1].split(',')[1:]

    data.to_csv(outfile, header=True)

    return