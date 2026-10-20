"""
select_period: the start/end window of plot_hourly_prod_by_tech.

The default dates ("01-01 00:00:00" to "12-29 23:00:00") never worked before
October 2026: pandas reads a date without a year as year 1. These tests use
an hourly series like the plotted ones: 8760 hours from 1 January.
"""
import pandas as pd
import pytest

from hypatia.analysis.plots import select_period


def hourly(year):
    return pd.Series(1.0, index=pd.date_range(f"{year}-01-01", periods=8760, freq="1h"))


def test_default_dates_select_the_model_year():
    selected = select_period(hourly(2030), 2030, "01-01 00:00:00", "12-29 23:00:00")
    assert selected.index[0] == pd.Timestamp("2030-01-01 00:00")
    assert selected.index[-1] == pd.Timestamp("2030-12-29 23:00")
    assert len(selected) == 363 * 24


def test_full_dates_are_used_as_they_are():
    selected = select_period(hourly(2030), 2030, "2030-03-01 00:00:00", "2030-03-01 23:00:00")
    assert len(selected) == 24


def test_wrong_date_keeps_the_original_error():
    with pytest.raises(ValueError, match="incorrect start or end") as caught:
        select_period(hourly(2030), 2030, "13-45 00:00:00", "12-29 23:00:00")
    assert caught.value.__cause__ is not None   # what pandas said is not lost


def test_interrupting_is_not_swallowed():
    # A bare "except:" also caught KeyboardInterrupt (Ctrl+C) and reported it
    # as a wrong date.
    class Interrupting:
        def __str__(self):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        select_period(hourly(2030), 2030, Interrupting(), "12-29 23:00:00")
