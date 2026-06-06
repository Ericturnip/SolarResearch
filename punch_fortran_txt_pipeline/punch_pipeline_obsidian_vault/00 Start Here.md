# PUNCH Pipeline Notes

This vault documents the FITS to tomography TXT pipeline. It is meant for future
debugging, handoff to a PI, and reminding yourself why the scripts behave the
way they do.

Start with:

- [[01 Pipeline Flow]]
- [[02 Commands]]
- [[04 Fortran TXT Contract]]
- [[06 Debugging Missing Data]]

The code repo README is the shortest command reference. These notes keep the
reasoning and operational details.

## Current Rule For Hourly p25

The hourly p25 map does not write an interpolated percentile brightness. It
computes p25 as a target, picks the real binned sample closest to that target,
and writes that sample with its own timestamp.

That matters because the Fortran code sees brightness and time in the same row.
