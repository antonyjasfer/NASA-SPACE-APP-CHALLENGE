# FLEX clean table — data dictionary

Source: NASA PSI-69 FLEX experimental table, DOI 10.60555/mbq8-0451 (file `data/raw/PSI-69_Experimental table_FLEX.csv`, cp1252).
Produced by `python -m src.ingest` -> `data/processed/flex_clean.csv`. 274 rows in, 274 rows out; no rows added, dropped or imputed.

| column | unit | meaning | source |
|---|---|---|---|
| `test_no` | - | FLEX test sequence number (raw 'FLEX Test #') | PSI-69, DOI 10.60555/mbq8-0451 |
| `flex_id` | - | FLEX test identifier (raw 'FLEX Identifier') | PSI-69, DOI 10.60555/mbq8-0451 |
| `test_date` | date (YYYY-MM-DD) | Test date, parsed from raw M/D/YYYY (50 rows are written M/D/YY; all 2009-2011). 80 distinct dates; used as the CV group | PSI-69, DOI 10.60555/mbq8-0451 |
| `test_gmt` | hh:mm:ss GMT | Test start time, kept as text | PSI-69, DOI 10.60555/mbq8-0451 |
| `fuel` | - | Droplet fuel: Methanol or Heptane | PSI-69, DOI 10.60555/mbq8-0451 |
| `p_mmhg` | mmHg | Ambient pressure. C11M201 raw value 0 -> NaN (physically impossible) | PSI-69, DOI 10.60555/mbq8-0451 |
| `p_atm` | atm | p_mmhg / 760 | PSI-69, DOI 10.60555/mbq8-0451 |
| `x_o2` | mole fraction | Initial ambient O2 (raw header 'O ...', subscript lost) | PSI-69, DOI 10.60555/mbq8-0451 |
| `x_n2` | mole fraction | Initial ambient N2 (raw header 'N ...') | PSI-69, DOI 10.60555/mbq8-0451 |
| `x_co2` | mole fraction | Initial ambient CO2 (raw header 'CO ...'; CO2, not carbon monoxide) | PSI-69, DOI 10.60555/mbq8-0451 |
| `x_he` | mole fraction | Initial ambient helium | PSI-69, DOI 10.60555/mbq8-0451 |
| `d0_mm` | mm | Droplet initial diameter. '–'/blank -> NaN | PSI-69, DOI 10.60555/mbq8-0451 |
| `d_ext_mm` | mm | Visible flame extinction diameter. '–'/blank -> NaN | PSI-69, DOI 10.60555/mbq8-0451 |
| `K_mm2_s` | mm^2/s | Burning-rate constant (raw header 'Burning rate; mm'; true units mm^2/s) | PSI-69, DOI 10.60555/mbq8-0451 |
| `burn_time_s` | s | Burn time. '~6', '~9' -> 6, 9 with burn_time_approx=True | PSI-69, DOI 10.60555/mbq8-0451 |
| `burn_time_approx` | bool | True where the raw burn time was written as approximate ('~') | PSI-69, DOI 10.60555/mbq8-0451 |
| `test_end` | - | Outcome: Extinction / Disruption / Completion | PSI-69, DOI 10.60555/mbq8-0451 |
| `p_was_zero` | bool | True where the raw pressure was 0 and was set to NaN | PSI-69, DOI 10.60555/mbq8-0451 |
| `dup_id` | bool | True where flex_id occurs more than once (193F001). Both rows kept | PSI-69, DOI 10.60555/mbq8-0451 |
