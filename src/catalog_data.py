"""Hand-curated investigation facts, written by Claude Code from source text only.

GUIDE = "A Researcher's Guide to: Combustion Science", NASA ISS Program Science Office,
NP-2015-10-034-JSC (data/raw/np-2015-10-034-jsc_...pdf). `p.N` = PDF page N.
Rules: a field is filled only if the cited source states it; otherwise it is None.
Numeric ranges for investigations with a PSI table are NOT typed here: src/catalog.py computes
them from the table (source = PSI ID). Ranges typed here carry `range_basis` + `range_source`.
"""
from __future__ import annotations

# Distance units shared by ranking and the recommender: 5 O2 percentage points or 0.2 atm.
O2_SCALE, P_SCALE = 0.05, 0.2

GUIDE = "NASA ISS Researcher's Guide: Combustion Science (NP-2015-10-034-JSC)"
MDCA_SENSORS = "analog color camera, radiometer, low-light-level UV camera, two HiBMs cameras (p.52)"
ACME_SENSORS = ("radiometers, two color cameras, low-light-level UV camera, two HiBMs cameras, "
                "photomultiplier tubes, ion current measurement, gas chromatograph (p.52)")
MSG_SENSORS = "digital still camera, video camera, air flow, fuel flow, radiometer (p.52)"

INVESTIGATIONS: list[dict] = [
    dict(key="flex", name="Flame Extinguishment Experiment (FLEX)", psi_id="PSI-69", fuel_type="droplet",
         fuels="n-heptane, methanol",
         objective="Study heptane and methanol droplets in O2/N2 mixtures diluted with a second inert gas "
                   "(nitrogen, carbon dioxide, helium or xenon) to see how an inert suppressant changes the "
                   "flammability limit of the two fuels.",
         what_varied="Ambient O2, inert diluent type and amount, pressure, droplet size.",
         flow_min_cms=0.0, flow_max_cms=3.0, flow_basis="tested",
         flow_source="GUIDE p.33 (quiescent ambient); p.12 Fig. 2 (droplet translated at 3 cm/s)",
         sensors=MDCA_SENSORS, data_available="table",
         key_result="The flammability limit is lower in reduced gravity; the limiting oxygen index (LOI) is "
                    "significantly lower than in normal gravity and lower than predicted. Stable cool flames "
                    "form after radiative extinction of heptane droplets.",
         implication="Terrestrial standards for inert-gas suppressants do not offer the same margin of safety "
                     "in low gravity as in normal gravity.",
         suppression_detection=True, supp_source="GUIDE p.12 (inert-gas suppressant)",
         source=GUIDE + "; PSI-69 (DOI 10.60555/mbq8-0451)", source_page="12-13, 33-34, 36, 52"),
    dict(key="flex2", name="FLEX-2", psi_id=None, fuel_type="droplet",
         fuels="mixtures of pure fuels, real-fuel surrogates; ethanol shown in Fig. 3",
         objective="Follow-on to FLEX: study fuel mixtures, real-fuel surrogates, soot formation, slow "
                   "sub-buoyant convective flow and droplet-droplet interactions.",
         what_varied="Fuel type and mixture, diluent, convective flow, droplet spacing.",
         o2_max=0.40, p_min_atm=0.0, p_max_atm=3.0, range_basis="facility capability (MDCA), not tested",
         range_source="GUIDE p.12 (MDCA: 0-3 atm, up to 40% O2)",
         sensors=MDCA_SENSORS, data_available="none",
         key_result="Fig. 3 shows sooting propensity differs between diluents for ethanol burning in 30% O2 "
                    "at 2.4 atm.",
         implication=None, suppression_detection=None,
         source=GUIDE, source_page="12-13, 36, 52"),
    dict(key="lsp", name="Laminar Soot Processes (LSP)", psi_id=None, fuel_type="gas", fuels=None,
         objective="Study soot processes in laminar gas-jet diffusion flames in a quiescent (ambient) "
                   "configuration inside a sealed chamber.",
         what_varied=None, sensors=None, data_available="none",
         key_result=None, implication=None, suppression_detection=None,
         source=GUIDE, source_page="13"),
    dict(key="tgdf", name="Turbulent Gas-jet Diffusion Flame (TGDF)", psi_id=None, fuel_type="gas", fuels=None,
         objective="Study a laminar gas-jet diffusion flame in which an iris at its base induced vortices "
                   "in the entrained air (flame-vortex interaction).",
         what_varied="Vortex forcing by the iris mechanism.", sensors=None, data_available="none",
         key_result=None, implication=None, suppression_detection=None,
         source=GUIDE, source_page="13, 37"),
    dict(key="elf", name="Enclosed Laminar Flames (ELF)", psi_id=None, fuel_type="gas", fuels=None,
         objective="Study flame stability of laminar gas-jet diffusion flames inside a fan-driven flow duct "
                   "(crew operated in the Middeck Glovebox).",
         what_varied=None, sensors=None, data_available="none",
         key_result=None, implication=None, suppression_detection=None,
         source=GUIDE, source_page="13-14"),
    dict(key="spice", name="Smoke Point In Coflow Experiment (SPICE)", psi_id=None, fuel_type="gas", fuels=None,
         objective="Study the transition from non-sooting to sooting coflow flames, aiming at simple "
                   "predictions of soot emissions (smoke point).",
         what_varied="Fuel flow rate (Fig. 4: non-smoking to smoking tip as fuel flow increases).",
         sensors=MSG_SENSORS, data_available="none",
         key_result="Flames showed the transition from non-smoking to a fully open, smoking flame tip as "
                    "fuel flow increased (Fig. 4).",
         implication=None, suppression_detection=None,
         source=GUIDE + "; Dotson et al., Fire Safety Journal 46 (2011) 550-555", source_page="13-14, 45, 49, 52"),
    dict(key="slice", name="Structure & Liftoff In Combustion Experiment (SLICE)", psi_id=None, fuel_type="gas",
         fuels=None,
         objective="Study lifting and lifted flames that detach from the burner and stabilize downstream, to "
                   "improve combustion models for fuel efficiency and lower pollutant emission.",
         what_varied="Coflow velocity (Fig. 5).", sensors=MSG_SENSORS, data_available="none",
         key_result=None, implication=None, suppression_detection=None,
         source=GUIDE, source_page="14-15, 45, 52"),
    dict(key="bre", name="Burning Rate Emulator (BRE)", psi_id=None, fuel_type="gas",
         fuels="gaseous fuels emulating paper, plastic, alcohol",
         objective="Improve understanding of materials flammability (ignition, extinction) and assess "
                   "existing flammability test methods for low and partial gravity, using a porous burner "
                   "whose gas flow mimics a burning solid or liquid.",
         what_varied="Gaseous fuel and flow rate (controlled from measured heat flux and surface temperature).",
         sensors=ACME_SENSORS, data_available="none",
         key_result=None,
         implication="Focused on fire prevention in spacecraft; tests the relevance of flammability test "
                     "methods for low- and partial-gravity environments.",
         suppression_detection=None, source=GUIDE, source_page="14-15, 44, 52"),
    dict(key="cld", name="Coflow Laminar Diffusion Flame (CLD Flame)", psi_id=None, fuel_type="gas",
         fuels="diluted methane, ethylene",
         objective="Extend the range of flames that can be predicted (very dilute near-extinction and sooty "
                   "flames) by developing and verifying kinetic and soot submodels.",
         what_varied="Fuel injection velocity and fuel dilution.", sensors=ACME_SENSORS, data_available="none",
         key_result=None, implication=None, suppression_detection=None,
         source=GUIDE, source_page="16, 44, 52"),
    dict(key="efield", name="Electric-Field Effects on Laminar Diffusion Flames (E-FIELD Flames)", psi_id=None,
         fuel_type="gas", fuels=None,
         objective="Understand flame-ion production and how ions can be used to control non-premixed flames.",
         what_varied="Electric field strength (up to 10 kV) and fuel dilution.",
         sensors=ACME_SENSORS, data_available="none",
         key_result=None, implication=None, suppression_detection=None,
         source=GUIDE, source_page="16, 44, 52"),
    dict(key="flame_design", name="Flame Design (ACME)", psi_id="PSI-10", fuel_type="gas",
         fuels="ethylene (PSI-10 table); ethylene or methane planned",
         objective="Understand soot inception and control so that oxygen-enriched, robust, soot-free "
                   "non-premixed flames can be designed; normal and inverse spherical flames.",
         what_varied="Injected fuel concentration and oxygen enrichment of the atmosphere; flow direction "
                     "(normal vs inverse flame).",
         sensors=ACME_SENSORS, data_available="table",
         key_result=None, implication=None, suppression_detection=None,
         source=GUIDE + "; PSI-10", source_page="16-17, 44, 52"),
    dict(key="sflame", name="Structure and Response of Spherical Diffusion Flames (s-Flame)", psi_id=None,
         fuel_type="gas", fuels="hydrogen, methane, ethylene",
         objective="Predict the structure and dynamics, including extinction, of soot-free and sooty "
                   "spherical diffusion flames on a porous spherical burner.",
         what_varied="Fuel, inert diluents in fuel and chamber atmosphere.",
         sensors=ACME_SENSORS, data_available="none",
         key_result=None, implication=None, suppression_detection=None,
         source=GUIDE, source_page="17-18, 44, 52"),
    dict(key="bass", name="Burning and Suppression of Solids (BASS)", psi_id="PSI-26", fuel_type="solid",
         fuels="flat samples, PMMA spheres, porous wax slabs, wax candles",
         objective="Examine burning and extinction of a wide variety of solid fuel samples in microgravity, "
                   "to improve 1-g material fire-safety tests and guide extinguishing strategies.",
         what_varied="Airflow speed (main variable), oxygen (nitrogen dilution), nitrogen suppressant jet.",
         o2_min=0.17, o2_max=0.21, p_min_atm=1.0, p_max_atm=1.0, range_basis="tested",
         range_source="GUIDE p.19 (glovebox ISS air ~21% O2, one atmosphere; tests as low as 17% O2)",
         flow_min_cms=None, flow_max_cms=12.0, flow_basis="tested",
         flow_source="GUIDE p.20 Fig. 11 (12 cm/s); Fig. 12 flow decreased from 10 to 'less than 1 cm/s' "
                     "(lower bound not given as a number)",
         sensors=MSG_SENSORS, data_available="table",
         key_result="NASA-STD-6001 Test 1 is not conservative: materials can burn at lower oxygen in "
                    "microgravity than on Earth. Local suppressant application was not adequate; the jet "
                    "entrained air and sustained the flame, and nitrogen could even strengthen it.",
         implication="Materials screened as safe in 1-g may burn in a spacecraft; extinguisher placement "
                     "needs rethinking because a suppressant jet can feed a microgravity flame.",
         suppression_detection=True, supp_source="GUIDE p.22, p.31 (nitrogen suppressant)",
         source=GUIDE + "; PSI-26 (sample list only)", source_page="18-22, 31, 45, 52"),
    dict(key="bass2", name="BASS-II", psi_id="PSI-25", fuel_type="solid", fuels=None,  # filled from table
         objective=None, what_varied=None,
         sensors="radiometer, camera frames, O2/CO2/CO gas analysis (PSI-25 columns)", data_available="table",
         key_result=None, implication=None, suppression_detection=None,
         source="PSI-25", source_page=None),
    dict(key="sofie", name="Solid Fuel Ignition and Extinction (SoFIE)", psi_id=None, fuel_type="solid",
         fuels="plastics such as PMMA or Delrin (likely candidates)",
         objective="Study and characterize ignition and flammability of solid spacecraft materials in "
                   "practical geometries and realistic atmospheres (up to five investigations).",
         what_varied="Material, geometry, external heating, oxygen, pressure, flow speed.",
         o2_min=0.10, o2_max=0.40, p_min_atm=0.2, p_max_atm=1.5, range_basis="planned",
         range_source="GUIDE p.19 (10-40% O2, 0.2-1.5 atm)",
         flow_min_cms=0.0, flow_max_cms=30.0, flow_basis="planned",
         flow_source="GUIDE p.21 (spacecraft ventilation range 0-30 cm/s)",
         sensors="same as ACME: " + ACME_SENSORS, data_available="none",
         key_result=None,
         implication="Planned to test Space Exploration Atmospheres (reduced pressure, elevated oxygen) and "
                     "evaluate oxygen margins of NASA-STD-6001 Test 1.",
         suppression_detection=True, supp_source="GUIDE p.24 (ignition and suppression with external heat flux)",
         source=GUIDE, source_page="19, 21-25, 38, 44, 52"),
    dict(key="sofball", name="Structure of Flame Balls at Low Lewis-number (SOFBALL)", psi_id=None,
         fuel_type="gas", fuels="very fuel-lean premixed mixtures",
         objective="Test whether stationary spherical premixed flames (flame balls) can exist in very lean, "
                   "low-Lewis-number mixtures.",
         what_varied="Premixed fuel/oxidizer/diluent mixture.", sensors=None, data_available="none",
         key_result="Flame balls exist: the weakest flames ever burned (about 1 W), some lasting over 40 "
                    "minutes until the experiment timed out.",
         implication=None, suppression_detection=None, source=GUIDE, source_page="25, 30"),
    dict(key="mist", name="Water Mist Fire Suppression (Mist)", psi_id=None, fuel_type="gas",
         fuels="premixed flames in a tube",
         objective="Measure how water droplet diameter and water concentration affect the speed, strength "
                   "and shape of premixed flames, to design better mist suppression systems replacing Halon.",
         what_varied="Water droplet diameter, water concentration, fuel and oxygen mixture.",
         sensors=None, data_available="none",
         key_result="Led to non-toxic water-mist portable fire extinguishers certified for spaceflight; the "
                    "first two of nine reached the ISS in December 2015 to replace CO2 units.",
         implication="Direct path from microgravity research to a crew fire-suppression tool.",
         suppression_detection=True, supp_source="GUIDE p.26", source=GUIDE, source_page="26"),
    dict(key="same", name="Smoke Aerosol Measurement Experiment (SAME)", psi_id=None, fuel_type="solid",
         fuels="smoke from spacecraft materials",
         objective="Measure the particle-size distribution of smoke from spacecraft materials.",
         what_varied="Spacecraft material.", sensors=None, data_available="none",
         key_result="Microgravity flames produce very different particulate sizes, so detectors that sense "
                    "particulates need different signatures in space.",
         implication="Guidance for designers of future spacecraft smoke detectors.",
         suppression_detection=True, supp_source="GUIDE p.26, p.35 (fire detection)",
         source=GUIDE, source_page="26, 35, 45"),
    dict(key="candle", name="Candle Flames (Mir)", psi_id=None, fuel_type="solid", fuels="candle wax",
         objective="Observe a candle flame in long-duration microgravity aboard Mir.",
         what_varied=None, sensors=None, data_available="none",
         key_result="Microgravity candle flames burned over 45 minutes, to a lower ambient oxygen than an "
                    "identical candle on Earth, and oscillated before extinction.",
         implication="Refutes the belief that a steady fire cannot exist in quiescent microgravity.",
         suppression_detection=None, source=GUIDE + "; Dietrich et al. (2000)", source_page="9, 30, 33"),
    dict(key="cool_flames", name="Cool Flames (CIR initiative)", psi_id=None, fuel_type="droplet",
         fuels="range of fuel types",
         objective="Find the pre-conditions for hot-flame/cool-flame transitions, cool-flame stability and "
                   "spontaneous re-ignition, using existing FLEX-2 MDCA hardware.",
         what_varied="Fuel type, pressure, diluent concentration.", sensors=MDCA_SENSORS, data_available="none",
         key_result="Builds on FLEX: after hot-flame extinction, heptane droplets kept burning with a "
                    "low-temperature cool flame.",
         implication="Very weak, persistent flames invisible to the eye can spontaneously re-ignite long after "
                     "the visible flame is gone.",
         suppression_detection=None, source=GUIDE, source_page="33-34, 41-42"),
    dict(key="cfig", name="CFI-G (normal flames)", psi_id="PSI-159", fuel_type="gas",
         fuels=None, objective=None, what_varied=None, sensors=None, data_available="table",
         key_result=None, implication=None, suppression_detection=None, source="PSI-159", source_page=None),
    dict(key="saffire1", name="Saffire-I", psi_id="PSI-98", fuel_type="solid", fuels=None,
         objective=None, what_varied=None, sensors="two cameras per sample (PSI-98 columns)",
         data_available="table", key_result=None, implication=None, suppression_detection=None,
         source="PSI-98", source_page=None),
    dict(key="saffire2", name="Saffire-II", psi_id="PSI-99", fuel_type="solid", fuels=None,
         objective=None, what_varied=None, sensors="cameras (PSI-99 columns)",
         data_available="table", key_result=None, implication=None, suppression_detection=None,
         source="PSI-99", source_page=None),
    dict(key="saffire3", name="Saffire-III", psi_id="PSI-100", fuel_type="solid", fuels=None,
         objective=None, what_varied=None, sensors="two cameras per sample (PSI-100 columns)",
         data_available="table", key_result=None, implication=None, suppression_detection=None,
         source="PSI-100", source_page=None),
]

# Guide p.35, quoted list.
MISCONCEPTIONS = [
    "A steady fire cannot exist in a quiescent microgravity environment.",
    "A fire will self-extinguish if ventilation is shut off.",
    "Because the reduced-gravity environment is inherently safer, terrestrial fire standards and tests "
    "represent a very conservative approach to fire safety.",
]

# Literature points: never PSI data, never training data. `value_verified` is False when the number
# comes from the project brief (CLAUDE.md) and the primary text could not be read in this session.
LITERATURE_POINTS: list[dict] = [
    dict(point_id="bass2_concurrent_ug", kind="literature", label="BASS-II steady concurrent spread (microgravity)",
         fuel_type="solid", x_o2=0.136, p_atm=None, p_assumed_atm=1.0, flow_cms=None,
         note="Microgravity concurrent-flow spread at 13.6% O2 vs 1-g limit 16%: oxygen margin -2.4 points.",
         citation="Candidate source: 'Flammability Limits from BASS-II Testing in Microgravity Compared to "
                  "Normal Gravity Limits', ICES 2019 (Texas Tech repository).",
         value_verified=False),
    dict(point_id="bass2_blowoff_1g", kind="literature", label="BASS-II material 1-g limit (normal gravity)",
         fuel_type="solid", x_o2=0.16, p_atm=None, p_assumed_atm=1.0, flow_cms=None,
         note="Normal-gravity comparison point for the BASS-II microgravity limit.",
         citation="Same candidate source as bass2_concurrent_ug.", value_verified=False),
    dict(point_id="saffire6_sibal", kind="literature", label="Saffire-VI SIBAL fabric burn",
         fuel_type="solid", x_o2=0.30, p_atm=0.54, p_assumed_atm=None, flow_cms=None,
         note="Large-scale fabric fire in an elevated-O2, reduced-pressure atmosphere (Cygnus).",
         citation="NASA Task Book (per project brief). Candidate: 'Preliminary Results from the Saffire VI "
                  "Experiment', ICES-2024-365, NTRS 20240002981. NASA describes the final run as higher O2 "
                  "and lower pressure; exact values not confirmed here.",
         value_verified=False),
    dict(point_id="target_moon_34_8_2", kind="exploration_target", label="Exploration atmosphere 34% O2 / 8.2 psia",
         fuel_type=None, x_o2=0.34, p_atm=round(8.2 * 0.0680460, 3), p_assumed_atm=None, flow_cms=None,
         note="Lunar/Mars habitat atmosphere (JSC 29353D: min 33.7% O2 at 8.2 psia).",
         citation="NASA/TM-2013-217377 (NTRS record: 8.0 psia/32% O2 later revised to 8.2 psia/34% O2); "
                  "JSC 29353D as cited in the project brief (not verified here).",
         value_verified=True),
    dict(point_id="target_challenge_30_10_2", kind="exploration_target", label="Challenge target >=30% O2 at <=10.2 psi",
         fuel_type=None, x_o2=0.30, p_atm=round(10.2 * 0.0680460, 3), p_assumed_atm=None, flow_cms=None,
         note="Upper-pressure / lower-O2 corner of the challenge's exploration atmosphere.",
         citation="NASA Space Apps 2026 'Flame in Freefall' challenge page (as quoted in the project brief).",
         value_verified=False),
]
