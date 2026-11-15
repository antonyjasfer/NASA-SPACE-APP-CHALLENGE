"""Lessons & Earth Impact (app tab 5): historical fires and NASA fire-safety spinoffs.

Each entry carries its NASA source and link. Facts are limited to what those pages state.
Also indexed into the chatbot knowledge base (src/knowledge.py).
"""
from __future__ import annotations

from src.catalog_data import GUIDE

LESSONS: list[dict] = [
    dict(key="apollo1", kind="incident", title="Apollo 1 cabin fire", year=1967,
         what_happened="During a launch-pad test the cabin was filled with high-pressure 100% oxygen; a spark "
                       "from an electrical short ignited flammable cabin material. Three astronauts died.",
         lesson="Oxygen-rich cabins turn ordinary materials into fuel: NASA changed the spacecraft (Block II), "
                "cabin materials and hatch, and designed fire-resistant spacesuits.",
         source="NASA Apollo 204 Review Board findings; NASA news release 'Fifty Years On, Apollo 1 Tragedy "
                f"Offers Vital Lessons'; {GUIDE} p. 34",
         url="https://www.nasa.gov/news-release/fifty-years-on-apollo-1-tragedy-offers-vital-lessons/"),
    dict(key="apollo13", kind="incident", title="Apollo 13 oxygen tank explosion", year=1970,
         what_happened="Wires damaged during pre-flight testing shorted inside oxygen tank no. 2 when the fans "
                       "were switched on; the Teflon insulation burned, pressure rose and the tank ruptured.",
         lesson="Ignition sources inside pure-oxygen systems are a fire hazard even far from the cabin.",
         source=f"NASA, '50 Years Ago: Apollo 13 Review Board Report'; {GUIDE} p. 34",
         url="https://www.nasa.gov/history/50-years-ago-apollo-13-review-board-report/"),
    dict(key="mir1997", kind="incident", title="Mir oxygen-generator fire", year=1997,
         what_happened="On 23 February 1997 a solid-fuel oxygen generator ('candle') in Kvant-1 burned with a "
                       "blowtorch-like flame; the crew put it out with water-based foam extinguishers.",
         lesson="Oxygen sources themselves can be fire sources; suppression in microgravity must reach a flame "
                "that does not behave like one on Earth.",
         source="NASA, '25 Years Ago: Fire Aboard Space Station Mir'",
         url="https://www.nasa.gov/history/25-years-ago-fire-aboard-space-station-mir/"),
    dict(key="air_tanks", kind="spinoff", title="Lighter firefighter air tanks", year=None,
         what_happened="NASA work on lightweight composite pressure vessels led to firefighter breathing-air "
                       "tanks with an aluminum liner wrapped in resin-impregnated glass fiber.",
         lesson="Lighter tanks mean less strain on firefighters carrying their own air supply.",
         source="NASA Spinoff, 'Fireman's Air Tanks'", url="https://spinoff.nasa.gov/node/9582"),
    dict(key="fire_detector", kind="spinoff", title="Smarter fire detector", year=None,
         what_happened="Microgravity research showed fire particulates differ in space. NASA Glenn built a device "
                       "that monitors particulates and chemical species together; in FAA testing it beat aircraft "
                       "smoke detectors at rejecting false alarms and detecting real events.",
         lesson="Detecting fire by what it really emits, not by Earth-based assumptions.",
         source=f"{GUIDE} p. 35", url=None),
    dict(key="fire_gear", kind="spinoff", title="Fire-resistant fabric (PBI)", year=None,
         what_happened="NASA helped develop PBI, a heat- and flame-resistant fiber now used in firefighting, "
                       "military and motor-sport gear.",
         lesson="Materials born from post-Apollo-1 fire safety protect firefighters on Earth.",
         source="NASA Spinoff 2008, 'Polymer Fabric Protects Firefighters, Military, and Civilians'",
         url="https://spinoff.nasa.gov/Spinoff2008/ps_3.html"),
]
