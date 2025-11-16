#!/usr/bin/env python3
"""Script that returns the home planets of all sentient species"""
import requests


def sentientPlanets():
    """Method that returns home planets of all sentient species"""
    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []
    while url:
        data = requests.get(url).json()
        for species in data["results"]:
            classification = (species.get("classification") or "").lower()
            designation = (species.get("designation") or "").lower()

            # Check if species is sentient
            if "sentient" in classification or "sentient" in designation:
                home = species.get("homeworld")
                if home:  # skip species without a home planet
                    planet_info = requests.get(home).json()
                    planet_name = planet_info.get("name")
                    if planet_name:
                        planets.append(planet_name)
        url = data["next"]   # pagination
    return planets
