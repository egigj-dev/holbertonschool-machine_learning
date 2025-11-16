#!/usr/bin/env python3
"""Script that returns the list of ships"""
import requests


def availableShips(passengerCount):
    """Method that returns the list of ships"""
    url = 'https://swapi-api.hbtn.io/api/starships/'
    ships = []
    while url:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for ship in data['results']:
                passengers = ship['passengers']
                if passengers not in ['unknown', 'n/a']:
                    passengersNo = int(passengers.replace(',', ''))
                    if passengersNo >= passengerCount:
                        ships.append(ship['name'])
        url = data['next']
    return ships
