#!/usr/bin/env python3
"""Script that returns the first launch date of a SpaceX rocket"""
import requests


def get_first_launch():
    """Returns the first launch date of a SpaceX rocket"""
    launches_url = 'https://api.spacexdata.com/v4/launches'
    rockets_url = 'https://api.spacexdata.com/v4/rockets'
    launchpads_url = 'https://api.spacexdata.com/v4/launchpads'
    
    # Get launches
    launches = requests.get(launches_url).json()

    # Sort launches
    launches.sort(key=lambda x: x.get("date_unix", float('inf')))

    # Get first launch info
    first_launch = launches[0]
    rocket_id = first_launch.get("rocket")
    launchpad_id = first_launch.get("launchpad")

    # Fetch rocket and launchpad details
    rocket = requests.get(f"{rockets_url}/{rocket_id}").json()
    launchpad = requests.get(f"{launchpads_url}/{launchpad_id}").json()

    # Get first launch details
    launch_name = first_launch["name"]
    date_local = first_launch["date_local"]
    rocket_name = rocket.get("name")
    launchpad_name = launchpad.get("name")
    launchpad_locality = launchpad.get("locality")

    # Print result
    print(f"{launch_name} ({date_local}) {rocket_name} - {launchpad_name} ({launchpad_locality})")


if __name__ == '__main__':
    get_first_launch()
    
