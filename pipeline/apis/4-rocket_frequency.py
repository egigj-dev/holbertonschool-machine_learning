#!/usr/bin/env python3
"""
Fetches all launches and prints the number of launches per rocket
"""

import requests

if __name__ == "__main__":
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets"

    # Fetch all launches
    launches = requests.get(launches_url).json()

    # Fetch all rockets
    rockets = requests.get(rockets_url).json()

    # Map rocket id → rocket name
    rocket_names = {r["id"]: r["name"] for r in rockets}

    # Count launches per rocket id
    frequency = {}
    for launch in launches:
        rocket_id = launch["rocket"]
        frequency[rocket_id] = frequency.get(rocket_id, 0) + 1

    # Convert rocket IDs to names and print sorted alphabetically
    sorted_ids = sorted(
        frequency,
        key=lambda rid: (-frequency[rid], rocket_names[rid])
    )
    for rid in sorted_ids:
        print(f"{rocket_names[rid]}: {frequency[rid]}")
