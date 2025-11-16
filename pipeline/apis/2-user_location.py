#!/usr/bin/env python3
"""Script that prints the location of a GitHub user"""
import sys
import requests
import time


def print_user_location(url):
    """Method that prints the location of a GitHub user"""
    response = requests.get(url)

    # Rate limit exceeded
    if response.status_code == 403:
        reset_time = response.headers.get("X-RateLimit-Reset")
        if reset_time:
            reset_timestamp = int(reset_time)
            now = int(time.time())
            minutes = (reset_timestamp - now) // 60
            print(f"Reset in {minutes} min")
        else:
            print("Forbidden")
        return

    # User not found
    if response.status_code == 404:
        print("Not found")
        return

    # Normal successful case
    if response.status_code == 200:
        data = response.json()
        print(data.get("location"))
        return

    # Any other unexpected status
    print("Error:", response.status_code)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Not the right number of arguments")
        sys.exit(1)

    url = sys.argv[1]
    print_user_location(url)
