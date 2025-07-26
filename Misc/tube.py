import requests
import json
import os

BASE_URL = "https://api.tfl.gov.uk/"
MODE = "tube"
OUTPUT_FILE = "Data/tube_stations.json"

def fetch_all_tube_stations():
    print("📡 Fetching all Tube stations in London...")
    response = requests.get(f"{BASE_URL}StopPoint/Mode/{MODE}")
    response.raise_for_status()
    data = response.json()

    # Some results may be nested under "stopPoints"
    if "stopPoints" in data:
        stations = data["stopPoints"]
    else:
        stations = data

    stations = [s for s in stations if "commonName" in s and "naptanId" in s]

    # Group by stationNaptan to remove platform-level duplicates
    seen = {}
    for s in stations:
        station_id = s.get("stationNaptan", s["naptanId"])  # fallback to naptanId
        if station_id not in seen:
            seen[station_id] = s
    stations = list(seen.values())

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(stations, f, indent=2)
    print(f"✅ {len(stations)} Tube stations saved to {OUTPUT_FILE}")

    return stations

def fetch_stations_for_tube_line(line_id):
    print(f"\n📡 Fetching stations for the '{line_id}' line...")
    response = requests.get(f"{BASE_URL}Line/{line_id}/StopPoints")
    response.raise_for_status()
    data = response.json()
    stations = [s for s in data if "commonName" in s and "naptanId" in s]
    print(f"✅ Found {len(stations)} stations on the {line_id} line.")
    return stations

def plan_journey(origin_id, destination_id):
    url = f"{BASE_URL}Journey/JourneyResults/{origin_id}/to/{destination_id}"
    print(f"\n🧭 Planning your journey...\n🔗 {url}")
    response = requests.get(url)
    response.raise_for_status()
    journey_data = response.json()
    return journey_data

def print_journey_summary(journey_data):
    journeys = journey_data.get("journeys", [])
    if not journeys:
        print("❌ No journeys found.")
        return

    seen_pairs = set()
    for i, journey in enumerate(journeys):
        if not journey["legs"]:
            continue
        first_leg = journey["legs"][0]
        last_leg = journey["legs"][-1]
        journey_key = (first_leg["departurePoint"]["commonName"], last_leg["arrivalPoint"]["commonName"])
        if journey_key in seen_pairs:
            continue
        seen_pairs.add(journey_key)

        print(f"\n🚆 Journey {len(seen_pairs)}: Duration: {journey['duration']} mins")
        for leg in journey["legs"]:
            print(f"  - {leg['mode']['name'].capitalize()} from {leg['departurePoint']['commonName']} to {leg['arrivalPoint']['commonName']}")
        if len(seen_pairs) == 3:
            break


def list_stations_by_line_name(stations, line_name):
    matching_stations = [s for s in stations if "lines" in s and any(line['name'].lower() == line_name.lower() for line in s["lines"])]
    if not matching_stations:
        print(f"❌ No stations found for line: {line_name}")
        return
    print(f"\n📍 Stations on the '{line_name}' line:")
    for station in matching_stations:
        print(f"- {station['commonName']} (ID: {station['naptanId']})")

if __name__ == "__main__":
    stations = fetch_all_tube_stations()
    for s in stations:
        print(f"- {s['commonName']} (ID: {s['naptanId']})")

    origin_name = input("\nEnter Origin Station Name: ").strip()
    destination_name = input("Enter Destination Station Name: ").strip()

    def find_station_by_name(name, stations):
        name = name.lower().strip()
        matches = [s for s in stations if name in s["commonName"].lower()]
        if len(matches) == 1:
            return matches[0]
        elif matches:
            print(f"🔎 Multiple matches found for '{name}':")
            for i, m in enumerate(matches):
                line_names = ", ".join(line["name"] for line in m.get("lines", []))
                print(f"{i+1}. {m['commonName']} (Lines: {line_names}) (ID: {m['naptanId']})")
            choice = input("Select a number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(matches):
                return matches[int(choice) - 1]
        else:
            print(f"❌ No matches found for '{name}'.")
        return None

    origin = find_station_by_name(origin_name, stations)
    destination = find_station_by_name(destination_name, stations)

    if not origin or not destination:
        print("❌ Could not find one or both stations by that name.")
    elif origin['naptanId'] == destination['naptanId']:
        print("❌ Origin and destination cannot be the same.")
    else:
        try:
            origin_id = origin.get("stationNaptan", origin["naptanId"])
            destination_id = destination.get("stationNaptan", destination["naptanId"])
            journey_data = plan_journey(origin_id, destination_id)
            print_journey_summary(journey_data)
        except requests.HTTPError as e:
            print(f"❌ Failed to plan journey from {origin['commonName']} to {destination['commonName']}: {e}")