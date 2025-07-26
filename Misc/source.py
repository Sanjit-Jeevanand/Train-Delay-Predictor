import requests
import json
import pandas as pd
from tabulate import tabulate
import os

SEARCHABLE_STOPPOINT_MODES = [
    "tube", "bus", "dlr", "overground", "tram",
    "river-bus", "coach", "cable-car", "elizabeth-line",
    "tflrail", "national-rail", "cycle-hire"
]

# API endpoint
base_url = "https://api.tfl.gov.uk/"

# Fetching Data from TfL API
def fetch_tfl_modes():
    response = requests.get(base_url + "Line/Meta/Modes")
    response.raise_for_status()
    return response.json()

def fetch_tfl_routes():
    response = requests.get(base_url + "Line/Route")
    response.raise_for_status()
    return response.json()

def fetch_tfl_route_details(mode):
    response = requests.get(f"{base_url}Line/Mode/{mode}/Route")
    response.raise_for_status()
    return response.json()

# Storing and Loading Data
def save_data(data, json_filename):
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {json_filename}")

def load_data(json_filename):
    with open(json_filename, "r", encoding="utf-8") as f:
        return json.load(f)

# Visualisation of Data
def visualise(data):
    df = pd.DataFrame(data)
    if "$type" in df.columns:
        df = df.drop(columns=["$type"])
    if "modeName" in df.columns:
        columns = ["modeName"] + [col for col in df.columns if col != "modeName"]
        df = df[columns]
    print(tabulate(df, headers="keys", tablefmt="pretty"))

# User Interaction Functions
def list_modes(modes):
    print("Available Modes:")
    for i, mode in enumerate(modes):
        print(f"{i+1}. {mode['modeName']}")

def filter_routes_by_mode(routes, selected_mode):
    return [route for route in routes if route["modeName"] == selected_mode]

def select_route_name(filtered_routes):
    route_names = [route["name"] for route in filtered_routes]
    print("\nAvailable route names:")
    for i, name in enumerate(route_names):
        print(f"{i + 1}. {name}")

    while True:
        user_input = input("Enter the number of the route you want to select: ").strip()
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(route_names):
                return route_names[idx]
        print("❌ Invalid choice. Try again.")

def get_route_by_name(filtered_routes, selected_name):
    for route in filtered_routes:
        if route["name"] == selected_name:
            return route
    return None

def select_stoppoints_by_mode(mode):
    print("\n🔍 Searching for stops...")
    response = requests.get(f"{base_url}StopPoint/Mode/{mode}")
    response.raise_for_status()
    stops = response.json().get("stopPoints", response.json())
    stops = [s for s in stops if "commonName" in s]

    def display_stops():
        print("\nAvailable Stops (showing top 20):")
        for i, stop in enumerate(stops[:20]):  # Limit to 20 for readability
            print(f"{i + 1}. {stop['commonName']}")

    def get_stop(prompt):
        while True:
            display_stops()
            user_input = input(prompt).strip()
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(stops[:20]):
                    return stops[idx]
            print("❌ Invalid input. Try again.")

    origin = get_stop("Enter origin stop number: ")
    print(f"✅ Selected Origin: {origin['commonName']}")

    destination = get_stop("Enter destination stop number: ")
    print(f"✅ Selected Destination: {destination['commonName']}")

    return origin, destination

def plan_journey(origin_id, destination_id):
    print("\n🧭 Planning your journey...")
    response = requests.get(f"{base_url}Journey/JourneyResults/{origin_id}/to/{destination_id}")
    response.raise_for_status()
    journey_data = response.json()
    return journey_data

def print_journey_summary(journey_data):
    journeys = journey_data.get("journeys", [])
    if not journeys:
        print("❌ No journeys found.")
        return

    for i, journey in enumerate(journeys[:3]):
        print(f"\n🚆 Journey {i+1}: Duration: {journey['duration']} mins")
        for leg in journey["legs"]:
            print(f"  - {leg['mode']['name'].capitalize()} from {leg['departurePoint']['commonName']} to {leg['arrivalPoint']['commonName']}")


# Main Execution Function
def main():
    try:
        modes = load_data("Data/tfl_modes.json")
        routes = load_data("Data/tfl_routes.json")
    except FileNotFoundError:
        print("📥 Fetching data from TfL API...")
        modes = fetch_tfl_modes()
        save_data(modes, "Data/tfl_modes.json")
        routes = fetch_tfl_routes()
        save_data(routes, "Data/tfl_routes.json")

    print("\n🚀 Welcome to the TfL Route Explorer!")

    selected_mode = "tube"  # Default mode

    filtered_routes = filter_routes_by_mode(routes, selected_mode)
    if not filtered_routes:
        print("⚠️ No routes found for that mode.")
        return

    selected_route_name = select_route_name(filtered_routes)
    print(f"\n✅ Selected Route Name: {selected_route_name}")

    selected_route = get_route_by_name(filtered_routes, selected_route_name)
    # print("\n📦 Full route data:")
    # print(json.dumps(selected_route, indent=2))

    filename = f"Data/selected_route_{selected_route_name.replace(' ', '_')}.json"
    save_data(selected_route, filename)

    origin, destination = select_stoppoints_by_mode(selected_mode)
    # print("\n📍 Origin Stop:")
    # print(json.dumps(origin, indent=2))
    # print("\n📍 Destination Stop:")
    # print(json.dumps(destination, indent=2))

    journey_data = plan_journey(origin["naptanId"], destination["naptanId"])
    print_journey_summary(journey_data)
    save_data(journey_data, "Data/journey_result.json")


if __name__ == "__main__":
    main()