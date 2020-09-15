#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=unbalanced-tuple-unpacking
"""
Gym environment for simulating the development of emergencies requiring ambulance in a city.

This environment will be used in order to train a resurce allocation agent. The agent receives
information about emergencies and their severity. The agent can send an ambulance to attend the
emergency from one of the different hospitals in the city, and make the ambulance go back to the
original hospital or to one of the other hospitals in the city.

If not responded to in time, the emergencies can result in failure situations, sampled from a
probability distribution according to their severity.

Emergencies are gerated from representative probability distributions.

Created by Enrique Basañez, Miguel Blanco, Alfonso Lagares, Borja Menéndez and Francisco Rueda.

TODO:
 - Put traffic input data in __init__ parameters

=======
"""

import calendar
import os
from collections import defaultdict, deque, namedtuple
from datetime import datetime, timedelta
from pathlib import Path
import shapefile
from shapely.geometry import (
    shape,
    Polygon,
    Point,
    MultiPolygon,
    LinearRing,
    MultiLineString,
    MultiPoint,
    LineString,
)

import gym
import numpy as np
import pandas as pd
import yaml
from gym import spaces
from gym.utils import seeding
from recordclass import recordclass

from .traffic_manager import TrafficManager


class CitySim(gym.Env):
    """Gym environment for simulating ambulance emergencies in a city.

    Attributes:
        city_config: str or Path, YAML file with parameters describing the city to simulate. If no 
            file is provided, default values will be used.
        city_geometry: str or Path, shapefile describing the limits of the city districts.
        time_step: int, seconds advanced at each step. Should be high to avoid sparse actions but low
            to enable accuracy. Compromise. One minute by default.
        stress: float, multiplier for the emergency generator, in order to artificially increase or
            decrease the amount of emergencies and modify the stress to the system.
        log_file: str or Path, text file where simulation events will be logged in chronological 
            order.
        mov_reward: int, reward that will be assigned to each ambulance that does not attend an 
            emergency, and only moves between hospitals.
    """

    metadata = {
        "render.modes": ["rgb_array", "console"],
        "video.frames_per_second": 30,
    }

    def __init__(
        self,
        city_config="data/city_defaults.yaml",  # YAML file w/ city and generator data
        city_geometry="data/madrid_districs_processed/madrid_districs_processed.shp",
        traffic_default_cols="data/default_columns.csv",
        traffic_models="data/traffic_models",
        time_start: datetime = datetime.fromisoformat("2020-01-01T00:00:00"),
        time_end: datetime = datetime.fromisoformat("2024-12-31T23:59:59"),
        time_step: int = 60,
        stress: float = 1.0,
        log_file=None,
        mov_reward: int = 0,
        actions_per_round: int = 1,
    ):
        """Initialize the CitySim environment."""
        assert os.path.isfile(city_config), "Invalid path for city configuration file"
        assert os.path.isfile(city_geometry), "Invalid path for city geometry file"
        assert os.path.isfile(traffic_default_cols), "Invalid path for traffic default file"
        assert os.path.isdir(traffic_models), "Invalid path for traffic models directory"

        self.time_start = time_start
        self.time_end = time_end
        self.time_step_seconds = time_step
        self.time_step = timedelta(seconds=self.time_step_seconds)
        self.stress = stress
        self.mov_reward = mov_reward

        # Named lists for status keeping
        self.hospital = recordclass("Hospital", ["name", "loc", "available_amb"])
        self.emergency = recordclass("Emergency", ["loc", "severity", "tappearance", "code"])
        self.moving_amb = recordclass(
            "MovingAmbulance",
            ["tobjective", "thospital", "origin", "destination", "severity", "code"],
        )

        # Read configuration file for setting up the city
        if type(city_config) is dict:
            config = city_config
        else:
            with open(city_config) as config_file:
                config = yaml.safe_load(config_file)
        with shapefile.Reader(str(city_geometry)) as sf:
            geometry = sf.shapes()
        self._configure(config, geometry)

        # Traffic model data
        default_df = pd.read_csv(traffic_default_cols, sep=";")
        self.traffic_manager = TrafficManager(
            time_start, self.districts, traffic_models, default_df
        )

        # Set up log file for registering simulation events
        self.log_events = log_file is not None
        if log_file is not None:
            self.log_file = Path(log_file)
            with self.log_file.open("w") as log:
                log.write("City Simulation Log." + "\n")
                log.write(
                    "#EM [timeISO] [severity] [coordXkm] [coordYkm] [district_code] [em_identifier]"
                    + "\n"
                )
                log.write(
                    "#AM [timeISO] [severity] [hosp_origin] [hosp_destination] [tobjective] [thospital] [reward] [em_identifier]"
                    + "\n"
                )

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        """Return the environment to the start of a new scenario, with no active emergencies. 
        """
        # Reset status variables
        self.time = self.time_start
        self.active_emergencies = ["dummy"] + [deque() for i in range(self.severity_levels)]
        self.outgoing_ambulances = []
        self.incoming_ambulances = []

        # Reset number of ambulances in hospitals to initial
        for i in self.hospitals.keys():
            self.hospitals[i]["available_amb"] = self.initial_ambulances[i]

        # Reset cumulative variables
        self.total_emergencies = {level: 0 for level in range(1, self.severity_levels + 1)}
        self.total_ambulances = {level: 0 for level in range(0, self.severity_levels + 1)}

        # Log the reset into the log file
        if self.log_events:
            with self.log_file.open("a") as log:
                log.write(f"Reset {self.time_start.isoformat()} {self.time_end.isoformat()}" + "\n")

        return self._get_obs()

    def step(self, action):

        # Check for objectives in outgoing ambulances and apply failures to reward
        new_outgoing = []
        reward = 0
        for ambulance in self.outgoing_ambulances:
            if self.time >= ambulance["tobjective"]:  # Ambulance arrived at emergency
                self.incoming_ambulances.append(ambulance)
            else:
                reward += -ambulance["severity"] * self.time_step_seconds
                new_outgoing.append(ambulance)
        self.outgoing_ambulances = new_outgoing

        # Check for final destinations in incoming ambulances and add them to the roster
        new_incoming = []
        for ambulance in self.incoming_ambulances:
            if self.time >= ambulance["thospital"]:
                self.hospitals[ambulance["destination"]]["available_amb"] += 1
            else:
                if ambulance["severity"] > 3:  # High severity em. still active until hospital
                    # The 1.0 is because once you are in the ambulance, the cost should be lower
                    reward += -ambulance["severity"] * self.time_step_seconds * 0.5
                new_outgoing.append(ambulance)
        self.incoming_ambulances = new_incoming

        # For every active emergencie still in queue, add the corresponing waiting cost
        for severity, severity_queue in enumerate(self.active_emergencies):
            if severity == 0:  # Skip the dummy level
                continue
            # Add cost proportional to number of active emergencies and severity
            reward += -severity * self.time_step_seconds * len(severity_queue)

        # Take actions.
        for every_action in action:
            severity, start_hospital_id, end_hospital_id = every_action
            start_hospital = self.hospitals[start_hospital_id]
            end_hospital = self.hospitals[end_hospital_id]
            if severity == 0:  # Move ambulances between hospitals, no emergency
                if end_hospital_id == start_hospital_id:
                    continue  # This movement would not make sense
                if self.hospitals[start_hospital_id]["available_amb"] == 0:
                    continue  # An empty hospital cannot launch ambulances
                if (end_hospital_id == 0) or (start_hospital_id == 0):
                    continue  # Null hospital does not launch or receive any ambulances
                self.hospitals[start_hospital_id]["available_amb"] -= 1
                tthospital = self._displacement_time(start_hospital["loc"], end_hospital["loc"])
                code = self.total_ambulances[0] + 1
                ambulance = self.moving_amb(
                    self.time, self.time + tthospital, start_hospital_id, end_hospital_id, 0, code,
                )
                self._log_ambulance(ambulance)
                self.total_ambulances[0] = code
                self.incoming_ambulances.append(ambulance)
                reward += self.mov_reward  # Possible cost associated with the movement
                continue

            if (
                start_hospital_id == 0
            ):  # Starting hospital #0 simbolizes null action for severity level
                continue
            if len(self.active_emergencies[severity]) == 0:
                # If the queue for this severity level is empty, no action
                continue
            if start_hospital["available_amb"] == 0:  # No ambulances in initial hospital, no action
                continue

            if end_hospital_id == 0:  # Null end hospital to return to start hospital
                end_hospital_id = start_hospital_id
                end_hospital = start_hospital

            # Launch an ambulance from start hospital towards emergency
            self.hospitals[start_hospital_id]["available_amb"] -= 1
            emergency = self.active_emergencies[severity].popleft()
            ttobj = self._displacement_time(start_hospital["loc"], emergency["loc"])
            tthospital = self._displacement_time(emergency["loc"], end_hospital["loc"]) + ttobj
            code = emergency["code"]
            ambulance = self.moving_amb(
                self.time + ttobj,
                self.time + tthospital,
                start_hospital_id,
                end_hospital_id,
                severity,
                code,
            )
            self._log_ambulance(ambulance)
            self.total_ambulances[severity] += 1
            self.outgoing_ambulances.append(ambulance)

        # Advance time
        self.time += self.time_step
        self.traffic_manager.update_traffic(self.time)

        # Generate new emergencies. Emergencies are a series of FIFO lists, one per severity
        self._generate_emergencies()

        # Return state, reward, and whether the end time has been reached
        return self._get_obs(), reward, self.time >= self.time_end, {}

    def render(self, mode="console"):
        print(self._get_obs())

    def close(self):
        pass

    def set_stress(self, stress):
        """Modify the stress factor at any moment in the execution."""
        self.stress = stress

    def _configure(self, config, geometry):
        """Set the city information variables to the configuration."""

        self.config = config.copy()

        self.hospitals = config["hospitals"].copy()
        self.districts = config["districts"].copy()
        self.severity_levels = config["severity_levels"]
        self.severity_dists = config["severity_dists"]
        self.shown_emergencies_per_severity = config["shown_emergencies_per_severity"]
        self.n_hospitals = len(self.hospitals) - 1

        # Generate a {district_code: Polygon} dict from the shapefile data
        self.geo_dict = {i + 1: shape(geometry[i]) for i in range(len(geometry))}

        # Correct possible discrepancies in hospital district data and geometry data
        for hospital_id, hospital in self.hospitals.items():
            point = Point(hospital["loc"]["x"], hospital["loc"]["y"])
            hospital_district_code = 0
            for district_code, polygon in self.geo_dict.items():
                if polygon.contains(point):
                    hospital_district_code = district_code
            self.hospitals[hospital_id]["loc"]["district_code"] = hospital_district_code

        # Store original state of available ambulances on its own
        self.initial_ambulances = [
            self.config["hospitals"][i]["available_amb"]
            for i in range(len(self.config["hospitals"]))
        ]

        # Define the action space
        self.action_space = spaces.Tuple(
            (
                spaces.Tuple([spaces.Discrete(self.n_hospitals + 1)] * (self.severity_levels + 1)),
                spaces.Tuple([spaces.Discrete(self.n_hospitals + 1)] * (self.severity_levels + 1)),
            )
        )

    def _get_obs(self, mode="tables"):
        """Build the part of the state that the agent can know about.

        This includes hospital locations, ambulance locations, incoming emergencies.
        """

        observation = []

        # Hospitals table
        # id x y available_amb incoming_amb ttamb
        hospitals_table = []
        for id, hospital in self.hospitals.items():
            x, y = hospital["loc"]["x"], hospital["loc"]["y"]
            district_code = hospital["loc"]["district_code"]
            incoming = 0
            for ambulance in self.outgoing_ambulances + self.incoming_ambulances:
                if ambulance["destination"] == id:
                    incoming += 1
            hospital_data = [id, x, y, district_code, hospital["available_amb"], incoming]
            hospitals_table.append(hospital_data)
        observation.append(np.array(hospitals_table))

        # Unattended emergencies, with locations and severity. 3D table in severity/order/data
        # Data for each emergency is severity order time_active x y
        emergencies_table = []
        for severity, queue in enumerate(self.active_emergencies):
            severity_table = []
            if severity == 0:
                continue
            for order in range(self.shown_emergencies_per_severity):
                if order < len(queue):
                    emergency = queue[order]
                    loc = emergency["loc"]
                    x, y, district_code = loc["x"], loc["y"], loc["district_code"]
                    tactive = int((self.time - emergency["tappearance"]) / self.time_step)
                    emergency_data = [severity, tactive, x, y, district_code]
                else:
                    emergency_data = [0, 0, 0, 0, 0]
                severity_table.append(emergency_data)
            emergencies_table.append(severity_table)
        observation.append(np.array(emergencies_table))

        # Districts data?

        # Time data
        time_data = np.array(
            [
                self.time_step_seconds,  # Information about potential reaction time
                self.time.month,
                self.time.day,
                self.time.weekday() + 1,
                self.time.hour,
                self.time.minute,
            ]
        )
        observation.append(time_data)

        # Traffic data always sorted to give the same order
        traffic_data = []
        all_districts = list(self.traffic_manager.traffic.keys())
        try:
            all_districts.remove("Missing")
        except:
            pass
        sorted_districts = sorted(all_districts)
        for district in sorted_districts:
            traffic_data.append([district, self.traffic_manager.traffic[district]])
        observation.append(np.array(traffic_data))

        if mode == "tables":
            return observation

        if mode == "flat":
            # Return a flattened vector with all values in the observation but no structure
            flat = np.concatenate([piece.flatten() for piece in observation])
            return flat

        return observation

    def _generate_emergencies(self):
        """For given city parameters and time, generate appropriate emergencies for a timestep.

        Emergencies come predefined with the time to failure, which is softly correlated to severity.

        The agent only knows about the location, severity and the time since it was generated.
        """

        hour = self.time.hour
        weekday = self.time.weekday() + 1
        month = self.time.month

        for severity in range(1, self.severity_levels + 1):
            base_frequency = self.severity_dists[severity]["frequency"]
            current_frequency = (
                base_frequency
                * self.severity_dists[severity]["hourly_dist"][hour]
                * self.severity_dists[severity]["daily_dist"][weekday]
                * self.severity_dists[severity]["monthly_dist"][month]
                * self.stress
            )

            # Assuming independent distributions per hour, weekday and month
            period_frequency = current_frequency * self.time_step_seconds  # Avg events per step

            # Poisson distribution of avg # of emergencies in period will give number of new ones
            num_new_emergencies = int(np.random.poisson(period_frequency, 1))

            # If no emergencies are to be generated, skip to the next severity level
            if num_new_emergencies == 0:
                continue

            # Get the district weights for the current severity
            probs_dict = self.severity_dists[severity]["district_prob"]
            district_weights = np.array([w for district, w in sorted(probs_dict.items())])
            district_weights = district_weights / district_weights.sum()

            for _ in range(num_new_emergencies):  # Skipped if 0 new emergencies
                district = np.random.choice(  # District where emergency will be located
                    np.arange(len(district_weights)) + 1, p=district_weights
                )
                loc = self._random_loc_in_distric(district)
                tappearance = self.time
                code = self.total_emergencies[severity] + 1
                emergency = self.emergency(loc, severity, tappearance, code,)
                self._log_emergency(emergency)
                self.total_emergencies[severity] = code  # Accumulate in history
                self.active_emergencies[severity].append(emergency)  # Add to queue

    def _displacement_time(self, start, end):
        """Given start and end points, returns a displacement time between both locations for an 
        ambulance, based on the current traffic, metheorology, and randomness.

        (x1, y1, district1) (x2, y2, district2)  [km], centro P. del Sol, x -> Este, y -> Norte
        """

        distance_per_district = self._get_segments_per_district(
            start["district_code"],
            (start["x"], start["y"]),
            end["district_code"],
            (end["x"], end["y"]),
        )
        total_time = self.traffic_manager.displacement_time(distance_per_district)

        return timedelta(seconds=total_time)

    def _get_random_point_in_polygon(self, polygon):
        min_x, min_y, max_x, max_y = polygon.bounds
        while True:
            point = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
            if polygon.contains(point):
                return point

    def _random_loc_in_distric(self, district_code):
        polygon = self.geo_dict[district_code]
        point = self._get_random_point_in_polygon(polygon)
        x, y = np.array(point.coords).flatten().tolist()
        return {"x": x, "y": y, "district_code": district_code}

    def _obtain_route_cuts(self, origin, destination):
        route = LineString([Point(origin[0], origin[1]), Point(destination[0], destination[1])])

        cuts = {}  # Dict with {district_code: list of (x, y) tuples},
        for district_code, polygon in self.geo_dict.items():
            # To allow handling more than 2 intersection points between route line and polygon
            distric_lr = LinearRing(list(polygon.exterior.coords))
            intersection = distric_lr.intersection(route)
            # Intersects district in ONE point
            if type(intersection) == Point:
                cuts[district_code] = [(intersection.x, intersection.y)]
            # Intersects district in TWO OR MORE points
            if type(intersection) == MultiPoint:
                cuts[district_code] = [(point.x, point.y) for point in intersection]
        return cuts

    # Calculate distances traversed across districts
    def _get_segments_per_district(
        self, district_origin, origin, district_destination, destination
    ):
        cuts = self._obtain_route_cuts(origin, destination)
        # If may return empty for same origin and destination district, but it will need an entry
        if len(cuts) == 0:
            cuts[district_origin] = []

        # Add points at start and end corresponding to the origin and destination
        cuts[district_origin].insert(0, origin)
        cuts[district_destination].append(destination)

        distances = {k: self._cartesian(*v) for (k, v) in cuts.items()}
        distances["Missing"] = self._cartesian(origin, destination) - sum(distances.values())

        return distances

    def _cartesian(self, *kwargs):
        """Given a series of points in th Cartesian plane, returns the sums of the distances 
        between consecutive pairs of such points. There must be an even number of points.
        """
        if len(kwargs) % 2 != 0:
            return 0
        result = 0
        for i in range(0, len(kwargs), 2):
            p1 = kwargs[i]
            p2 = kwargs[i + 1]
            result += np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        return result

    def _reward_f(self, time_diff, severity):
        """Possible non-linear fuction to apply to the time difference between an ambulance arrival
        and the time reference of the emergency in order to calculate a reward for the agent.
        """
        return time_diff * severity  # Right now linear with time to emergency and severity

    def _log_emergency(self, emergency):
        if self.log_events:
            time = self.time.isoformat()
            severity = emergency["severity"]
            loc = emergency["loc"]
            x, y, district_code = loc["x"], loc["y"], loc["district_code"]
            code = emergency["code"]
            info = f"EM {time} {severity} {x:.8f} {y:.8f} {district_code} {code}"
            with self.log_file.open("a") as log:
                log.write(info + "\n")

    def _log_ambulance(self, ambulance):
        if self.log_events:
            time = self.time.isoformat()
            severity = ambulance["severity"]
            origin = ambulance["origin"]
            destination = ambulance["destination"]
            tobjective = ambulance["tobjective"].isoformat()
            thospital = ambulance["thospital"].isoformat()
            code = ambulance["code"]
            info = f"AM {time} {severity} {origin} {destination} {tobjective} {thospital} {code}"
            with self.log_file.open("a") as log:
                log.write(info + "\n")
