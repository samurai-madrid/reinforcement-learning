
from datetime import datetime, timedelta
import random
import os
import pickle

import pandas as pd

class TrafficManager():

    def __init__(self, start_time, districts, dir_traffic_models, default_df,
        updates_per_hour: int = 4,
        max_avg_speed: float = 60.0,
        max_load: float = 100.0,
        perc = 0.1,
        ):

        self.update_period = 60 / updates_per_hour
        self.update_points = {i: int(self.update_period * i) for i in range(updates_per_hour)}

        self.last_update = self._normalize_time(start_time)
        self.districts = districts
        self.models = self._load_traffic_models(dir_traffic_models)
        self.default_df = default_df

        self.max_avg_speed = max_avg_speed
        self.max_load = max_load
        self.perc = perc

        self.traffic = {district : 0 for district in districts.keys()}

    def _normalize_time(self, time):
        return time.replace(minute=self.update_points[int(time.minute / self.update_period)])

    def _load_traffic_models(self, dir_traffic_models):
        models = {}

        for model_file in os.listdir(dir_traffic_models):
            district = int(model_file.split('_')[-1].split('.')[0])
            models[district] = pickle.load(open(os.path.join(dir_traffic_models, model_file), 'rb'))

        return models

    def _prepare_data(self, time):
        data = self.default_df[0:0]

        new_row = {'year': time.year,
                'day_{}'.format(time.day): 1,
                'month_{}'.format(time.month): 1,
                'hour-minute_{}'.format(time.strftime('%H:%M')): 1,
                'weekday_{}'.format(time.weekday()): 1}

        data = data.append(new_row, ignore_index=True).fillna(0).astype(int).iloc[0]

        return data

    def _get_speed(self, traffic_load):
        return self.max_avg_speed * (1 - traffic_load / self.max_load)

    def update_traffic(self, time):
        norm_time = self._normalize_time(time)
        if norm_time > self.last_update:
            data = self._prepare_data(norm_time)
            self.traffic = {district: self.models[district].predict([data])[0] * (1 + random.uniform(-self.perc, self.perc))
                            for district in self.traffic.keys()
                            if district != 'Missing'}
            self.last_update = norm_time

    def displacement_time(self, distance_per_district):
        # If something is outside the limits, it gets assigned average traffic of present districts
        other_districts_traffic = [self.traffic[district] 
                                   for district in distance_per_district.keys() 
                                   if district != 'Missing']
        self.traffic['Missing'] = sum(other_districts_traffic) / len(other_districts_traffic)

        total_time = sum([distance / self._get_speed(self.traffic[district]) 
                          for district, distance in distance_per_district.items()]) * 3600

        #print('Total time: {}'.format(total_time))

        return total_time