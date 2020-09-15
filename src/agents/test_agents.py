import numpy as np

DEFAULT_ACTION = (0, 0, 0)

def cartesian(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def distances_to_hospitals(em_x, em_y, hospitals):
    distances = []
    for hosp in hospitals:
        distances.append((hosp[0], cartesian(em_x, em_y, hosp[1], hosp[2])))
        
    distances = sorted(distances, key=lambda h: h[1])
    
    return distances

def first_free_hospital(distances, hospitals):
    for el in distances:
        if hospitals[int(el[0])][4] > 0:
            return int(el[0])
        
    return None

class RandomAgent():
    def __init__(self, n_hospitals, n_severity_levels, n_actions):
        self.n_hospitals = n_hospitals
        self.n_severity_levels = n_severity_levels
        self.n_actions = n_actions

    def __call__(self, observation):
        severities = np.random.randint(self.n_severity_levels+1, size=self.n_actions)
        start_hospitals = np.random.randint(self.n_hospitals+1, size=self.n_actions)
        end_hospitals = np.random.randint(self.n_hospitals+1, size=self.n_actions)
        
        to_return = [(severities[i], start_hospitals[i], end_hospitals[i]) for i in range(self.n_actions)]

        return to_return
    
class NaiveGreedyAgent():
    def __init__(self, n_hospitals, n_severity_levels, n_actions):
        self.n_hospitals = n_hospitals
        self.n_severity_levels = n_severity_levels
        self.n_actions = n_actions
        
    def __call__(self, observation):
        to_return = []
        num_actions_taken = 0
        
        hospitals = observation[0]
        emergencies = observation[1]
        
        for severity in range(len(emergencies) - 1, -1, -1):
            for em in emergencies[severity]:
                if em[-1] != 0:
                    distances = distances_to_hospitals(em[2], em[3], hospitals)
                    ff_hospital = first_free_hospital(distances, hospitals)
                    hospitals[ff_hospital][4] -= 1
                    to_return.append((int(em[0]), ff_hospital, int(distances[0][0])))
                    num_actions_taken += 1
        
        to_return += [DEFAULT_ACTION for i in range(self.n_actions - num_actions_taken)]
        
        return to_return