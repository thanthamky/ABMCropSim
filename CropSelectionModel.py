import pickle
import random
import numpy as np
from datetime import datetime
from meteostat import Monthly, Point, Daily

class CropSelector:
    
    def __init__(self, dt_model_path, knn_model_path, encoder_path):
        
        self.feature_names = ['temp_min', 'temp_max', 'temp_mean', 'precip']
        self.class_names = ['cassava', 'maize', 'oilpalm', 'pararubber', 'rice', 'sugarcane', 'none']
        self.dt_model = pickle.load(open(dt_model_path, 'rb'))
        self.knn_model = pickle.load(open(knn_model_path, 'rb'))
        self.encoder = pickle.load(open(encoder_path, 'rb'))
        self.crop_constraint = {'rice': 0,
                       'maize': 0,
                       'sugarcane': 0,
                       'cassava': 0,
                       'oilpalm': 15,
                       'pararubber': 20,
                       'none': 0}
    
    def _is_switchable(self, crop_name, count):
    
        return count >= self.crop_constraint[crop_name] if crop_name in self.crop_constraint.keys() else None
    
    def _get_weather(self, year, lat, lon):

        # Set time period
        start = datetime(year-2, 1, 1)
        end = datetime(year-1, 12, 31)

        # Create Point for Vancouver, BC
        #location = Point(12.667,102.17)
        #location = Point(19.384244, 98.930763)
        location = Point(lat, lon)

        # Get daily data for 2018
        data = Daily(location, start, end)
        data = data.fetch()
        data = data.interpolate(limit_direction='both')

        # example of return 22.09, 33.4, 26.84, 2816.55
        #print(data)
        return np.mean(data['tmin']), np.mean(data['tmax']), np.mean(data['tavg']), np.sum(data['prcp'])

    def find_crop(self, year, lat, lon, current_crop, crop_count, agent_id, is_knn=False, is_random=False):
        
        if current_crop not in self.class_names: raise Exception(f'{current_crop} is not in list {self.class_names}')
        
        if is_random:
            
            return self.class_names[random.randint(0, len(self.class_names)-1)], 0
    
        if self._is_switchable(current_crop, crop_count):

            tmin, tmax, tavg, prcp = self._get_weather(year, lat, lon)
            
            #print(tmin, tmax, tavg, prcp)
            
            try:

                if not is_knn:

                    next_crop = self.encoder.inverse_transform(self.dt_model.predict([[tmin, tmax, tavg, prcp]]))[0]

                    if next_crop == current_crop:

                        crop_count += 1

                    else:

                        crop_count = 0

                    return next_crop, crop_count

                else:

                    next_crop = self.encoder.inverse_transform(self.knn_model.predict([[tmin, tmax, tavg, prcp]]))[0]

                    if next_crop == current_crop:

                        crop_count += 1
                    else:

                        crop_count = 0


                    return next_crop, crop_count
            
            except:
                
                
                
                if current_crop == 'none':
                    
                    print('agent id: ', agent_id, ' : getting weather data error! random crop...' )
                    return self.class_names[random.randint(0, len(self.class_names)-1)], 0
                
                else:
                    
                    print('agent id: ', agent_id, ' : getting weather data error! select previous crop...' )
                    return current_crop, crop_count+1   

        else:

            crop_count += 1
            
            return current_crop, crop_count
        
    def select_crop(self, data):
    
        output = []

        for i in range(len(data)):

            agent_id = data[i][0]
            crop, count = self.find_crop(data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], agent_id)
            output.append([agent_id, crop, count])

        return output
        
        