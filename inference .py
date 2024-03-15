import pickle
import os
import pandas as pd 
import numpy as np 
import datetime as datetime
from scipy.stats import yeojohnson

class Inference:
    
    def __init__(self,model_path,pt_path):
       
        self.__model_path=model_path
        self.__pt_path=pt_path
        
        if os.path.exists(self.model_path) and os.path.exists(self.pt_path):
            self.model=pickle.load(open(self.model_path,'rb'))
            self.__pt=pickle.load(open(self.pt_path,'rb'))
        else:
            print('given path is incorrect')
    
    def User_input(self):
        print('Enter correct information to predict the rented bike count ')
        
        self.__date=input('Date format (dd/mm/yyyy) :')
        
        self.__hour=int(input('Hours (0-23) :'))
        
        self.__Temprature=float(input('Temprature in (°C) :'))
        
        self.__Humidity=float(input('Humidity (%) :'))
        
        self.__wind_speed=float(input('Wind speed (m/s) :'))
        
        self.__Visibility =float(input('Visibility (10m)  :'))
        
        self.__Dew_point_temperature=float(input('Dew point temperature(°C):'))
        
        self.__Solar_Radiation=float(input('Solar Radiation (MJ/m2) :'))
        
        self.__holiday=input(' holiday (holiday,no holiday )in lower case:')
        
        self.__functinoning_day =input(' functinoning_day (yes,no) in lower case:')
        
        self.__seasons=input(' Seasons (autumn, spring, summer ,winter) in lower case:')

        self.__weekday=input(' weekday (monday,tuesday,wednesday ,thursday ,friday,saturday)in lower case:')
        
        # transforming the wind speed and solar radaition to yoe-jhonson form 
        
        self.__lmbda_wind_speed=self.pt.lambdas_[0]
        
        self.__lmbda_Solar_Radiation=self.pt.lambdas_[1]   
        
        def __yeojohnson_transform_(self,value, lmbdas):
            transformed_value=yeojohnson([value], lmbda=lmbdas)
            return transformed_value[0]
        
        self.__wind_speed_trans=self.yeojohnson_transform_(self.wind_speed,self.lmbda_wind_speed)
        
        self.__Solar_Radiation_trans=self.yeojohnson_transform_(self.Solar_Radiation,self.lmbda_Solar_Radiation)    
        
        # transforming date in to consumeable form 
        
        
        def __get_string_to_datetime(self,date):
            
            self.__dt = datetime.strptime(date, '%d/%m/%Y').date()
            
            return{'day':self.dt.day,'month':self.dt.month,'year':self.dt.year,'weekday':self.dt.strftime('%A')}
        
        self.__str_to_date=self.get_string_to_datetime(self.date)
        
        # transforming holiday and functining day into numeric 
         
        self.__holiday_dic={'no holiday':1,'holiday':0}
        
        self.__Functioning_Day_dic={'yes':1,'no':0} 

        # storing in to dataframe
        self.__user_input_list=[
            self.hour,self.Temprature,self.Humidity,self.wind_speed_trans,self.Visibility,self.Dew_point_temperature,
            self.Solar_Radiation_trans,self.holiday_dic[self.holiday],self.Functioning_Day_dic[self.functinoning_day] ,
            self.str_to_date['day'], self.str_to_date['month'] ,self.str_to_date['year']  
            ]
        self.__features_list=['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
               'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 'Holiday',
               'Functioning Day', 'Day', 'Month', 'Year']
        self.__df_user=pd.DataFrame([self.user_input_list],columns=self.features_list)
        # converting seasons and weekday into numeric 
        
        def __seasons_to_df(self,seasons):
            
            self.__season_col=['autumn', 'spring', 'summer','winter']
           
            self.__seasons_data=np.zeros((1,len(self.season_col)),dtype='int')
            
            self.__df_seasons=pd.DataFrame(self.seasons_data,columns=self.season_col)
    
            if seasons in self.__season_col:
               
                self.__df_seasons[seasons]=1
            
            return self.df_seasons
        
        self.__season_df=self.seasons_to_df(self.seasons)

        
        def __weekday_to_df(self,weekday):
            
            self.__weekday_cols=['friday','monday','saturday','sunday','thursday','tuesday','wednesday']
            
            self.__weekday_data=np.zeros((1,len(self.weekday_cols)),dtype='int')
            
            self.__df_weekday=pd.DataFrame(self.weekday_data,columns=self.weekday_cols)
            
            if weekday in self.weekday_cols:
               
                self.df_weekday[weekday]=1
            
            return self.df_weekday
        
        self.__week_day_df=self.weekday_to_df(self.weekday)
        
        # now combine into singe dataframe
        
        self.__df=pd.concat([self.df_user,self.season_df,self.week_day_df],axis=1)
        
        self.__df.columns=['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
            'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 'Holiday', 'Functioning Day',
            'Day', 'Month', 'Year', 'Seasons_Autumn', 'Seasons_Spring', 'Seasons_Summer', 
            'Seasons_Winter', 'Weekday_Friday', 'Weekday_Monday', 'Weekday_Saturday', 
            'Weekday_Sunday', 'Weekday_Thursday', 'Weekday_Tuesday', 'Weekday_Wednesday']
        
        self.user_data={'date':self.date,'Hour':self.hour, 'Temperature(°C)':self.Temprature, 'Humidity(%)':self.Humidity, 
                        'Wind speed (m/s)':self.wind_speed, 'Visibility (10m)':self.Visibility,
            'Dew point temperature(°C)':self.Dew_point_temperature, 'Solar Radiation (MJ/m2)':self.Solar_Radiation, 
            'Holiday':self.holiday,  'Functioning Day':self.functinoning_day,'seasons':self.seasons,'weekday':self.weekday}
    
        return self.df
    
    def predict(self):
        
        self.__prediction=self.model.predict(self.df)
        
        return print(f"Rented Bike Demand on : {self.date} on Time:{self.hour} is {round(self.prediction[0])}")
        


if __name__ =="__main__":
    
    ml_model_path=r"C:\Users\Neon4\Machine learning code\project code\souel bike demand project\models\XGBoostRegressor_r2_0_938_v1.pkl"
        
    power_transformer_path=r"C:\Users\Neon4\Machine learning code\project code\souel bike demand project\models\pt_transformer.pkl"    
    
    inference=Inference(ml_model_path, power_transformer_path)
    
    inference.predict()