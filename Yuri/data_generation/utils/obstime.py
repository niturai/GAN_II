import numpy as np

class ObsTime():
      """
      Creates an object to calculate the number of steps of observation in a given observational interval.
      Calculate the starting, observing, and ending steps in every observational interval from the first observation.
      Also, it creates an array of Julian days of observation for a given average time from each observational interval.
      """
      def obslen(self, s_obs, e_obs, N_obs):
          """
          Calculate the number of steps at each observational day due to observational interval.
          
          Parameters :
          ----------

          s_obs : list
                  List of starting time (julian day) of each observational interval.
          e_obs : list
                  List of ending time (julian day) of each observational interval.
          N_obs : float
                  The average time of observation of SII in days.
             
          Returns :
          -------
          steps : array
                  Return the number of steps in each observational interval.
          """
          st = []
          for i in range(0, len(s_obs), 1):
               dobs = e_obs[i] - s_obs[i]
               dobs /= N_obs
               st.append(int(dobs+1))
          step = np.array(st)
          return step   

      def nobs(self, s_obs, steps, N_obs):
          """
          Start and end time after each observational interval.
          
          Parameters :
          ----------
          
          s_obs : list
                  List of starting time (julian day) of each observational interval.
          steps : array
                  The number of steps in each observational interval.
          N_obs : float
                  The average time of observation of SII in days.
             
          Returns :
          -------
          dst, den : array
                     It return start timing and end timing from the first observation for each observational interval.
          """
          ds = []
          for i in range(0, len(s_obs), 1):
              a = s_obs[i] - s_obs[0]
              a /= N_obs
              ds.append(int(a))
          dst = np.array(ds)                                 
          den = dst + steps                        
          return dst, den

      # julian day for given observational interval
      def julday(self, s_obs, e_obs, steps):
          """
          Julian days for given observational interval.
          
          Parameters :
          ----------
          
          s_obs : list
                  List of starting time (julian day) of each observational interval.
          e_obs : list
                  List of ending time (julian day) of each observational interval.
          steps : array
                  The number of steps in each observational interval. 
               
          Returns :
          -------
          jlday : array
                  Return the array of all observational julian days.
          """
          jul = []
          for i in range(0, len(steps), 1):
              jday = np.linspace(s_obs[i], e_obs[i], steps[i])
              jul.append(jday)               
          return np.concatenate(jul)

