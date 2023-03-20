import time

class ClockSet:
    def __init__(self,fname_time) -> None:
        self.fname_time = fname_time
        with open(self.fname_time,'w') as f:
            f.write("Timings file for C2Ray run \n\n")
        self._walltime1 = time.time()
        self._cputime1 = time.process_time()
    
    # Write current timings to timefile
    def report_clocks(self):
        s1 = "Wall clock time: " + self.timestamp_wall()
        s2 = "CPU time: " + self.timestamp_cpu()
        with open(self.fname_time,'a') as f:
            f.write(s1+s2)

    def write_walltimestamp(self,s):
        tw = str(s) + ": " + self.timestamp_wall()
        with open(self.fname_time,'a') as f:
            f.write(tw)
        
    # Wall time formatted as hh:mm:ss
    def timestamp_wall(self):
        hrs_wall, mins_wall, secs_wall = self.s2hms(self.walltime())
        s1 = f"{hrs_wall:n} hour(s), {mins_wall:n} minute(s), {secs_wall:2.2f} second(s). \n"
        return s1
    
    # CPU time formatted as hh:mm:ss
    def timestamp_cpu(self):
        hrs_cpu, mins_cpu, secs_cpu = self.s2hms(self.cputime())
        s2 = f"{hrs_cpu:n} hour(s), {mins_cpu:n} minute(s), {secs_cpu:2.2f} second(s). \n"
        return s2

    # Current Wall time in seconds
    def walltime(self):
        return time.time() - self._walltime1
    
    # Current CPU time in seconds
    def cputime(self):
        return time.process_time() - self._cputime1
    
    # Convert seconds to hh:mm:ss
    def s2hms(self,secs):
        hrs = secs // 3600
        secs = secs % 3600
        mins = secs // 60
        secs = secs % 60
        return hrs, mins, secs