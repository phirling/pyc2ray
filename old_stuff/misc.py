def set_timesteps(cosmo,zred0,zred_end,num_ts,num_out):
    t1 = cosmo.lookback_time(zred0)
    t2 = cosmo.lookback_time(zred_end)
    dt = (t2-t1)/num_ts
    dt_output = (t2-t1)/num_out
    return dt, dt_output