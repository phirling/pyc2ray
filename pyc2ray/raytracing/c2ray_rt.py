import sys
sys.path.append("../")
import c2ray as c2r

def do_all_sources(srcflux,srcpos,r_RT,subboxsize,coldensh_out,sig,dr,ndens,xh_av,phi_ion,loss_fraction):
    c2r.raytracing.do_all_sources(srcflux,srcpos,r_RT,subboxsize,coldensh_out,sig,dr,ndens,xh_av,phi_ion,loss_fraction)