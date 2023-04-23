import sys
sys.path.append("../")
import numpy as np
import astropy.units as u
import pyc2ray as pc2r
import argparse

parser = argparse.ArgumentParser("Generate random source lists used for benchmarks and tests")

parser.add_argument("fname",nargs=1,type=str,help="Name of the source file to write")
parser.add_argument("-N",type=int,default=300,help="Grid Size")
parser.add_argument("-numsrc",type=int,default=100,help="Number of sources")
parser.add_argument("-strength",type=float,default=5e49,help="Ionizing flux of each source")
parser.add_argument("-seed",type=int,default=100,help="Random seed to use with numpy.random")

args = parser.parse_args()
"""
ABOUT
Generate random source lists used for benchmarks and tests

"""

# Parameters
N       = args.N # 300               # Grid Size
numsrc  = args.numsrc #100               # Number of sources
#fname = "100_src_3e50_N300.txt"    # File name
fname = args.fname[0]
flux = args.strength #5e49 #5.0e55
rndmseed = int(args.seed)

pc2r.generate_test_sourcefile(fname,N,numsrc,flux,rndmseed)