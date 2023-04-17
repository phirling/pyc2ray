from readsources import read_sources

srcpos, srcflux, numsrc = read_sources("sourcelist.txt",20,"pyc2ray_octa")

print(numsrc)
print(srcpos)
print(srcflux)
