import sys
sys.path.insert(0, '/Users/zhangronghua/Documents/科研小组/gnn-euclidean-local')
from build_highway import load_off, build_mesh_graph, build_quadtree, find_boundary_nodes
p='/Users/zhangronghua/魏俊秋教授科研小组/gnn-highewaynetwork-oracle/dataset/small/small_terrain.off'
v,f=load_off(p); gi,coords=build_mesh_graph(v,f)
print('|V|',len(v))
for md,cap in [(2,256),(3,64),(3,128),(3,256),(4,64),(4,128),(4,256),(5,64),(5,128),(5,256)]:
    leaf_of,nl,occ=build_quadtree(coords, md, cap, True)
    b=find_boundary_nodes(gi, leaf_of)
    print(f'd{md}/c{cap}: leaves={nl} K={len(b)} ({100*len(b)/len(v):.1f}%)')
