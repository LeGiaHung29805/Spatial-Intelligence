import osmnx as ox

# Tải toàn bộ mạng lưới giao thông (kể cả đường mòn, ngõ xóm nhỏ nhất)
G_osm = ox.graph_from_place("Bat Xat, Lao Cai, Vietnam", network_type='all')

# Lưu lại thành file GraphML mới, hoặc xuất ra Shapefile để gộp với file cũ
ox.save_graphml(G_osm, "models/baxat_osm_full.graphml")