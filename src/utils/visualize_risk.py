import os
import sys
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
import geopandas as gpd
from sqlalchemy import text
import joblib
from datetime import datetime
from shapely.geometry import Point

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
csdl_path = os.path.join(project_root, 'src', 'CSDL')
if csdl_path not in sys.path: sys.path.append(csdl_path)

from config.db_config import get_engine

SHP_PATH = os.path.join(project_root, 'data', 'shapefiles', 'boundary', 'BatXat(moi)1.shp')
CSV_GEOGRAPHY = os.path.join(project_root, 'data', 'processed', 'landslide_training_data.csv')
CSV_WEATHER = os.path.join(project_root, 'data', 'processed', 'flood_full_history.csv')
RESULTS_DIR = os.path.join(project_root, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_building_risk_map():
    print("Đang khởi tạo hệ thống giám sát tích hợp (Bản đầy đủ)...")
    engine = get_engine()
    
    boundary_gdf = gpd.read_file(SHP_PATH).to_crs("EPSG:4326")
    boundary_poly = boundary_gdf.unary_union

    try:
        with engine.connect() as conn:
            m_path = conn.execute(text("SELECT model_path FROM batxat_model_registry WHERE model_target = 'LANDSLIDE' AND is_active = TRUE ORDER BY created_at DESC LIMIT 1")).scalar()
            raw_buildings = conn.execute(text("SELECT id, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon FROM batxat_buildings")).fetchall()
            try:
                raw_roads = conn.execute(text("SELECT id, COALESCE(name, 'Đường không tên') as name, ST_AsGeoJSON(geom) as geojson FROM batxat_roads")).fetchall()
            except: raw_roads = []
        
        rf_model = joblib.load(os.path.join(project_root, m_path) if not os.path.isabs(m_path) else m_path)
    except Exception as e:
        print(f"Lỗi nạp hệ thống: {e}"); return

    df_weather = pd.read_csv(CSV_WEATHER)
    p3d = float(df_weather.iloc[-1]['precip_3d_BAT_XAT'])
    soil = float(df_weather.iloc[-1].get('soil_BAT_XAT', 0.5))

    df_geo = pd.read_csv(CSV_GEOGRAPHY)
    gdf_geo = gpd.GeoDataFrame(df_geo, geometry=gpd.points_from_xy(df_geo.Coord_X, df_geo.Coord_Y), crs="EPSG:4326")
    gdf_geo = gdf_geo[gdf_geo.geometry.within(boundary_poly)].copy()
    
    ls_features = ['Slope', 'Elevation', 'Dist_to_Water', 'precip_3d_BAT_XAT', 'soil_BAT_XAT']
    gdf_geo['precip_3d_BAT_XAT'], gdf_geo['soil_BAT_XAT'] = p3d, soil
    gdf_geo['risk_score'] = rf_model.predict_proba(gdf_geo[ls_features].fillna(0))[:, 1]

    df_buildings = pd.DataFrame(raw_buildings, columns=['id', 'lat', 'lon'])
    gdf_buildings = gpd.GeoDataFrame(df_buildings, geometry=gpd.points_from_xy(df_buildings.lon, df_buildings.lat), crs="EPSG:4326")
    gdf_buildings = gdf_buildings[gdf_buildings.geometry.within(boundary_poly)].copy()
    gdf_buildings = gpd.sjoin_nearest(gdf_buildings, gdf_geo[['risk_score', 'geometry']], distance_col="dist")

    m = folium.Map(location=[22.58, 103.70], zoom_start=12, tiles='CartoDB dark_matter')

    # Lớp 1: Heatmap
    HeatMap(gdf_geo[['Coord_Y', 'Coord_X', 'risk_score']].values.tolist(), radius=15, blur=10, min_opacity=0.3, name="Vùng nguy cơ (AI)").add_to(m)

    # Lớp 2: Đường xá (Road Risk)
    road_group = folium.FeatureGroup(name="Rủi ro Tuyến đường").add_to(m)
    road_risk_color = 'red' if p3d > 70 else ('orange' if p3d > 40 else 'lime')
    for r in raw_roads:
        folium.GeoJson(r.geojson, style_function=lambda x, color=road_risk_color: {'color': color, 'weight': 4, 'opacity': 0.8}).add_to(road_group)

    # Lớp 3: Nhà dân
    marker_cluster = MarkerCluster(name="Giám sát hộ dân").add_to(m)
    for idx, row in gdf_buildings.iterrows():
        r = row['risk_score']
        color = 'red' if r > 0.7 else ('orange' if r > 0.4 else 'cyan')
        gmaps_link = f"https://www.google.com/maps?q={row.lat},{row.lon}"
        
        popup_html = f"""<div style='width:180px'><b>ID: {row.id}</b><br>Rủi ro: {r*100:.1f}%<br><a href='{gmaps_link}' target='_blank'>📍 Google Maps</a></div>"""
        folium.CircleMarker([row.lat, row.lon], radius=4, color=color, fill=True, popup=folium.Popup(popup_html, max_width=200)).add_to(marker_cluster)

    dashboard_html = f'''
        <div style="position: fixed; top: 10px; right: 10px; width: 280px; z-index:9999; 
        background-color: rgba(0,0,0,0.8); color: white; padding: 15px; border-radius: 10px; border: 1px solid cyan; font-family: sans-serif;">
            <h4 style="margin:0 0 10px 0; color:cyan;">GuardBátXát Intelligence</h4>
            <small>{datetime.now().strftime('%d/%m/%Y %H:%M')}</small><br>
            <hr style="border: 0.5px solid #333;">
            <small>Mưa 3 ngày: <b style="color:yellow;">{p3d}mm</b></small><br>
            <small>Nhà dân quét: <b>{len(gdf_buildings)}</b></small><br>
            <small>Tình trạng đường: <b style="color:{road_risk_color};">{'NGUY HIỂM' if p3d > 40 else 'AN TOÀN'}</b></small>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(dashboard_html))
    
    folium.LayerControl().add_to(m)
    m.add_child(folium.LatLngPopup())
    m.save(os.path.join(RESULTS_DIR, "Bao_Cao_Tien_Do_Full.html"))
    print(f"Đã tạo xong bản báo cáo đầy đủ chức năng!")

if __name__ == "__main__":
    generate_building_risk_map()