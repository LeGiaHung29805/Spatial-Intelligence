import geopandas as gpd
import rasterio
from rasterio.mask import mask
import os

class SpatialHandler:
    def __init__(self, boundary_path):
        """Khởi tạo với file ranh giới chuẩn BatXat(moi)1.shp"""
        
        if not os.path.exists(boundary_path):
            raise FileNotFoundError(f"Không tìm thấy file shapefile tại: {boundary_path}")
            
        self.boundary_gdf = gpd.read_file(boundary_path)
        
        
        if self.boundary_gdf.crs != "EPSG:4326":
            self.boundary_gdf = self.boundary_gdf.to_crs("EPSG:4326")
        
    def get_bbox(self):
        """Trả về [minx, miny, maxx, maxy] để lắp vào API OpenTopography/Sentinel Hub"""
        bounds = self.boundary_gdf.total_bounds
        return bounds.tolist()

    def clip_raster(self, input_path, output_path):
        """Cắt ảnh vệ tinh hoặc DEM theo hình dạng huyện Bát Xát"""
        with rasterio.open(input_path) as src:
        
            boundary_projected = self.boundary_gdf.to_crs(src.crs)

            out_image, out_transform = mask(src, boundary_projected.geometry, crop=True)
            out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
        return output_path

    def clean_vector_layers(self, vector_path, output_path):
        """Cắt các lớp Sông, Hồ, Đường theo ranh giới huyện"""
        gdf = gpd.read_file(vector_path)
        
        if gdf.crs != self.boundary_gdf.crs:
            gdf = gdf.to_crs(self.boundary_gdf.crs)
            
        clipped_gdf = gpd.clip(gdf, self.boundary_gdf)
        clipped_gdf.to_file(output_path)
        return output_path