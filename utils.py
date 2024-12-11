import os
import numpy as np
import sys
from dotenv import load_dotenv
import yaml
import getpass

user = getpass.getuser()

load_dotenv()
DVUTILS_LOCAL_CLONE_PATH = os.environ.get("DVUTILS_LOCAL_CLONE_PATH")
sys.path.insert(0, DVUTILS_LOCAL_CLONE_PATH)

from utils_io import *

## Create ArcGIS Client
client = create_arcgis_client()

## Define Box System Root Directory
box_path = os.path.join("/Users", user, "Library", "CloudStorage", "Box-Box")

## Set the PCA Layers configuration file: PCA Geographies and PCA Types
yaml_file = 'pca-layers.yml'

eval_dir = '_data/evaluation_assignments'
feather_dir = "_data/feather_files"


def load_dict_from_yaml(yaml_file=yaml_file):
    """
    Load a dictionary from a YAML file
    """
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data


def create_subset_dict(in_dict, key, val):
    """
    Create a subset dictionary based on a key-value pair
    """
    return {k: v for k, v in in_dict.items() if v.get(key) == val}


def _set_feather_dir(data_dir=None):
    """
    Set or create directory for Feather files
    """
    # Set Data Directory
    if data_dir is None: data_dir = feather_dir
    if not os.path.exists(data_dir): os.makedirs(data_dir) 

    return data_dir


def _load_agol_if_not_exists(data_dir, filename, feather_file, url, client):
    """
    Check if file exists in directory and extract if not.
    """
    if not os.path.exists(feather_file):
        try:
            # Extract the file
            gdf = pull_geotable_agol(url, client=client)
            print(f"File '{filename}' extracted from AGOL")
            return gdf
        except Exception as e:
            print(f"ERROR extracting '{filename}' from AGOL\n")
            return None
    else:
        print(f"File '{filename}' already exists in '{data_dir}'.\n")
        return None


def agol_to_feather(filename, url, data_dir=None, client=client):
    """
    Extract AGOL dataset to local Feather file
    """
    # Set Data Directory
    data_dir = _set_feather_dir(data_dir)
    feather_file = os.path.join(data_dir, f"{filename}.feather")
    # Load AGOL dataset if file does not exist as Feather file
    gdf = _load_agol_if_not_exists(data_dir, filename, feather_file, url, client)
    if gdf is not None:
        # Save to Feather
        gdf.to_feather(feather_file)
        # Return the file path
        print(f"File saved to {feather_file}\n")


def open_feather(filename, data_dir=None):
    """
    Extract AGOL dataset to local Feather file
    """
    # Set Data Directory
    data_dir = _set_feather_dir(data_dir)
    feather_file = os.path.join(data_dir, f"{filename}.feather")
    # Load Feather file dataset 
    print(f"Opening file from {feather_file}\n")

    return gpd.read_feather(feather_file)


def shapefiles_list(dirs, box_path=box_path):
    """
    Create a list of shapefiles from a list of directories
    """
    shapefiles = []
    for dir in dirs:
        dir_path = os.path.join(box_path, dir)
        shapefiles.extend([(dir_path, file) for file in os.listdir(dir_path) if file.endswith('.shp')])
    
    return shapefiles


def read_shapefiles(shapefiles):
    """
    Read a list of shapefiles into a single GeoDataFrame
    """
    # Create an empty list to store GeoDataFrames
    gdfs = []
    # Iterate over each shapefile and read it into a GeoDataFrame
    for shapefile in shapefiles:
        dir, file = shapefile
        # Construct the full path to the shapefile
        shapefile_path = os.path.join(dir, file)
        try:
            # Read the shapefile into a GeoDataFrame
            gdf = gpd.read_file(shapefile_path)
            # Tag the GeoDataFrame with the source shapefile
            gdf['source'] = file
            # Append the GeoDataFrame to the list
            gdfs.append(gdf.to_crs(26910))
        except Exception as e:
            print(f"Error reading shapefile {shapefile}: {str(e)}")
    
    # Merge all GeoDataFrames into a single GeoDataFrame
    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=26910)
    # Apply fixes
    gdf = _shp_fixes(gdf)

    
    return gdf


def _shp_fixes(gdf):
    """
    Apply fixes to the shapefiles
    """
    # Fix FIPCO column data types for Feather file compatibility
    gdf['fipco'] = gdf['fipco'].astype(str).str.zfill(5)
    
    # Dissolve to single record
    gdf = gdf.fillna(0).dissolve(by=['source', 'joinkey']).reset_index(drop=False)

    # Drop extra columns
    # gdf = gdf.loc[:, :'source']

    return gdf


def replace_nulls(df, columns, replace_dict=None):
    """
    Replace null values in a DataFrame
    """
    if replace_dict is None:
        replace_dict = {0: np.nan, '0': np.nan, 'nan': np.nan}
    df[columns] = df[columns].replace(replace_dict)


def coalesce_columns(df, col_name, col_inputs):
    """
    Coalesce columns in a DataFrame
    """
    df[col_name] = df[col_inputs].fillna(method='ffill', axis=1).iloc[:, -1]
    return df


def process_data_load(dict):
    """
    Process the data_load into tte data dictionary item,
    and create a gdf_id column with unique identifiers
    """
    dict['data'] = dict['data_load'].copy()
    dict['data'].reset_index(drop=True, inplace=True)
    dict['data']['gdf_id'] = 1 + dict['data'].index


def data_key(dict):
    """
    Return the key for the data in the dictionary
    """
    return "data" if "data" in dict.keys() else "data_load"


def create_data_dictionary(dict, filename, data_dir=None):
    """
    Create a data dictionary from a dictionary and save to CSV
    """
    data_dir = _set_feather_dir(data_dir)
    filepath = os.path.join(data_dir, f"{filename}_data_dictionary.csv")
    # Create DataFrame from dictionary
    df = pd.DataFrame(dict).T.reset_index()[['filename', 'name', 'agol', 'url']]
    # Add file extension
    df['filename'] = df['filename'].apply(lambda x: x + '.feather')
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Data Dictionary saved to {filepath}\n")


def simplify_geoms(gdf):
    """
    Simplify and clean Geometries by applying dissolve, explode,
    simplify and repair_geometry functions
    """
    print("Checking geometry validity and repairing geometries prior to dissolve/explode steps") ## JC: Added print statement
    ## Check/Repair Geometries
    gdf = repair_geometry(gdf.query("geometry.notnull()"))
    print(f"GDF Geometry Types: {gdf.geom_type.unique()}")
    ## Convert Multipart features to Single part
    gdf = gdf.dissolve(by=None).reset_index(drop=True)
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    ## Repair Geometries
    gdf = repair_geometry(gdf.query("geometry.notnull()"))
    print(f"GDF Geometry Types: {gdf.geom_type.unique()}")

    return gdf


def create_footprint(gdf, flag_name, dist=False):
    """
    Create Area Footprint from GeoDataframe and assign Flag Name
    to Column
    """
    ## Simplify Dataset
    gdf = gdf[["geometry"]].copy()
    gdf = simplify_geoms(gdf)
    ## Set Area Name
    gdf[flag_name] = 1

    return gdf


def create_footprints_for_dict(input_dict, flag_name, export=True, data_dir=None):
    """
    Iterate through Dictionary object and create area footprints
    by applying the create_footprint function
    """
    for k, v in input_dict.items():
        data_dir = _set_feather_dir(data_dir)
        footprint_filename = f"{v['filename']}_footprint"
        feather_file = os.path.join(data_dir, f"{footprint_filename}.feather")
        if os.path.exists(feather_file):
            print(f"Found Footprint for dataset: {k}. Loading from file")
            v["footprint"] = open_feather(footprint_filename, data_dir)
        else:
            print(f"Creating Footprint for dataset: {k}")
            try:
                print(f"Using: {data_key(v)}")
                v["footprint"] = create_footprint(v[data_key(v)], v[flag_name])
                v["footprint"].plot()
                print("Creation of Footprint completed\n")
                if export:
                    print(f"Exporting Footprint to Feather file")
                    # Set Data Directory
                    v["footprint"].to_feather(feather_file)
                    print(f"Footprint saved to {feather_file}\n")
            except:
                print("Creation of Footprint failed!\n")


def assign_footprint(
                gdf_base,
                gdf_over,
                flag_name,
                gdf_base_id="gdf_id",
                return_share=True
                ):
    """Given an Input Geodataframe, runs Spatial Overlay
    to Parcels and returns Parcel Assignment crosswalk
    """
    ## Check for gdf_id or create
    if (gdf_base_id == 'gdf_id') and (not gdf_base_id in gdf_base.columns):
        print('Creating gdf_id')
        gdf_base.reset_index(drop=True, inplace=True)
        gdf_base["gdf_id"] = 1 + gdf_base.index
    ## Create Base GeoDataframe to Input GeoDataframe correspondence
    print('Creating Base GeoDataframe to Input GeoDataframe correspondence')
    gdf_over_corresp = geo_assign_fields(
        id_df=gdf_base[[gdf_base_id, 'geometry']],
        id_field=gdf_base_id,
        overlay_df=gdf_over,
        overlay_fields=[flag_name],
        return_intersection_area=return_share,
    )
    ## Merge p10 Parcels GeoDataframe to Input GeoDataframe using correspondence,
    ## return Dataframe
    gdf_base_fields = [i for i in gdf_base.columns if i != "geometry"]
    if return_share:
        if (not 'area_sq_m' in gdf_base.columns):
            print('Creating area_sq_m')
            gdf_base['area_sq_m'] = gdf_base.geometry.area
            gdf_base_fields.append("area_sq_m")
    base_over = pd.merge(gdf_base[gdf_base_fields], gdf_over_corresp, on=gdf_base_id, how="left")
    if return_share:
        intersect_area_col = f"{flag_name}_intersect_sq_m"
        share_pct_col = f"{flag_name}_share_pct"
        base_over.rename(columns={"intersection_sq_m": intersect_area_col}, inplace=True)
        base_over[share_pct_col] = base_over[intersect_area_col] / base_over["area_sq_m"]

    return base_over


def assign_footprints_from_dictionary(base_dict, overlay_dict, flag_col):
    """
    Assigns a footprint flag to base_dict for each item in the overlay_dict, based on flag_name.
    """
    for pca_type, type_att in overlay_dict.items():
        try:
            print(f'Dataset {pca_type}: Starting Assignment of Footprint')
            type_att['assignment'] = assign_footprint(
                gdf_base=base_dict['data'],
                gdf_over=type_att['footprint'],
                flag_name=type_att[flag_col]
                )
            print(f'Dataset {pca_type}: Assignment of Footprint complete\n')
        except:
            print(f'Dataset {pca_type}: Assignment of Footprint failed!\n')
            pass


def overlay_surface_percentage_matrix(gdf, id_column='gdf_id', how='intersection'):
    """
    Creates a matrix of overlay surface percentages for each pair of features in the GeoDataFrame.
    
    Parameters:
    gdf (GeoDataFrame): The input GeoDataFrame.
    id_column (str): The name of the column with unique numeric identifiers.
    how (str): The type of overlay operation (default is 'intersection').
    
    Returns:
    matrix (DataFrame): A DataFrame where each cell (i, j) contains the percentage of the surface area of feature i
                        that is overlapped by feature j.
    """
    if id_column not in gdf.columns:
        raise ValueError(f"GeoDataFrame must contain a column named '{id_column}'")

    ids = gdf[id_column].values
    matrix = pd.DataFrame(index=ids, columns=ids, dtype=float)
    
    for i, id_i in enumerate(ids):
        print(f"Processing feature {i + 1} of {len(ids)}")
        feature_i = gdf[gdf[id_column] == id_i]
        for j, id_j in enumerate(ids):
            if id_i == id_j:
                # The percentage of overlay with itself is 100%
                matrix.at[id_i, id_j] = 100.0
            else:
                feature_j = gdf[gdf[id_column] == id_j]
                overlay_result = gpd.overlay(feature_i, feature_j, how=how, keep_geom_type=True)
                
                if not overlay_result.empty:
                    # Calculate the percentage of the surface area of feature i that is overlapped by feature j
                    area_i = feature_i.geometry.area.values[0]
                    area_overlay = overlay_result.geometry.area.sum()
                    percentage_overlay = (area_overlay / area_i) * 100
                else:
                    percentage_overlay = 0.0
                
                matrix.at[id_i, id_j] = percentage_overlay
    
    return matrix