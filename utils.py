import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.validation import explain_validity, make_valid
from shapely.geometry import Polygon
import yaml
from datetime import datetime


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


def open_feather(filename, data_dir=None):
    """
    Open Feather file from Data Directory location
    """
    # Set Data Directory
    data_dir = _set_feather_dir(data_dir)
    feather_file = os.path.join(data_dir, f"{filename}.feather")
    # Load Feather file dataset 
    print(f"Opening file from {feather_file}")

    return gpd.read_feather(feather_file)


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


def process_data_load(dict, data_key='data_load'):
    """
    Process the data_load into the data dictionary item,
    and create a gdf_id column with unique identifiers
    """
    dict['data'] = dict[data_key].copy()
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
    gdf = repair_geometry(gdf.query("geometry.notnull()")) #
    print(f"GDF Geometry Types: {gdf.geom_type.unique()}") #
    ## Convert Multipart features to Single part
    gdf = gdf.dissolve(by=None).reset_index(drop=True)
    # gdf = gdf.explode(index_parts=False).reset_index(drop=True) #
    ## Repair Geometries
    # gdf = repair_geometry(gdf.query("geometry.notnull()")) #
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


def load_footprints_for_dict(input_dict):
    """
    Iterate through Dictionary object and load area footprints
    """
    for k, v in input_dict.items():
        try:
            print(f"Loading data for {k}")
            ## Load PCA type from Feather file
            footprint_filename = f"{v['filename']}_footprint"
            v["footprint"] = open_feather(footprint_filename)
            v['footprint'].plot()       
            print(f"Footprint loaded successfully from {footprint_filename}\n")
        except Exception as e:
            print(f"Failed to load data for {k}!\n")


def assign_footprint(
                gdf_base,
                gdf_over,
                flag_name,
                gdf_base_id="gdf_id",
                return_share=True
                ):
    """Given an Overlay Geodataframe, runs Spatial Overlay
    to a Base Geodataframe and returns Parcel Assignment crosswalk
    """
    ## Check for gdf_id or create
    if (gdf_base_id == 'gdf_id') and (not gdf_base_id in gdf_base.columns):
        print('Creating gdf_id')
        gdf_base.reset_index(drop=True, inplace=True)
        gdf_base["gdf_id"] = 1 + gdf_base.index
    ## Create Base GeoDataframe to Overlay GeoDataframe correspondence
    print('Creating Base GeoDataframe to Overlay GeoDataframe correspondence')
    gdf_over_corresp = geo_assign_fields(
        id_df=gdf_base[[gdf_base_id, 'geometry']],
        id_field=gdf_base_id,
        overlay_df=gdf_over,
        overlay_fields=[flag_name],
        return_intersection_area=return_share,
    )
    ## Merge Base GeoDataframe to Overlay GeoDataframe using correspondence,
    ## return Dataframe
    gdf_base_fields = [i for i in gdf_base.columns if i != "geometry"]
    if return_share:
        print('Calculating area_sq_m')
        if (not 'area_sq_m' in gdf_base.columns):
            gdf_base_fields.append("area_sq_m")
        gdf_base['area_sq_m'] = gdf_base.geometry.area
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


def repair_geometry(gdf):
    """Given a geopandas GeoDataFrame, tests the validity of and repairs GeoDataFrame geometries.

    If no invalid geometries are found, returns the original GeoDataFrame. The function leverages
    the shapely methods is_valid() to check validity and the explain_validity() and make_valid()
    functions. For more information about how these methods and functions work, please refer to the
    shapely documentation: https://shapely.readthedocs.io/en/stable/manual.html#diagnostics

    Author: Joshua Croff

    Args:
        gdf: A Geopandas GeoDataFrame object.

    Returns:
        GeoDataFrame: A Geopandas GeoDataFrame object.
    """

    if gdf.geometry.is_valid.all():
        print("Geodataframe contains valid geometry. No repair necessary.")
        return gdf
    else:
        repaired_gdf = gdf.copy()
        print("Geodataframe contains invalid geometry, starting geometry repair process...\n")
        print(repaired_gdf.geometry.apply(explain_validity).value_counts())
        invalid_before_ct = repaired_gdf[~repaired_gdf.geometry.is_valid].shape[0]

        # Make valid
        repaired_gdf["geometry"] = repaired_gdf.geometry.apply(make_valid)
        invalid_after_ct = repaired_gdf[~repaired_gdf.geometry.is_valid].shape[0]

        if repaired_gdf.geometry.is_valid.all():
            msg = f"\nGeometry repair complete.\nInvalid geometries before repair: {invalid_before_ct}\nInvalid Geometries after repair: {invalid_after_ct}"
            print(msg)
        else:
            msg = "\nGeodataframe still contains invalid geometries. Consider manual fix or revisiting geoprocess for issues that may create invalid geometries."
            print(msg)
        return repaired_gdf


def spot_check_square(pca_types, x_coords, y_coords):
    """Spot check a square area of the PCA types to see if they are clipped correctly.
    Args:
        pca_types (dict): Dictionary of PCA types.
        x_coords (list): List of x coordinates for the square.
        y_coords (list): List of y coordinates for the square.
    """
    # Create the polygon
    polygon = Polygon(zip(x_coords, y_coords))
    # Create a GeoDataFrame with the polygon
    clip_boundary = gpd.GeoDataFrame(geometry=[polygon], crs='EPSG:26910')
    for i in pca_types.keys():
        print(f"Checking {i}")
        layer_source = pca_types[i]['footprint']
        clipped_layer = gpd.clip(layer_source, clip_boundary)
        print(f"Clipped layer has {clipped_layer.shape[0]} rows")
        clipped_layer.plot()

############### MTCPY library functions ###############


# Default CRS for analysis
ANALYSIS_CRS = "EPSG:26910"


def geo_assign_fields(
    id_df,
    id_field,
    overlay_df,
    overlay_fields,
    return_intersection_area=False,
    id_within_pct=None,
):
    """Given an id_df and an overlay_df, assigns the overlay fields.

    Methodology:
    Assigns based on the area with the largest intersection with each id_field (where there are
    duplicate assignments).

    Notes:
        - This is primarily used for generating correspondences, such as new to old parcel id
        - If any overlay_fields also occur in the id_df, append a _y suffix to the overlay field

    Args:
        id_df (geopandas GeoDataFrame): The ID GeoDataFrame
        id_field (str): The name of the ID column in the ID GeoDataFrame
        overlay_df (geopandas GeoDataFrame): The overlay GeoDataFrame
        overlay_fields (list): A list of overlay fields to assign to the ID GeoDataFrame
        return_intersection_area (bool, optional): Flag for whether to return the intersection area
            of the overlay. Defaults to False.
        id_within_pct (float, optional): Value between 0 and 1. If provided, will only assign overlay df values if the id_df
            is within this percentage of the overlay df. Defaults to None.
    Returns:
        geopandas GeoDataFrame: The ID GeoDataFrame with the overlay fields assigned by largest
            intersection area
    """
    a = time.time()
    if id_df.crs != ANALYSIS_CRS or overlay_df.crs != ANALYSIS_CRS:
        print(f"base geo crs: {id_df.crs}")
        print(f"overlay geo crs: {overlay_df.crs}")
        print("Both GeoDataFrames must be in EPSG:26910. Reprojecting:")
        id_df = project_to_analysis_crs(id_df)
        overlay_df = project_to_analysis_crs(overlay_df)

    join_df = gpd.overlay(id_df, overlay_df, how="intersection")
    join_df["intersection_sq_m"] = join_df.geometry.area
    join_df["idx"] = join_df.index

    max_idxs = (
        join_df.groupby(id_field, as_index=False)
        .agg({"intersection_sq_m": "idxmax"})
        .rename(columns={"intersection_sq_m": "idx"})
    )
    join_df = join_df.merge(max_idxs)

    final_fields = [id_field] + overlay_fields

    # calculate intersection area and share of id_df in intersection
    id_df["base_sq_m"] = id_df.geometry.area
    final_assignment = id_df[[id_field, "base_sq_m"]].merge(
        join_df[final_fields + ["intersection_sq_m"]], how="left"
    )
    final_assignment["area_share"] = (
        final_assignment["intersection_sq_m"] / final_assignment["base_sq_m"]
    )

    # set the assignment to None if no more than id_within_pct of the id_df is within the overlay_df
    if id_within_pct is not None:
        final_assignment.loc[final_assignment["area_share"] < id_within_pct, overlay_fields] = None

    b = time.time()
    print(f"took {print_runtime(b-a)}")
    if return_intersection_area:
        return final_assignment[final_fields + ["base_sq_m", "intersection_sq_m", "area_share"]]
    else:
        return final_assignment


def print_runtime(run_seconds):
    """Formats runtime for more readable logging.

    Args:
        run_seconds (float): Runtime (in seconds).

    Returns:
        str: Readable runtime string for logging.
    """
    if run_seconds > 60:
        mins = run_seconds / 60.0
        if mins < 60:
            return "{} minutes".format(round(mins, 4))
        else:
            return "{} hours".format(round(mins / 60.0, 4))
    else:
        return "{} seconds".format(round(run_seconds, 4))
    

def project_to_analysis_crs(geo_df):
    """Checks for whether a GeoDataFrame is in the analysis CRS (EPSG:26910) and reprojects if not.

    Args:
        geo_df (geopandas GeoDataFrame): A geopandas GeoDataFrame needs to be reprojected for
            spatial analysis

    Returns:
        geopandas GeoDataFrame: A geopandas GeoDataFrame in the analysis CRS (EPSG:26910)
    """
    if geo_df.crs != ANALYSIS_CRS:
        print("GeoDataFrame must be in EPSG:26910. Reprojecting:")
        try:
            geo_df = geo_df.to_crs(ANALYSIS_CRS)
        except:
            print("Error reprojecting, correct geometries and re-run.")
            return
    return geo_df
