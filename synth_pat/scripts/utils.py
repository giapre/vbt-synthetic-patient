import numpy as np 
import pandas as pd
import glob
from paths import Paths
import os

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), '../..'))
RESOURCES_DIR = Paths.RESOURCES

import os
import numpy as np

def compact_bold_results(output_dir, means, stds, wes):
    """
    Stacks all saved .npy files into a single .npz and saves parameter list.
    Deletes individual files ONLY after successful save.

    Parameters
    ----------
    output_dir : str
        Directory containing bold .npy files
    means : array-like
    stds : array-like
    wes : array-like
    """
    out_file = os.path.join(output_dir, "JJa_bold_all.npz")

    # ---- do nothing if output already exists ----
    if os.path.exists(out_file):
        return out_file

    bold_list = []
    param_list = []
    files_to_delete = []

    for mean in means:
        for std in stds:
            fname = os.path.join(output_dir, f"JJa_{mean}_{std}_bold.npy")

            bold = np.load(fname)  # (T, R, n_we)
            bold_list.append(bold)
            files_to_delete.append(fname)

            for w_idx in range(bold.shape[2]):
                param_list.append((mean, std, wes[w_idx]))

    # concatenate along simulation axis
    bold_all = np.concatenate(bold_list, axis=2)  # (T, R, S)
    param_list = np.array(param_list)              # (S, 3)

    np.savez(
        out_file,
        bold=bold_all,
        params=param_list,
    )

    if not os.path.exists(out_file):
        raise RuntimeError("bold_all.npz was not created. Aborting deletion.")

    # Optional: try loading it to be extra safe
    _ = np.load(out_file)

    # delete small files
    for fname in files_to_delete:
        os.remove(fname)

    return out_file


def prepare_fs_default(resources_dir=Paths.RESOURCES):
    """Read and prepare the freesurfer look up table"""

    fs_default = pd.read_csv(f'{resources_dir}/fs_default.txt', sep='\s+', comment='#')
    fs_default.rename(columns={'Unknown': 'Region', '???': 'Label'}, inplace=True)
    # Remove the two extra thalamic regions
    idx_to_remove = [fs_default[fs_default['Region']=='Left-Thalamus-Proper'].index[0], fs_default[fs_default['Region']=='Right-Thalamus-Proper'].index[0]]
    fs_default.drop(index=idx_to_remove, inplace=True)

    return fs_default

def prepare_FreeSurferColorLUT(resources_dir=Paths.RESOURCES):
    
    lut = pd.read_csv(f'{Paths.RESOURCES}/FreeSurferColorLUT.txt', sep='\s+', comment='#', names=["No", "Region", "R", "G", "B", "A"])
    return(lut)


def get_subcortical_labels(atlas):
    """
    Return subcortical regions of the corresponding atlas
    """
    if atlas == 'dk':
        return ['L.CER', 'L.TH', 'L.CA', 'L.PU', 'L.PA', 'L.HI', 'L.AM', 'L.AC',
                'R.TH', 'R.CA', 'R.PU', 'R.PA', 'R.HI', 'R.AM', 'R.AC', 'R.CER',
                'L.SN', 'L.VTA', 'L.RN', 'R.SN', 'R.VTA', 'R.RN']
    elif atlas == 'schaefer':
        return ['Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen',
                'Left-Pallidum', 'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens-area',
                'Right-Cerebellum-Cortex', 'Right-Thalamus-Proper', 'Right-Caudate',
                'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
                'Right-Accumbens-area', 'Left-Nigra', 'Left-VTA', 'Right-Nigra', 'Right-VTA']
    elif atlas == 'aal2':
        return ['Caudate_L', 'Caudate_R', 'Putamen_L', 'Putamen_R', 'Pallidum_L', 'Pallidum_R',
                'Thalamus_L', 'Thalamus_R', 'Amygdala_L', 'Amygdala_R',
                'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R', 'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R',
                'Cerebelum_3_L', 'Cerebelum_3_R', 'Cerebelum_4_5_L', 'Cerebelum_4_5_R',
                'Cerebelum_6_L', 'Cerebelum_6_R', 'Cerebelum_7b_L', 'Cerebelum_7b_R',
                'Cerebelum_8_L', 'Cerebelum_8_R', 'Cerebelum_9_L', 'Cerebelum_9_R',
                'Cerebelum_10_L', 'Cerebelum_10_R',
                'Vermis_1_2', 'Vermis_3', 'Vermis_4_5', 'Vermis_6', 'Vermis_7', 'Vermis_8',
                'Vermis_9', 'Vermis_10', 'Nigra_L', 'VTA_L', 'Nigra_R', 'VTA_R']
    else:
        raise ValueError(f"Atlas {atlas} not recognized")
    
def get_cortical_labels(atlas):
    """
    Return cortical labels
    """
    fs_default = prepare_fs_default()
    labels = fs_default['Label']
    subcortical_labels = get_subcortical_labels(atlas)
    cortical_labels = [lab for lab in labels if lab not in subcortical_labels]

    return cortical_labels

def get_cortical_indices(atlas):
    """
    Return cortical labels
    """
    fs_default = prepare_fs_default()
    fs_default_labels = fs_default['Label'].to_list()
    cortical_labels = get_cortical_labels(atlas)
    cortical_indices = [fs_default_labels.index(cortical_label) for cortical_label in cortical_labels]

    return cortical_indices

def get_raw_thickness(RAW_DATA_DIR):
    """Read the thickness csv file for the two haemispheres, concatenate them and assures they have the good shape"""
    lgmv = pd.read_csv(f'{RAW_DATA_DIR}/gray_matter_data_left.csv', sep='\t', index_col='lh.aparc.thickness')
    rgmv = pd.read_csv(f'{RAW_DATA_DIR}/gray_matter_data_right.csv', sep='\t', index_col='rh.aparc.thickness')
    gmv = pd.concat([lgmv, rgmv], axis=1)

    assert gmv.shape[1] == 2 * lgmv.shape[1]
    assert gmv.shape[1] == 2 * rgmv.shape[1]

    gmv.rename(columns={'lh_MeanThickness_thickness':'LThickness', 'rh_MeanThickness_thickness':'RThickness'}, inplace=True)
    if gmv.index[0].startswith('sub'):
        gmv.index = gmv.index.str.replace('sub-', '', regex=False)

    return gmv

def adjust_thick_template(demo_and_gmv, template):
    """
    Transforms the demographics and thickness dataframe in the correct format for Centile software
    
    :param demo_and_gmv: dataframe with demographics and thickness information for each patient
    :param template: the reference template csv from centile softwaer
    """
    for col in demo_and_gmv.columns:
        for tempcol in template.columns:
            if col.startswith('lh_') & tempcol.startswith('L_'): 
                if col.split('_')[1] == tempcol.split('_')[1]:
                    demo_and_gmv.rename(columns={col: tempcol}, inplace=True)
            elif col.startswith('rh_') & tempcol.startswith('R_'): 
                if col.split('_')[1] == tempcol.split('_')[1]:
                    demo_and_gmv.rename(columns={col: tempcol}, inplace=True)

    demo_and_gmv.reset_index(inplace=True)
    demo_and_gmv.rename(columns={'pid':'SubjectID'}, inplace=True)
    demo_and_gmv.rename(columns={'lh_entorhinal_thickness':'L_entorhil_thickavg', 'lh_supramarginal_thickness':'L_supramargil_thickavg', 'rh_entorhinal_thickness':'R_entorhil_thickavg', 'rh_supramarginal_thickness':'R_supramargil_thickavg'}, inplace=True)
    demo_and_gmv['SITE'] = ['A'] * len(demo_and_gmv)
    demo_and_gmv['Vendor'] = ['na'] * len(demo_and_gmv)
    demo_and_gmv['FreeSurfer_Version'] = ['7.4'] * len(demo_and_gmv)
    
    Males = demo_and_gmv[demo_and_gmv['sex'] == 'Male']
    Males = Males[template.columns]
    Females = demo_and_gmv[demo_and_gmv['sex'] == 'Female']
    Females = Females[template.columns]

    return Males, Females

import pandas as pd

def rename_to_fs_lut_labels(df, lut_df):
    """
    Renames columns from 'L_region_thickavg' format to LUT 'Label' format.
    """
    # 1. Create a helper dictionary from the LUT: { 'ctx-lh-bankssts': 'L.BSTS', ... }
    # We strip the 'ctx-lh-' and 'ctx-rh-' to make matching easier
    lut_map = {}
    for _, row in lut_df.iterrows():
        region_name = str(row['Region'])
        # Standardize FS region names to match your CSV format (L_ or R_)
        clean_reg = region_name.replace('ctx-lh-', 'L_').replace('ctx-rh-', 'R_')
        lut_map[clean_reg] = row['Label']

    new_column_names = {}
    for col in df.columns:
        # Remove the '_thickavg' suffix to get 'L_bankssts'
        base_name = col.replace('_thickavg', '')
        
        # Look up the base name in our LUT map
        if base_name in lut_map:
            new_column_names[col] = lut_map[base_name]
        else:
            # Keep original name if no match is found (e.g., SubID or metadata)
            new_column_names[col] = col

    df.rename(columns=new_column_names, inplace=True)
    df.rename(columns={'L_entorhil_thickavg': 'L.EC', 'R_entorhil_thickavg': 'R.EC',
                       'L_supramargil_thickavg': 'L.SMG', 'R_supramargil_thickavg': 'R.SMG'}, inplace=True)
            
    return df

def rename_to_fs_lut_region(df, lut_df):
    """
    Renames columns from 'L_region_thickavg' format to LUT 'Region' format.
    """
    # 1. Create a helper dictionary from the LUT: { 'ctx-lh-bankssts': 'L.BSTS', ... }
    # We strip the 'ctx-lh-' and 'ctx-rh-' to make matching easier
    lut_map = {}
    for _, row in lut_df.iterrows():
        region_name = str(row['Region'])
        # Standardize FS region names to match your CSV format (L_ or R_)
        clean_reg = region_name.replace('ctx-lh-', 'L_').replace('ctx-rh-', 'R_')
        lut_map[clean_reg] = row['Region']

    new_column_names = {}
    for col in df.columns:
        # Remove the '_thickavg' suffix to get 'L_bankssts'
        base_name = col.replace('_thickavg', '')
        
        # Look up the base name in our LUT map
        if base_name in lut_map:
            new_column_names[col] = lut_map[base_name]
        else:
            # Keep original name if no match is found (e.g., SubID or metadata)
            new_column_names[col] = col

    df.rename(columns=new_column_names, inplace=True)
    df.rename(columns={'L_entorhil_thickavg': 'ctx-lh-entorhinal', 'R_entorhil_thickavg': 'ctx-rh-entorhinal',
                       'L_supramargil_thickavg': 'ctx-lh-supramarginal', 'R_supramargil_thickavg': 'ctx-rh-supramarginal'}, inplace=True)
            
    return df

def dk_extract_gray_matter(regions_lines, stats_folder):
    #regions_lines = np.loadtxt(stats_folder + "fs_default.txt", dtype=str) 
    lh_aparc_lines = np.loadtxt(stats_folder + "lh.aparc.stats", dtype=str) 
    rh_aparc_lines = np.loadtxt(stats_folder + "rh.aparc.stats", dtype=str) 
    aseg_lines = np.loadtxt(stats_folder + "aseg.stats", dtype=str) 
    
    print(f'Number of regions from fs_default: {len(regions_lines)}')
    print(f'Number of regions from aparc: {len(rh_aparc_lines)}')
    print(f'Number of regions from aseg: {len(aseg_lines)}')

    gm_region_volume = []
    for regions_line in regions_lines:
        for aseg_line in aseg_lines:
            if aseg_line[4] in regions_line[2]:
                gm_region_volume.append([regions_line[1], regions_line[2], aseg_line[3]])
                
    for regions_line in regions_lines:
        for lh_aparc_line in lh_aparc_lines:
            if '-lh-' in regions_line[2]:
                if lh_aparc_line[0] == regions_line[2].split('-')[2]:
                    gm_region_volume.append([regions_line[1], regions_line[2], lh_aparc_line[3]])
                    
    for regions_line in regions_lines:            
        for rh_aparc_line in rh_aparc_lines:
            if '-rh-' in regions_line[2]:
                if rh_aparc_line[0] == regions_line[2].split('-')[2]:
                    gm_region_volume.append([regions_line[1], regions_line[2], rh_aparc_line[3], ])
        
    for region in gm_region_volume:
        if "Proper" in region[2]: gm_region_volume.remove(region)

    print(f'Elements in the gray matter volume list: {len(gm_region_volume)}')
    return gm_region_volume

# ======================
# CONNECTOME UTILS
# ======================

def adjust_dopamine_connectome(sub, weights_file, atlas):
    """
    Add the dopaminergic nuclei to the connectome weights and the respective connection weights based on the local D1R density. 
    Takes a patient weights and returns the corresponding new weights_with_dopa dataframe. 
    """

    # Load and normalize the weights and the receptors
    weights = np.loadtxt(weights_file)
    d1r = pd.read_csv(os.path.join(RESOURCES_DIR, 'Masks', f'{atlas}_D1_D2_5HT2A_receptor_data.csv'), index_col=0)
    weights = weights/np.max(weights)
    d1r['D1_density'] = d1r['D1_density'] / d1r['D1_density'].max()

    # Take the correct labels 
    if atlas == 'dk':
        lut = prepare_fs_default()
        regions_labels_default = lut['Label']
        
        weightsdf = pd.DataFrame(data=weights, index=regions_labels_default, columns=regions_labels_default)
        
        for region in d1r.index:
            if 'L.' in region:
                weightsdf.loc[region, 'L.VTA'] = d1r.loc[region, 'D1_density']/10
                weightsdf.loc[region, 'L.SN'] = d1r.loc[region,'D1_density']/10
                weightsdf.loc['L.VTA', region] = d1r.loc[region,'D1_density']/10
                weightsdf.loc['L.SN', region, ] = d1r.loc[region,'D1_density']/10
        
            if 'R.' in region:
                weightsdf.loc[region, 'R.VTA'] = d1r.loc[region,'D1_density']/10
                weightsdf.loc[region, 'R.SN'] = d1r.loc[region,'D1_density']/10
                weightsdf.loc['R.VTA', region] = d1r.loc[region,'D1_density']/10
                weightsdf.loc['R.SN', region, ] = d1r.loc[region,'D1_density']/10
                    
    elif atlas == 'aal2':
        regions_labels_default = pd.read_csv(f'{RESOURCES_DIR}aal2_default.txt', sep='\t', index_col=0, header=None)[1]
        weightsdf = pd.DataFrame(data=weights, index=regions_labels_default, columns=regions_labels_default)
        
        for i, row in d1r.iterrows():
            region = str(row['x_labels'])
            if region.endswith('_L'):
                weightsdf.loc[region, 'VTA_L'] = row['receptor_density']/10
                weightsdf.loc[region, 'SN_L'] = row['receptor_density']/10
                weightsdf.loc['VTA_L', region] = row['receptor_density']/10
                weightsdf.loc['SN_L', region, ] = row['receptor_density']/10
        
            if region.endswith('_R'):
                weightsdf.loc[region, 'VTA_R'] = row['receptor_density']/10
                weightsdf.loc[region, 'SN_R'] = row['receptor_density']/10
                weightsdf.loc['VTA_R', region] = row['receptor_density']/10
                weightsdf.loc['SN_R', region, ] = row['receptor_density']/10
    else:
        print("More atlases will be available soon!")
        return
    
    weightsdf.fillna(0, inplace=True)
    Ce_mask = pd.read_csv(f'{RESOURCES_DIR}/Masks/{atlas}_exc_mask.csv', index_col=0)
    weightsdf = weightsdf.loc[Ce_mask.index, Ce_mask.columns]
    print(f"Connectome for patient {sub} has been adjusted")
    
    return weightsdf

def adjust_serotonine_connectome(sub, weights_file, atlas):
    """
    Add the serotoninergic nuclei to the connectome weights and the respective connection weights based on the local 5HT2a density. 
    Takes the weights_with_dopa of a patient and returns the corresponding new weights_with_sero_and_dopa dataframe. 
    """
    weightsdf = pd.read_csv(weights_file, index_col=0)
    Ser = pd.read_csv(os.path.join(RESOURCES_DIR, 'Masks', f'{atlas}_D1_D2_5HT2A_receptor_data.csv'), index_col=0)
    weightsdf = weightsdf/np.max(weightsdf)
    Ser['5HT2A_density'] = Ser['5HT2A_density'] / Ser['5HT2A_density'].max()
    print(f'Weights before adjustment: {weightsdf.shape}')
    
    if atlas == 'dk':
        
        weightsdf.loc['L.RN'] = 0.
        weightsdf.loc['R.RN'] = 0.
        weightsdf['L.RN'] = 0.
        weightsdf['R.RN'] = 0.

        for region in Ser.index:
            if region.startswith('L.'):
                weightsdf.loc[region, 'L.RN'] = Ser.loc[region, '5HT2A_density']/10
                weightsdf.loc['L.RN', region, ] = Ser.loc[region, '5HT2A_density']/10
        
            if region.startswith('R.'):
                weightsdf.loc[region, 'R.RN'] = Ser.loc[region, '5HT2A_density']/10
                weightsdf.loc['R.RN', region] = Ser.loc[region, '5HT2A_density']/10
        
        print(f'Weights after adjustment: {weightsdf.shape}')
    
    elif atlas == 'aal2':
        
        for i, row in Ser.iterrows():
            region = str(row['x_labels'])
            if region.endswith('_L'):
                weightsdf.loc[region, 'RN_L'] = row['receptor_density']/10
                weightsdf.loc['RN_L', region, ] = row['receptor_density']/10
        
            if region.endswith('_R'):
                weightsdf.loc[region, 'RN_R'] = row['receptor_density']/10
                weightsdf.loc['RN_R', region, ] = row['receptor_density']/10
    
            if region.startswith('Vermis'):
                weightsdf.loc[region, 'RN_R'] = row['receptor_density']/10
                weightsdf.loc['RN_R', region, ] = row['receptor_density']/10
                weightsdf.loc[region, 'RN_L'] = row['receptor_density']/10
                weightsdf.loc['RN_L', region, ] = row['receptor_density']/10
    
    weightsdf.fillna(0, inplace=True)
    Ce_mask = pd.read_csv(f'{RESOURCES_DIR}/Masks/{atlas}_sero_exc_mask.csv', index_col=0)
    weightsdf = weightsdf.loc[Ce_mask.index, Ce_mask.columns]
    print(f"Connectome for patient {sub} has been adjusted")
    
    return weightsdf

def adjust_serotonin_lengths(pid, lenghts_file, atlas):
    """
    Add the  dopaminergic and serotoninergic nuclei to the connectome lengths and the respective connection lengths copying those of the cerebellum. 
    Takes the lenghts_file of a patient and returns the corresponding new lengths_with_sero_and_dopa dataframe. 
    """

    L = np.loadtxt(lenghts_file) # read the file created by the dwi preprocessing pipeline

    # create the dataframe using freesurfer default list of regions
    lut = prepare_fs_default()
    regions_labels_default = lut['Label']
    Ldf = pd.DataFrame(L, index=regions_labels_default, columns=regions_labels_default)

    # adding the midbrain nuclei (serotonin and dopamine)
    if atlas=='dk':
        regions_to_add = ['L.VTA', 'L.SN', 'L.RN', 'R.VTA', 'R.SN', 'R.RN']

        for region in regions_to_add:
            if region.startswith('L.'):
                Ldf.loc[region, :] = Ldf.loc['L.CER', :]
                Ldf.loc[:, region] = Ldf.loc[:, 'L.CER']
            elif region.startswith('R.'):
                Ldf.loc[region, :] = Ldf.loc['R.CER', :]
                Ldf.loc[:, region] = Ldf.loc[:, 'R.CER']
    
    # Ensure the region order is the same as in the masks
    Ldf.fillna(0, inplace=True)
    Ce_mask_dir = os.path.join(RESOURCES_DIR, 'Masks', f'{atlas}_sero_exc_mask.csv')
    Ce_mask = pd.read_csv(Ce_mask_dir, index_col=0)
    Ldf = Ldf.loc[Ce_mask.index, Ce_mask.columns]

    return Ldf

def merge_centile_results(xlsx_files, score, PROCESSED_DIR):
    """
    Merging centile csv results in one csv only (male and females together usually)
    
    :param xlsx_files: the list of csv files to merge
    :param score: the centile score (zscre, MAE, prediction...)
    :param PROCESSED_DIR: where the centile results are
    """
    all_dfs = []

    for xlsx_file in xlsx_files:

        xlsx_path = os.path.join(PROCESSED_DIR, f"{xlsx_file}.xlsx")
        base_name = os.path.splitext(os.path.basename(xlsx_path))[0] + '_centile_results'
        folder_path = os.path.join(PROCESSED_DIR, base_name)

        if not os.path.isdir(folder_path):
            print(f"Skipping {base_name}: folder not found")
            continue

        print(f"Processing {base_name}")

        # ---- Read metadata ----
        meta_df = pd.read_excel(xlsx_path)

        required_cols = {"SubjectID", "age", "sex"}
        if not required_cols.issubset(meta_df.columns):
            raise ValueError(f"{xlsx_path} missing required columns")

        subject_id = meta_df["SubjectID"]
        age = meta_df["age"]
        sex = meta_df["sex"]

        # ---- Read z_score CSVs ----
        score_files = glob.glob(os.path.join(folder_path, f"{score}*.csv"))

        if len(score_files) == 0:
            print(f"No {score} files in {folder_path}")
            continue

        for csv_path in score_files:
            z_df = pd.read_csv(csv_path)

            # Add metadata columns
            z_df["SubjectID"] = subject_id
            z_df["age"] = age
            z_df["sex"] = sex

            all_dfs.append(z_df)

    # ---- Concatenate everything ----
    if len(all_dfs) == 0:
        raise RuntimeError("No data collected")

    final_df = pd.concat(all_dfs, ignore_index=True)
    # Remove useless columns for group scores
    if score not in ['zscore', 'prediction']:
        final_df.drop(columns=['age', 'SubjectID'], inplace=True)

    return final_df
