import argparse

import pandas as pd
import numpy as np

import laspy
import anvil
from tqdm import tqdm

NETHERITE = 0
ANDESITE = 1
QUARTZ = 2
LEAVES = 3
DIRT = 4
STONE = 5
WATER = 6
GLASS = 7
TALLGRASS = 8
DOUBLE_PLANT = 9
MOSS_BLOCK = 10

def initialize():
    tqdm.pandas()

    parser = argparse.ArgumentParser(
                    prog="Minecraftizzzer",
                    description="Great script",
                    epilog="I am really proud... /j")

    parser.add_argument("filename")
    parser.add_argument("-g", "--ground", default="stone")
    parser.add_argument("-n", "--noise", action="store_true") # include noise
    parser.add_argument("-r", "--rest", default="building")

    args = parser.parse_args()

    filename = args.filename
    ground = args.ground
    noise = args.noise
    rest = args.rest

    blocks_rgb = np.array([[0]*3, [32767]*3, [65535]*3])
    blockmap = [anvil.Block("minecraft", b) for b in ["netherite_block", "polished_andesite", "quartz", "oak_leaves", "dirt", "stone", "water", "light_gray_stained_glass", "tallgrass", "double_plant", "moss_block"]]

    return filename, ground, noise, rest, blocks_rgb, blockmap


def load_data(filename):
    with laspy.open(filename) as f:
        las = f.read()

    df = pd.DataFrame(las.points.array)
    df["block"] = pd.NA

    colors = "red" in df.columns
    if not colors:
        print("No colors in data, buildings will be filled with polished andesite\n")

    return df, colors


def convert_init(df):
    df = df.copy()

    df.X = (df.X - df.X.min()) / 100
    df.Y = (df.Y - df.Y.min()) / 100
    df.Z = (df.Z - df.Z.min()) / 100

    df.X = np.ceil(df.X).astype(int)
    df.Y = np.ceil(df.Y).astype(int)
    df.Z = np.ceil(df.Z).astype(int)

    return df


def process_buildings(df, blocks_rgb, colors):
    df = df.copy()
    # Filter out to building-classified points
    df_buildings = df[df.raw_classification == 6]
    print(f"No. buildings: {len(df_buildings)}")
    if colors:
        df_buildings = determine_closest(df_buildings, blocks_rgb)
    else:
        df_buildings["block"] = ANDESITE
    # Concatenate with original DataFrame
    df = df.merge(df_buildings.block, how="left", left_index=True, right_index=True)
    df = df.assign(block=df.block_y).drop(columns=["block_x", "block_y"])

    return df


def determine_closest(df, blocks_rgb):
    df = df.copy()
    # Take only rgb
    df_rgb = df[["red", "green", "blue"]]
    # Calculate distances for each block
    dists = np.array([calc_dist(df_rgb, b) for b in tqdm(blocks_rgb)])
    # Choose closest distance for each point
    dists = dists.argmin(axis=0)
    # Insert into block column
    df["block"] = dists

    return df


def calc_dist(rgb, block_rgb):
    diff = rgb - block_rgb
    
    diff.red = diff.red.astype("uint64")
    diff.green = diff.green.astype("uint64")
    diff.blue = diff.blue.astype("uint64")
    
    dist = (diff["red"] ** 2 + diff["green"] ** 2 + diff["blue"] ** 2) ** 1/2
    return dist


def process_plants(df):
    # :))
    df = df.copy()
    print(f"No. plants: {len(df[df.raw_classification.isin({3,4,5})])}")
    # Filter out to vegetation-classified points
    df.loc[df.raw_classification == 3, "block"] = TALLGRASS
    df.loc[df.raw_classification == 4, "block"] = DOUBLE_PLANT
    df.loc[df.raw_classification == 5, "block"] = MOSS_BLOCK

    return df


def process_water(df):
    # :)))
    df = df.copy()
    print(f"No. water: {len(df[df.raw_classification == 8])}")
    # Filter out to water-classified points
    df.loc[df.raw_classification == 8, "block"] = WATER

    return df


def process_ground(df, ground):
    # lol
    df = df.copy()
    print(f"No. ground: {len(df[df.raw_classification == 2])}")
    # Filter out to ground-classified points
    df.loc[df.raw_classification == 2, "block"] = DIRT if ground == "dirt" else STONE # not the best

    return df

def process_noise(df, noise):
    df = df.copy()
    print(f"No. noise before filtering: {len(df[df.raw_classification == 7])}")
    print("NOISE FILTERING " + ("OFF" if noise else "ON"))
    if not noise:
        df = df[df.raw_classification != 7]
    
    return df


def process_rest(df, blocks_rgb, rest, ground, colors):
    df = df.copy()
    print(f"No. rest before filling: {len(df[df.block.isna()])}")
    if rest == "ground":
        print("Rest will be filled with ground")
        df.loc[df.block.isna()] = DIRT if ground == "dirt" else STONE # not the best one
    else: # not the best, again
        df_rest = df[df.block.isna()]
        if colors:
            df_rest = determine_closest(df_rest, blocks_rgb)
        else:
            df_rest["block"] = ANDESITE
        # Join with original DataFrame
        df = df.merge(df_buildings.block, how="left", left_index=True, right_index=True)
        df= df.assign(block=df.block_y).drop(columns=["block_x", "block_y"])

    return df


def aggregate(df):
    df_final = df.groupby(["X", "Y", "Z"])["block"].progress_aggregate(lambda x: pd.Series.mode(x).iloc[0]).reset_index()

    return df_final


def obstructed(p, df_elevs):
    p = np.array(p)
    a = (p[0] - 1, p[1])
    b = (p[0], p[1] + 1)
    c = (p[0] + 1, p[1])
    d = (p[0], p[1] - 1)
    adj = set()
    i = 0
    for x in {a, b, c, d}:
         if df_elevs[(df_elevs.X == x[0]) & (df_elevs.Y == x[1])].empty:
              return False

    return True


def main():
    filename, ground, noise, rest, blocks_rgb, blockmap = initialize()

    # Load data
    df, colors = load_data(filename)
    print("Data loaded into a DataFrame\n")

    # Convert to absolute integer units in meters
    # (default unit = 0.01m)
    df = convert_init(df)

    # Limit to one region
    # TODO: option to select multiple
    df = df[(df["X"] < 512) & (df["Y"] < 512) & (df["Z"] < 256)]

    # Process buildings
    print("Proceeding to determining the blocks for buildings...")
    df = process_buildings(df, blocks_rgb, colors)
    print("Done\n")

    # Process vegetation
    print("For plants...")
    df = process_plants(df)
    print("Done\n")

    # Process water
    print("Water stuff...")
    df = process_water(df)
    print("Done\n")

    # Process ground
    print("Ground stuff...")
    df = process_ground(df, ground)
    print("Done\n")

    # Exclude noise
    print("Noise stuff...")
    df = process_noise(df, noise)
    print("Done\n")

    # Fill the rest
    print("Rest stuff...")
    df = process_rest(df, blocks_rgb, rest, ground, colors)
    df.loc[df.block.isna(), "block"] = 4
    print("Done\n")

    # Summarize
    print("Choosing dominant block for each m2...")
    df_final = aggregate(df)
    print("Done\n")

    # Convert type
    df_final["block"] = df_final["block"].astype(int)

    df_buildings = df_final[df_final.block.isin({0, 1, 2})]

    df_elevs = df_buildings.groupby(["X", "Y"]).Z.max().reset_index()


    # Add glass:))
    # Try to look on it from below, quite crazy
    # But I'll fix it anyway
    obs = df_elevs.progress_apply(lambda x: obstructed(x, df_elevs), axis=1)
    df_elevs = df_elevs.assign(obstructed=obs)
    df_glass = df_elevs[~df_elevs["obstructed"]]
    #df_final.merge(df_elevs, how="left", on=["X", "Y", "Z"])


    # Create region
    region = anvil.EmptyRegion(0, 0)
    # Add blocks
    print("Adding blocks to region...")
    df_final.progress_apply(lambda x: region.set_block(blockmap[x.block], x.X, x.Z, x.Y), axis=1)
    print("Added\n")

    # Add glass blocks
    print("Adding glass blocks to region...")
    for row in df_glass.iterrows():
        omitem = np.array(df_buildings[(df_buildings.X == row[1].X) & (df_buildings.Y == row[1].Y)].Z)
        zs = pd.Series(range(row[1].Z))
        zs = zs[~zs.isin(omitem)]
        for z in zs:
            region.set_block(blockmap[GLASS], row[1].X, z, row[1].Y)
    print("Added\n")

    # Save to file
    print("Saving...")
    df_final.to_parquet("df_final.parquet")
    region.save("r.0.0.mca")
    print("Saved")


if __name__ == "__main__":
    main()

