def map_mates_to_bounds(row, clip):
    """When is_mate == 1 the eval column denote moves to mate"""
    if row["is_mate"] == 0:
        return row["eval"]
    elif row["eval"] > 0:
        return clip
    else:
        return -clip


def prepare_chess_frame(df, normalize=True, clip=150):
    df["eval"] = df.apply(lambda row: map_mates_to_bounds(row, clip), axis=1)
    df["eval"] = df["eval"].clip(lower=-clip, upper=clip)
    if normalize:
        df["eval"] = df["eval"] / clip
    return df