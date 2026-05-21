import pandas as pd
import numpy as np


def feature_engineering(input_df, verbose=False):

    """
    Create features for machine learning anomaly detection.
    Takes dataframe with the imported Modbus/TCP data 

    If input_df includes the column label, it is kept in the output for evaluation data

    Returns a dataframe with the enginereed features and label if included in the input
    """
    


    #Create copy of original input
    feature_df = input_df.copy()

    #Removes additional spaces in column names
    feature_df.columns = feature_df.columns.str.strip()

    if verbose:
        print("Steg 1 ferdig: Kopiert dataframe")


    #Convert to timestamps
    feature_df["Time_dt"] = pd.to_datetime(
        feature_df["Time"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
        utc=True
    )

    #Remove rows with errors in time conversion
    feature_df = feature_df.dropna(subset=["Time_dt"]).copy()

    #Sort on time
    feature_df = feature_df.sort_values("Time_dt").reset_index(drop=True)

    if verbose:
        print("Steg 2 ferdig: Time konvertert til datetime")


    #Create source_destination pairs
    feature_df["pair"] = (
        feature_df["Source"].astype(str) + " -> " + feature_df["Destination"].astype(str)
    )

    if verbose:
        print("Steg 3 ferdig: Source-Destination pair laget")


    #Convert columns to numerical values
    numeric_cols = [
        "Transaction Identifier",
        "Length.1",
        "Unit Identifier",
        "Function Code",
        "Reference Number",
        "Register Number",
        "Register Value(UNIT16)"
    ]

    for col in numeric_cols:
        if col in feature_df.columns:
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    if verbose:
        print("Steg 4 ferdig: Numeriske kolonner konvertert")


    #Create time based features
    feature_df["Time_since_previous_packet"] = (
        feature_df["Time_dt"]
        .diff()
        .dt.total_seconds()
    )

    feature_df["Time_since_previous_packet_same_pair"] = (
        feature_df
        .groupby("pair")["Time_dt"]
        .diff()
        .dt.total_seconds()
    )

    feature_df["Time_since_previous_packet_source"] = (
        feature_df
        .groupby("Source")["Time_dt"]
        .diff()
        .dt.total_seconds()
    )

    if verbose:
        print("Steg 5 ferdig: Time-based features laget")


    #Create time windows
    feature_df["time_window_1s"] = feature_df["Time_dt"].dt.floor("1s")
    feature_df["time_window_10s"] = feature_df["Time_dt"].dt.floor("10s")

    if verbose:
        print("Steg 6 ferdig: Tidsvinduer laget")


    # 7. packet_count_1s
    feature_df["packet_count_1s"] = (
        feature_df
        .groupby("time_window_1s")["Time_dt"]
        .transform("count")
    )

    if verbose:
        print("Steg 7 ferdig: packet_count_1s laget")



    # 8. packet_count_10s_for_pair
    
    feature_df["packet_count_10s_for_pair"] = (
        feature_df
        .groupby(["time_window_10s", "pair"])["Time_dt"]
        .transform("count")
    )

    if verbose:
        print("Steg 8 ferdig: packet_count_10s_for_pair laget")


   
    # 9. Transaction ID delta for same pair
   

    feature_df["tid_delta_pair"] = (
        feature_df
        .groupby("pair")["Transaction Identifier"]
        .diff()
    )

    if verbose:
        print("Steg 9 ferdig: tid_delta_pair laget")


    # =========================
    # 10. Samme Transaction ID som forrige pakke for samme pair
    # =========================

    feature_df["same_tid_as_previous_pair"] = (
        feature_df["Transaction Identifier"] ==
        feature_df.groupby("pair")["Transaction Identifier"].shift(1)
    ).astype(int)

    if verbose:
        print("Steg 10 ferdig: same_tid_as_previous_pair laget")


    # =========================
    # 11. Samme Function Code som forrige pakke for samme pair
    # =========================

    feature_df["same_function_as_previous_pair"] = (
        feature_df["Function Code"] ==
        feature_df.groupby("pair")["Function Code"].shift(1)
    ).astype(int)

    if verbose:
        print("Steg 11 ferdig: same_function_as_previous_pair laget")


    # =========================
    # 12. Read/write function features
    # =========================

    feature_df["is_read_function"] = (
        feature_df["Function Code"]
        .isin([1, 2, 3, 4])
        .astype(int)
    )

    feature_df["is_write_function"] = (
        feature_df["Function Code"]
        .isin([5, 6, 15, 16])
        .astype(int)
    )

    if verbose:
        print("Steg 12 ferdig: read/write features laget")


    # =========================
    # 13. Missing register value feature
    # =========================

    feature_df["is_missing_register_value"] = (
        feature_df["Register Value (UINT16)"]
        .isna()
        .astype(int)
    )

    if verbose:
        print("Steg 13 ferdig: missing value feature laget")


    # =========================
    # 14. Velg features til ny ML dataframe
    # =========================

    selected_features = [
        "Time_since_previous_packet",
        "Time_since_previous_packet_same_pair",
        "Time_since_previous_packet_source",
        "packet_count_1s",
        "packet_count_10s_for_pair",
        "tid_delta_pair",
        "same_tid_as_previous_pair",
        "same_function_as_previous_pair",
        "Transaction Identifier",
        "Length.1",
        "Unit Identifier",
        "Function Code",
        "Reference Number",
        "Register Number",
        "Register Value (UINT16)",
        "is_read_function",
        "is_write_function",
        "is_missing_register_value"
    ]

    # Legg til Label hvis den finnes
    if "Label" in feature_df.columns:
        selected_features.append("Label")

    # Beholder bare kolonner som faktisk finnes
    selected_features = [
        col for col in selected_features
        if col in feature_df.columns
    ]

    ml_df = feature_df[selected_features].copy()

    if verbose:
        print("Steg 14 ferdig: ML dataframe laget")


    # =========================
    # 15. Håndter NaN
    # =========================

    if "Label" in ml_df.columns:
        label_col = ml_df["Label"].copy()
        ml_df = ml_df.drop(columns=["Label"])
        ml_df = ml_df.fillna(0)
        ml_df["Label"] = label_col.fillna(0).astype(int)
    else:
        ml_df = ml_df.fillna(0)

    if verbose:
        print("Steg 15 ferdig: NaN håndtert")
        print()
        print(ml_df.head())
        print()
        print(ml_df.dtypes)
        print()
        print("Shape:", ml_df.shape)

    return ml_df

def label(X,y, verbose=False):
    #Matches evaluation data and labeling based on time
    #label data(X, y)

    #convets time columnd to a timestamp in X
    X["Time_dt"] = pd.to_datetime(
        X["Time"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
        utc=True
    )

    #convets time columnd to a timestamp in y
    y["Timestamp_dt"] = pd.to_datetime(
        y["Timestamp"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
        utc=True
    )

    #Remove any rows where there were errors with timeconversion
    X = X.dropna(subset=["Time_dt"]).copy()
    y = y.dropna(subset=["Timestamp_dt"]).copy()


    #Find timespan in X
    start_time = X["Time_dt"].min()
    end_time = X["Time_dt"].max()

    if verbose:
        print("X tidsrom:")
        print(start_time, "til", end_time)


    #Remove any labels outside of X's timeframe
    y_filtered = y[
        (y["Timestamp_dt"] >= start_time) &
        (y["Timestamp_dt"] <= end_time)
    ].copy()

    if verbose:
        print("Antall rader i y før filtrering:", len(y))
        print("Antall rader i y etter filtrering:", len(y_filtered))


    #Sort based on time before merging
    X_sorted = X.sort_values("Time_dt").copy()
    y_sorted = y_filtered.sort_values("Timestamp_dt").copy()


   
    #Create "match_found" column 
    y_sorted["match_found"] = 1


    #Match the closest timestamp
    matched_df = pd.merge_asof(
        X_sorted,
        y_sorted[["Timestamp_dt", "match_found"]],
        left_on="Time_dt",
        right_on="Timestamp_dt",
        direction="nearest",
        tolerance=pd.Timedelta(milliseconds=1)
    )


    #Create a label column
    matched_df["Label"] = matched_df["match_found"].fillna(0).astype(int)


    #Drop the match found column
    matched_df = matched_df.drop(columns=["match_found"])


    #Return matched dataframe
    X = matched_df.copy()

    if verbose:
        print(X["Label"].value_counts())
        print(X.head())

    return X

def create_sequences(data, window_size, labels=None):
    #Create sequences of the given data based of the window size
    #If lables is included the sequences is labeled
    #Every sezuence containing at least one anomaly is labeled
    
    sequences = []
    sequence_labels = []

    for i in range(len(data) - window_size + 1):
        sequence = data[i:i + window_size]
        sequences.append(sequence)

        if labels is not None:
            label_window = labels[i:i + window_size]

            sequence_label = int(np.max(label_window))

            sequence_labels.append(sequence_label)

    X_seq = np.array(sequences)

    if labels is not None:
        y_seq = np.array(sequence_labels)
        return X_seq, y_seq

    return X_seq

def import_csv(csv, folder_path):
    #Imports the given csv's from the given path
    #Merges them together to a dataframe and returns one single dataframe
    
    dfs = []
    
    for file in csv:
        file_path = folder_path / file
        print("Reading:", file_path)
        temp = pd.read_csv(file_path)
        dfs.append(temp)

    df = pd.concat(dfs, ignore_index=True)
    return df