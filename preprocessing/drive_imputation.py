import pandas as pd

# Change from Richard to enable use of split data sets
def impute_drive(df):
    """imputes drive by selecting the most common type label for a given drive label. Returns the transformed drive column."""

    # Verteilung von `drive` pro `type` vor Imputation anzeigen
    print("Verteilung von `drive` pro `type` vor Imputation:")
    print(df.groupby(["type", "drive"]).size().unstack(fill_value=0))

    # Verteilung von `drive` pro `type` berechnen
    drive_distribution = df.groupby(["type", "drive"]).size().unstack(fill_value=0)

    # Modus und prozentualen Anteil berechnen
    mode_info = {}
    for type_name in drive_distribution.index:
        # Modus (häufigster drive-Wert) für diesen type
        mode_drive = drive_distribution.loc[type_name].idxmax()
        # Gesamtanzahl der Einträge für diesen type
        total_count = drive_distribution.loc[type_name].sum()
        # Anzahl des Modus
        mode_count = drive_distribution.loc[type_name, mode_drive]
        # Prozentualer Anteil des Modus
        mode_percentage = (mode_count / total_count * 100) if total_count > 0 else 0
        # Speichern der Informationen
        mode_info[type_name] = {"mode": mode_drive, "percentage": mode_percentage}

    # Modus von `drive` pro `type` für die Imputation
    drive_mode_per_type = {type_name: info["mode"] for type_name, info in mode_info.items()}

    # Fehlende Werte in `drive` imputieren
    df["drive_imputed"] = df.apply(
        lambda row: drive_mode_per_type[row["type"]] if pd.isna(row["drive"]) and row["type"] in drive_mode_per_type else row["drive"],
        axis=1
    )

    

    # Ergebnisse speichern
    #df.to_csv("vehicles_imputed.csv", index=False)

    # Verteilung nach Imputation anzeigen
    print("\nVerteilung von `drive` pro `type` nach Imputation:")
    print(df.groupby(["type", "drive"]).size().unstack(fill_value=0))

    # Modus und prozentualen Anteil ausgeben
    print("\nModus und prozentualer Anteil pro `type` (vor Imputation):")
    for type_name, info in mode_info.items():
        print(f"{type_name}: Modus = {info['mode']}, Anteil = {info['percentage']:.1f}%")
    
    return df[["id","drive_imputed"]]


