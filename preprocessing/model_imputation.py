import pandas as pd

# 1. raw Dataset laden
#df = pd.read_csv('../data/vehicles.csv')  # Pfad zum Dataset anpassen


# 2. Funktion zum Extrahieren der Stellen 4–8 der VIN
def extract_vin_positions(vin):
    if pd.isna(vin) or len(str(vin)) < 8:
        return None
    return str(vin)[3:8]  # Stellen 4–8 (Index 3 bis 7)


# Change from Richard to enable use of split data sets
def impute_model(df):

    
    # 3. Daten bereinigen: Nur Zeilen mit VIN und Model behalten
    df_vin_model = df[['VIN', 'model', 'manufacturer', 'year']].dropna(subset=['VIN', 'model'])

    # 4. VIN-Stellen 4–8 extrahieren
    df_vin_model['vin_vds'] = df_vin_model['VIN'].apply(extract_vin_positions)

    # 5. Zuordnung erstellen: VIN VDS (Stellen 4–8) zu häufigstem Modell
    vin_vds_model_mapping = df_vin_model.groupby('vin_vds')['model'].agg(lambda x: x.mode()[0] if not x.empty else None).to_dict()

    # 6. Imputation durchführen
    # Neue Spalte für imputierte Modelle
    df['model_imputed'] = df['model']

    # Für Zeilen, wo 'model' fehlt, aber 'VIN' vorhanden ist
    mask = df['model'].isna() & df['VIN'].notna()

    # VIN-Stellen 4–8 für Imputation extrahieren
    df['vin_vds'] = df['VIN'].apply(extract_vin_positions)

    # Imputation mit VIN VDS (Stellen 4–8)
    df.loc[mask, 'model_imputed'] = df.loc[mask, 'vin_vds'].map(vin_vds_model_mapping)

    # 7. Ergebnisse analysieren
    total_rows = len(df)
    missing_models = df['model'].isna().sum()
    imputed_models = df['model_imputed'].notna().sum() - df['model'].notna().sum()

    print(f"Anzahl der Zeilen im Dataset: {total_rows}")
    print(f"Anzahl der Zeilen mit fehlendem 'model': {missing_models}")
    print(f"Anzahl der imputierten Modelle: {imputed_models}")

    # 8. Beispiel der Zuordnung anzeigen
    print("\nBeispiel für VIN VDS (Stellen 4–8) zu Modell-Zuordnung:")
    print(list(vin_vds_model_mapping.items())[:5])

    # 9. Ausgabe von 30 Zeilen, in denen das Modell imputiert wurde
    print("\n30 Zeilen mit imputierten Modellen (manufacturer, VIN, model_imputed):")
    imputed_rows = df[mask & df['model_imputed'].notna()][['manufacturer', 'VIN', 'model_imputed']].head(60)
    print(imputed_rows.to_string(index=False))

    #Restlichen NaN-Werte in der Hersteller-Spalte entfernen
    df = df.dropna(subset=['model'])

    # return transformed column
    return df[["id","model_imputed"]]

