from pathlib import Path

import pandas as pd

def summarize_csv(csv_filename):
    """
    Summarize the columns of the csv.
    Save the summary as <csv_filename>.summary.txt
    :param csv_filename:
    :return: Path to summary file
    """
    input_path = Path(csv_filename)
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(input_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    lines = [
        f"File: {input_path}",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        "",
    ]

    for column in df.columns:
        series = df[column]
        non_null = int(series.notna().sum())
        null_count = int(series.isna().sum())
        unique_non_null = int(series.nunique(dropna=True))

        lines.append(f"Column: {column}")
        lines.append(f"  dtype: {series.dtype}")
        lines.append(f"  non_null: {non_null}")
        lines.append(f"  null: {null_count}")
        lines.append(f"  unique_non_null: {unique_non_null}")

        sample_values = series.dropna().unique().astype(str)[:5].tolist()
        lines.append(
            "  sample_values: "
            + (", ".join(sample_values) if sample_values else "<none>")
        )

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                lines.append(f"  min: {numeric.min()}")
                lines.append(f"  max: {numeric.max()}")
                lines.append(f"  mean: {numeric.mean()}")

        lines.append("")

    summary_path = input_path.with_name(f"{input_path.name}.summary.txt")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path

def collectContacts():
    """

    """
    
    dfpub = pd.read_csv("OSPI_publicschools/Washington_District_Directory_20251019.csv")
    dfpub = dfpub[[
    "AddressLine1",
    "AddressLine2",
    "City",
    "State",
    "Zipcode",
    "AdministratorName",
    "Phone",
    "Email",
    "OrgCategoryList",
    "GradeCategory"
    ]]

    dfproviders = pd.read_csv("childcare_providers/DCYF_Licensed_Childcare_Center_and_School_Age_Program_Providers.csv")
    dfproviders = dfproviders[[
        "ProviderName",
        "DoingBusinessAs",
        "FacilityTypeGeneric",
        "StartingAge",
        "EndingAge",
        "PrimaryContactPersonName",
        "PrimaryContactPhoneNumber",
        "PrimaryContactEmail",
        "GeoCodedPhysicalAddress",
        "PhysicalStreetAddress",
        "PhysicalCity",
        "PhysicalState",
        "PhysicalZip",
        "PhysicalCounty",
        "ProviderShippingAddress",
        "ProviderShippingCity",
        "ShippingZip",
        "ProviderShippingState",
        ]]
    #
    # dfprivate = pd.read_sas("WA_privateschools/nces/pss2122_pu_sas/Create PU 2021-22 PSS SAS Dataset.sas",
    #                         format="sas7bdat")
    # dfprivate = pd.read_spss("WA_privateschools/nces/pss2122_pu.sav")
    # # dfprivate = dfprivate[[
    #
    # dfprivate = pd.read_excel("WA_privateschools/2025-26 Approved Private Schools.xlsx")
    # # dfprivate = dfprivate[[
    """
    Column: Approval Date
    Column: School Name
    Column: Street Address
    Column: City
    Column: State
    Column: ZIP
    Column: County
    Column: School District
    Column: Profit
    Column: Online Option
    Column: Lasst Updated 11/7/2025
    """

if __name__ == '__main__':
    files = [
        "childcare_providers/DCYF_Licensed_Childcare_Center_and_School_Age_Program_Providers.csv",
        "OSPI_publicschools/Report_Card_Enrollment_2025-26_School_Year_-_Preliminary_20260122.csv",
        "OSPI_publicschools/Washington_District_Directory_20251019.csv",
        "OSPI_publicschools/Washington_School_Directory_20251019.csv",
        "WA_privateschools/2025-26 Approved Private Schools.xlsx",
        "WA_privateschools/2024-2025 Private School Enrollment Data (Amended March 2025)_0.xlsx",
        "WA_privateschools/2025-26-ood-private-school-list.xlsx"
    ]
    for f in files:
        summarize_csv(f)
    pass