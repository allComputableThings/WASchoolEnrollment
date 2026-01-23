import functools
import pandas as pd
from jinja2.filters import do_capitalize

from schools_directory import SCHOOL_NAME, DISTRICT_NAME
from util import fixCaps


# assert directory.schoolname_city_to_zip("Yakima Adventist Christian School","Yakima") == 98908
# directory.schoolname_city2zip("Yakima Adventist Christian School","Yakima")


@functools.cache
def publicData() -> pd.DataFrame :
    """
    :return: A DataDrame with columns: 'Year', SCHOOL_NAME, DISTRICT_NAME, "Grade", "Total", "County"
    """
    # 2022-23 For Bellevue only. One school at a time. *Sigh*
    # https://washingtonstatereportcard.ospi.k12.wa.us/ReportCard/ViewSchoolOrDistrict/101387

    # CAVEAT -- the 2022 disagree with
    # https://washingtonstatereportcard.ospi.k12.wa.us/ReportCard/ViewSchoolOrDistrict/100019
    # The figures here show:
    # 22-23: 19354
    # def bellevue2023():
    #     def gen():
    #         for filename in glob.glob(f"{publicDataRoot}/OSPI_publicschools/2022_Bellevue/*.csv"):
    #             df = pd.read_csv(filename)
    #             df = df.rename(
    #                 columns={'Organization Name': 'SchoolName', 'School Year': "Year", 'Gradelevel': 'GradeLevel',
    #                          'Number Of Students': 'All Students'})
    #             df['DistrictName'] = 'Bellevue'
    #             # print(df[SCHOOL_NAME].unique(), filename)
    #             yield df
    #
    #     return pd.concat(gen())

    # Public school data: https://www.k12.wa.us/data-reporting/data-portal
    year2Pubdata = {
        2014: pd.read_csv('OSPI_publicschools/Report_Card_Enrollment_2014-15_School_Year.csv'),
        2015: pd.read_csv('OSPI_publicschools/Report_Card_Enrollment_2015-16_School_Year.csv'),
        2016: pd.read_csv('OSPI_publicschools/Report_Card_Enrollment_2016-17_School_Year.csv'),
        2017: pd.read_csv('OSPI_publicschools/Report_Card_Enrollment_2017-18_School_Year.csv'),
        2018: pd.read_csv('OSPI_publicschools/Report_Card_Enrollment_2018-19_School_Year.csv'),
        2019: pd.read_csv('OSPI_publicschools/Report_Card_Enrollment_2019-20_School_Year.csv'),
        2020: pd.read_csv('OSPI_publicschools/Report_Card_Enrollment_2020-21_School_Year.csv'),
        2021: pd.read_csv('OSPI_publicschools/Report_Card_Enrollment_2021-22_School_Year.csv'),
        2022: pd.read_csv('OSPI_publicschools/Report_Card_Enrollment_2022-23_School_Year.csv'),
        #   2023: pd.read_csv('OSPI_publicschools/Report_Card_Enrollment_2023-24_School_Year_20240130.csv'),
        2023: pd.read_csv('OSPI_publicschools/Report_Card_Enrollment_2023-24_School_Year_20240312.csv'),
        # 2024: pd.read_csv('OSPI_publicschools/Report_Card_Enrollment_2024-25_School_Year_-_Preliminary_20250203.csv'),
        2024: pd.read_csv('OSPI_publicschools/Report_Card_Enrollment_2024-25_School_Year_20250717.csv'),
        #   2022: bellevue2023()
        2025: pd.read_csv("OSPI_publicschools/Report_Card_Enrollment_2025-26_School_Year_-_Preliminary_20260122.csv")
    }

    # Normalize data - just want ['Year', SCHOOL_NAME, "Region", "Grade", "Total"]
    # This format is common the this collections of workbooks
    years = list(year2Pubdata.keys())
    datacols = ['Year', SCHOOL_NAME, DISTRICT_NAME, "Grade", "Total", "County", "SchoolCode"]
    grades = ['K', 'P'] + list(range(1, 13))  # NB. range in Python is exclusive of last values\
    gradeMap = {  # For standardize grade categories
        'Pre-Kindergarten': 'P',
        'Transition to Kindergarten': 'P',
        'Kindergarten': 'K',
        'Half-day Kindergarten': 'K',  # ?
        'Half-Day Kindergarten': 'K',  # ?
        '1st Grade': '1',
        '2nd Grade': '2',
        '3rd Grade': '3',
        '4th Grade': '4',
        '5th Grade': '5',
        '6th Grade': '6',
        '7th Grade': '7',
        '8th Grade': '8',
        '9th Grade': '9',
        '10th Grade': '10',
        '11th Grade': '11',
        '12th Grade': '12',
        'Total Pre-K': 'P',
        'KG': 'K',
        'Grade 1': '1',
        'Grade 2': '2',
        'Grade 3': '3',
        'Grade 4': '4',
        'Grade 5': '5',
        'Grade 6': '6',
        'Grade 7': '7',
        'Grade 8': '8',
        'Grade 9': '9',
        'Grade 10': '10',
        'Grade 11': '11',
        'Grade 12': '12',
    }

    # Cleanup each CSV file
    def mapData(year, df):
        df = df.copy()
        df = df.rename(columns={'School Name': SCHOOL_NAME,
                                'DistrictName': DISTRICT_NAME,
                                'All Students': 'Total'})
        df['Year'] = year
        df = df[df['GradeLevel'] != 'AllGrades']
        df = df[df['GradeLevel'] != 'All Grades']
        df['Grade'] = df['GradeLevel'].replace(gradeMap)
        df = df[df[SCHOOL_NAME] != 'State Total']
        df = df[df[SCHOOL_NAME] != 'District Total']
        df = df[(df[SCHOOL_NAME] != '') & ~(df[SCHOOL_NAME].isna())]
        def to_int(x):
            if isinstance(x, int):
                return x
            if isinstance(x, str):
                if ',' in x:
                    return int(x.replace(',', ''))
            return int(x)
        df['Total'] = df['Total'].apply(to_int)
        # df['Total'] = df['Total'].astype(int)

        return df[datacols + list(set(df.columns) - set(datacols))]

    print(years)
    year2PubdataNormalized = {year: mapData(year, df) for (year, df) in year2Pubdata.items()}
    pubdataNormalized = pd.concat(year2PubdataNormalized.values()).reindex()
    pubdataNormalized[DISTRICT_NAME] = (pubdataNormalized[DISTRICT_NAME]
                                   .str.strip()
                                   .str.replace("Woononville", "Woodinville"))

    pubdataNormalized.loc[pubdataNormalized["SchoolCode"].isna(), "SchoolCode"] = -1
    pubdataNormalized["SchoolCode"] = pubdataNormalized["SchoolCode"].astype(int)
    print(f"Columns in:  {pd.concat(year2Pubdata.values()).columns}")
    print(f"Columns out: {pubdataNormalized.columns}")

    # Santity checks
    print(pubdataNormalized.columns)
    print("Grade", sorted(pubdataNormalized["Grade"].unique()))
    print("Year", sorted(pubdataNormalized["Year"].unique()))
    print("Counties:", pubdataNormalized["County"].unique())
    print("County", pubdataNormalized.index.size, pubdataNormalized["County"].isna().size)
    # display(pubdataNormalized[pubdataNormalized["County"].isna()])
    # dAll = pubdataNormalized.copy()
    print("Total WA public enrollment by year")

    # display(dAll.groupby("Year")["Total"].sum())
    return pubdataNormalized


@functools.cache
def privateData() -> pd.DataFrame :
    gradeMap = {'Total PreK': 'P',
                'Total KG': 'K',
                'Total G1': '1',
                'Total G2': '2',
                'Total G3': '3',
                'Total G4': '4',
                'Total G5': '5',
                'Total G6': '6',
                'Total G7': '7',
                'Total G8': '8',
                'Total G9': '9',
                'Total G10': '10',
                'Total G11': '11',
                'Total G12': '12',
                'PK Total': 'P',
                #        'PK Total.1',
                'KG Total': 'K',
                'G1 Total': '1',
                'G2 Total': '2',
                'G3 Total': '3',
                'G4 Total': '4',
                'G5 Total': '5',
                'G6 Total': '6',
                'G7 Total': '7',
                'G8 Total': '8',
                'G9 Total': '9',
                'G10 Total': '10',
                'G11 Total': '11',
                'G12 Total': '12',

                'PreK TOTAL': 'P',
                'KG TOTAL': 'K',
                'G1 TOTAL': '1',
                'G2 TOTAL': '2',
                'G3 TOTAL': '3',
                'G4 TOTAL': '4',
                'G5 TOTAL': '5',
                'G6 TOTAL': '6',
                'G7 TOTAL': '7',
                'G8 TOTAL': '8',
                'G9 TOTAL': '9',
                'G10 TOTAL': '10',
                'GI0 TOTAL': '10',
                'G11 TOTAL': '11',
                'G12 TOTAL': '12',

                'PreK': 'P',
                'Total Pre-K': 'P',
                'KG': 'K',
                'Grade 1': '1',
                'Grade2': '2',
                'Grade 2': '2',
                'Grade 3': '3',
                'Grade 4': '4',
                'Grade 5': '5',
                'Grade 6': '6',
                'Grade 7': '7',
                'Grade 8': '8',
                'Grade 9': '9',
                'Grade 10': '10',
                'Grade 11': '11',
                'Grade 12': '12',
                1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 11: '11', 12: '12'}

    # Final columns, except Grade, Total
    datacols = [
        SCHOOL_NAME,
        # Some are "District Name", some are "City"
        # Will use to hold both and search for substrings: "Bellevue"
        'Street Address', # Missing in some
        "City",
        # 'State',
        'Zipcode', #Missing in 2018
        # 'County',
        'Year',  # Added
    ]
    allFinalCols = datacols + ["Grade", "Total"]
    # Final grade categories
    grades = ["P", "K"] + [str(g) for g in range(1, 13)]

    import pandas as pd
    pd.set_option('display.max_columns', None)  # set the max columns to display to none
    # Private data source: https://www.sbe.wa.gov/our-work/private-schools
    d2018 = pd.read_excel('WA_privateschools/Final Excel P105B Combined Data for 201819.xlsx')
    d2019 = pd.read_excel('WA_privateschools/Copy of 2019-20 Private School Enrollment_073120.xlsx', sheet_name=1,
                          skiprows=[0, 1])
    # d2020 = pd.read_excel('WA_privateschools/Copy of 2020-21 Private School Enrollment x Grade_for the website.xlsx')
    d2020 = pd.read_excel(
        'WA_privateschools/Fixed - Copy of 2020-21 Private School Enrollment x Grade_for the website.xlsx')
    d2021 = pd.read_excel('WA_privateschools/2021-22 Private School Enrollment (Website).xlsx')
    d2022 = pd.read_excel('WA_privateschools/2022 - 2023 Private School Enrollment (website).xlsx')
    d2023 = pd.read_excel('WA_privateschools/2023 - 2024 Private School Enrollment (website).xlsx')
    # d2024 = pd.read_excel('WA_privateschools/2024-2025 Private School Enrollment Data.xlsx')
    d2024 = pd.read_excel('WA_privateschools/2024-2025 Private School Enrollment Data (Amended March 2025)_0.xlsx')

    def d2018normalized():
        # Normalize 2018 --- Not used. Format has quite a lot difference. Suspicious of the changes
        # d2018_ = d2018.rename(columns={"District Name": DISTRICT_NAME}) # "City" is missing in this dataset. We only have "District Name"
        # d2018_['Grade'] = d2018['Grade'].replace(gradeMap)
        # d2018_['Year'] = 2018
        # d2018_ = d2018_[d2018_[SCHOOL_NAME].notnull()]
        # print(d2018_.shape, list(d2018_[allFinalCols].columns))
        pass

    def d2019normalized():
        # Normalize 2019
        df = d2019.rename(columns=gradeMap).rename(columns={"School Name": SCHOOL_NAME})
        df = df.rename(columns={"Total": "_Total", "Name of School": SCHOOL_NAME,
                                # "City": DISTRICT_NAME
                                })
        # print(sorted(df.columns))
        df = df.melt(id_vars=set(df.columns) - set(grades), value_vars=grades, var_name="Grade", value_name="Total")
        df['Year'] = 2019
        df = df[df[SCHOOL_NAME].notnull()]
        print(df.shape, list(df[allFinalCols].columns))
        checkNulls(df)
        return df

    def d2020normalized():
        # Normalize 2020
        df = d2020.rename(columns=gradeMap).rename(columns={"School Name": SCHOOL_NAME})
        print(df.shape, list(df.columns))
        df[grades]  # The grade columns should all exist
        df = df.melt(id_vars=set(df.columns) - set(grades), value_vars=grades, var_name="Grade", value_name="Total")
        df['Grade'] = df['Grade'].replace(gradeMap)
        df['Year'] = 2020
        df = df[df['City'].notnull()]
        checkNulls(df)
        # display(df[df[DISTRICT_NAME].isnull()])
        # display(d2020["City"].isnull().sum())
        print(f'null cities: {d2020["City"].isnull().sum()}')
        # print(f'Schools with null cities:')
        # for school in d2020.loc[d2020["City"].isnull(),SCHOOL_NAME].unique():
        #     display(d2020[d2020[SCHOOL_NAME]==school])
        # print(f'Schools with null totals:')
        # for school in df.loc[df["Total"].isnull(),SCHOOL_NAME].unique():
        #     display(d2020[d2020[SCHOOL_NAME]==school])

        # display(df[df["Total"].isnull()])
        # display(df[df[DISTRICT_NAME].isnull() & (df[SCHOOL_NAME]=="Carden Country School")])
        # display(df[df[SCHOOL_NAME]=="Carden Country School"])
        return df

    # def d2021normalized():
    #     # Normalize 2020
    #     df = d2020.rename(columns=gradeMap).rename(columns={"City": DISTRICT_NAME})
    #     print(df.shape, list(df.columns))
    #     df[grades]  # The grade columns should all exist
    #     df = df.melt(id_vars=set(df.columns) - set(grades), value_vars=grades, var_name="Grade", value_name="Total")
    #     df['Grade'] = df['Grade'].replace(gradeMap)
    #     df['Year'] = 2021
    #     df = df[df[SCHOOL_NAME].notnull()]
    #     checkNulls(df)
    #     # display(df[df[DISTRICT_NAME].isnull()])
    #     # display(d2020["City"].isnull().sum())
    #     print(f'null cities: {d2020["City"].isnull().sum()}')
    #     # print(f'Schools with null cities:')
    #     # for school in d2020.loc[d2020["City"].isnull(),SCHOOL_NAME].unique():
    #     #     display(d2020[d2020[SCHOOL_NAME]==school])
    #     # print(f'Schools with null totals:')
    #     # for school in df.loc[df["Total"].isnull(),SCHOOL_NAME].unique():
    #     #     display(d2020[d2020[SCHOOL_NAME]==school])
    #
    #     # display(df[df["Total"].isnull()])
    #     # display(df[df[DISTRICT_NAME].isnull() & (df[SCHOOL_NAME]=="Carden Country School")])
    #     # display(df[df[SCHOOL_NAME]=="Carden Country School"])
    #     return df

    def checkNulls(df):
        schools = set()
        for col in ['Grade', 'Year', 'Total',
                    SCHOOL_NAME,
                    'City',
                    #                 'Street Address'
                    ]:
            mask = (df[col].isna() | df[col].isnull())
            if mask.sum():
                print(f"{col} has {len(mask)} nulls")
                schools.intersection(df.loc[mask, SCHOOL_NAME].unique())

        for sname in schools:
            display(df[df[SCHOOL_NAME] == sname])

    # print(d2021.columns)
    def d2021normalized():
        df = d2021.rename(columns=gradeMap).rename(
            columns={"Name of School": SCHOOL_NAME,
                     'ZIP': "Zipcode",
                     "School Name": SCHOOL_NAME,
                     # "City": DISTRICT_NAME
                     })
        df[grades]  # The grade columns should all exist
        df = df.melt(id_vars=set(df.columns) - set(grades), value_vars=grades, var_name="Grade", value_name="Total")
        df['Grade'] = df['Grade'].replace(gradeMap)
        df['Year'] = 2021
        df.loc[df["Total"].isna(), "Total"] = 0
        df["Total"] = df["Total"].astype(int)

        df = df[df[SCHOOL_NAME].notnull()]
        checkNulls(df)
        print(df.shape, list(df[allFinalCols].columns))
        return df

        # return _normalized2021(d2021)

    # Normalize 2022
    def _normalize(df, year):
        df = df.rename(columns=gradeMap).rename(
            columns={'ZIP': "Zipcode",
                     # "City": DISTRICT_NAME,  # District
                     "School Name": SCHOOL_NAME,
                     "Address": "Street Address"})
        print(df.columns)
        df[grades]  # The grade columns should all exist
        df = df.melt(id_vars=set(df.columns) - set(grades), value_vars=grades, var_name="Grade", value_name="Total")
        df['Grade'] = df['Grade'].replace(gradeMap)
        df['Year'] = year
        df.loc[df["Total"].isna(), "Total"] = 0
        df["Total"] = df["Total"].astype(int)
        df = df[df[SCHOOL_NAME].notnull()]

        checkNulls(df)
        #     print(df.shape, list(df[allFinalCols].columns))
        print(df.shape, list(df.columns))
        return df

    # Index([SCHOOL_NAME, 'Street Address', DISTRICT_NAME, 'State', 'Zipcode', 'P',
    #        'PK Total.1', 'K', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    #        '11', '12', 'K12 Total Reported'],
    #       dtype='object')
    def d2022normalized():
        return _normalize(d2022, year=2022)

    # Index([SCHOOL_NAME, 'Street Address', DISTRICT_NAME, 'State', 'Zipcode', 'County',
    #        'District', 'PreK TOTAL', 'KG TOTAL', 'G1 TOTAL', 'G2 TOTAL',
    #        'G3 TOTAL', 'G4 TOTAL', 'G5 TOTAL', 'G6 TOTAL', 'G7 TOTAL', 'G8 TOTAL',
    #        'G9 TOTAL', 'GI0 TOTAL', 'G11 TOTAL', 'G12 TOTAL'],
    #       dtype='object')
    def d2023normalized():
        return _normalize(d2023, year=2023)

    def d2024normalized():
        return _normalize(d2024, year=2024)

    # 2018 seems to have some very different reporting. Not sure it is consistent. Dropped here
    dAll = pd.concat(
        [d2019normalized(),
         d2020normalized(),
         d2021normalized(),
         d2022normalized(),
         d2023normalized(),
         d2024normalized()
         ]).reindex()
    # dAll = pd.concat([d2022_, ]).reindex()
    # Place the important data in the first columns.
    dAll = dAll[allFinalCols + list(set(dAll.columns) - set(allFinalCols))].sort_values(
        ["Year", "Grade", SCHOOL_NAME])

    dAll = dAll[dAll[SCHOOL_NAME].map(lambda s: isinstance(s, str))]  # Zap rows without string school names
    dAll = dAll[~(dAll[SCHOOL_NAME].str.lower().str.contains("state total"))]  # Zap state total rows
    dAll.loc[dAll["Total"].isna(), "Total"] = 0
    dAll = dAll.astype({"Total": int})
    dAll = dAll.astype({"Year": int})
    dAll = dAll[dAll["City"].map(lambda s: isinstance(s, str))]  # Zap rows without school names

    city = dAll["City"]
    city = city.replace({
        "Woononville": "Woodinville",
        "Issaqua": "Issaquah",
        "Woodonville": "Woodinville",
        "Bellvue": "Bellevue"
    })
    # Capitalize city name. (Fixes uppercase)
    dAll["City"] = fixCaps(city)

    # Capitalize school name. (Fixes uppercase)
    dAll[SCHOOL_NAME] = fixCaps(dAll[SCHOOL_NAME])

    dAll = dAll.reset_index()


    # display(dAll[dAll["Street Address"].isna()])
    # dAll = normalizeAddresses(dAll)
    # dAll = fixDuplicateAddress(dAll)

    # There are a number of school in the data without a district or city, or a district of 0 for some reason.
    # We'll just ignore them from bellevue processing since none are in Bellevue
    # dAll = dAll[dAll["City"].astype(str)!='0'] # 0 is not a region
    print(f"Years reporting {dAll['Year'].unique()}")

    print("Schools reporting their city/district as 0")
    # display(dAll[dAll[DISTRICT_NAME].isnull()][SCHOOL_NAME].unique())

    print("\nTotal WA private enrollment by year")
    # display(dAll.groupby("Year")["Total"].sum())
    # d = dAll[[SCHOOL_NAME, "Street Address"]]
    # d[d["Street Address"].isna()]
    return dAll
