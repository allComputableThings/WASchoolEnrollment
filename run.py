import pandas as pd

import schools_directory
from dataUtil import publicData, privateData, noOp
from dataUtil import DataSet
import re

from schools_directory import ZIP, SCHOOL_NAME, SCHOOL_CODE, DISTRICT_NAME


# import openpyxl

def makePlots(public, private=None, schoolDetail=False, addDir=""):
    # public_district_names = sorted(public.df['DistrictName'].unique())
    # public_school_names = sorted(public.df[SCHOOL_NAME].unique())
    # public_school_zips = clean_unique_zips(public.df[ZIP])
    # private_schools_zips = clean_unique_zips(private.df[ZIP])
    #
    # private_district_names = sorted(private.df['DistrictName'].unique())
    # private_school_names = sorted(private.df[SCHOOL_NAME].unique)
    # public_school_zips = set(public_school_zips)
    # private_schools_zips = set(private_schools_zips)

    # combined_zips = sorted(set(public_school_zips)|set(private_schools_zips))
    # print("Combined zips:", combined_zips)
    # print("Zips only in public:", public_school_zips - private_schools_zips)
    # print("Zips only in private:", private_schools_zips - public_school_zips)
    #
    # print("Public district names:", sorted(public_district_names))
    # print("Public schools:", sorted(public_school_names))
    # print("Public zips:", sorted(public_school_zips))
    #
    # print("Public district names:", sorted(private_district_names))
    # print("Private schools:", sorted(private_school_names))
    # print("Private zips:", sorted(private_schools_zips))

    sets = []
    if private is not None:
        assert len(private.df) > 10
        sets.append(private)

    if public is not None:
        assert len(public.df) > 10
        sets.append(public)

    for ds in sets:
        ds.plotTotalEnrollment()
        ds.plotGradeTierEnrollment()

    if private is not None:
        a = public.select_since(2019, False)
        b = private.select_since(2019, False)
        savedir = b.filename(addDir=addDir) + f"/vs_public"
        DataSet.plotStack(
            a.totalsByYear,
            b.totalsByYear,
            ylabel="Enrollment",
            savedir=savedir
        )

        for grade in ["Elementary", "Middle", "High"]:
            a = public.select_since(2019, False).select_grades(grade)
            b = private.select_since(2019, False).select_grades(grade)
            DataSet.plotStack(
                a.totalsByYear,
                b.totalsByYear,
                ylabel="Enrollment",
                savedir=savedir
            )


    # Cohort for all
    for dfBase in sets:
        dfBase.plotCohortProgression(post=f"{'/'if addDir else ''}{addDir}/cohorts")

    # Plot schools
    for ds in sets:
        for name, gradeYears in gradeSets:
            ds.select_grades(name, name=name).plotSchools(addDir=f"{addDir}school_enrollment_trends")

    if schoolDetail:
        # Cohort for school
        for ds in sets:
            for dfBase in [public.select_school(s) for s in ds.schools]:
                dfBase.plotCohortProgression(addDir=f"{addDir}school_cohorts")

    for ds in sets:
        for name, gradeYears in gradeSets:
            ds.plotGrades(name, title=name, addDir=addDir)

    for ds in sets:
        ds.reportYearlyChangesSinceBaselineYear(2019)





gradeSets = [
    ("Elementary", ['K'] + [str(s) for s in range(1, 6)]),
    ("Middle", [str(s) for s in range(6, 9)]),
    ("High", [str(s) for s in range(9, 13)]),
    ("K-12", ['K'] + [str(s) for s in range(1, 13)]),
    ("P-12", ['P', 'K'] + [str(s) for s in range(1, 13)]),
    ("P-5", ['P', 'K'] + [str(s) for s in range(1, 6)]),
    ("K-5", ['K'] + [str(s) for s in range(1, 6)]),
]


def pub_district2priv_zips(dfpub, district=None):
    dfpub = dfpub.df.set_index(SCHOOL_CODE)
    pub_district = set(dfpub[DISTRICT_NAME].unique())
    assert len(pub_district) == 1
    pub_district = next(iter(pub_district))
    dfpub_directory = schools_directory.public.select_district(pub_district).df.set_index(SCHOOL_CODE)
    pub_zips = dfpub_directory[dfpub_directory.index.isin(dfpub.index)][ZIP].unique()
    if district:
        priv_zips = schools_directory.private.select_district(district).df[ZIP].unique()
    else:
        priv_zips = []

    zips = set(priv_zips) | set(pub_zips)
    return zips


def run():

    public = publicData()
    private = privateData()

    # public_city = schools_directory.public.select_zip(zips).df["City"].unique()
    # private_city = schools_directory.private.select_zip(zips).df["City"].unique()
    # import pandas as pd
    # with pd.option_context('display.max_rows', None,
    #                        'display.max_columns', None,
    #                        'display.width', None,
    #                        'display.max_colwidth', None):
    #     schools_directory.public.select_district(district_name)
    #     private.df.tail(10)


    for school_district, title, private_rule in [
        ("Renton.*", "Renton", {"city": "Renton.*"}),
        ("Mercer Island School District", "Mercer Island", {"city": "Mercer Island"}),
        ("Lake Washington School District", "Lake Washington", {}),
        ("Issaquah School District", "Issaquah", {"city": "Issaquah"}),
        ("Bellevue School District", "Bellevue",  {"city": "Bellevue"}),
        ("Northshore School District", "Northshore", {"district": ".*Northshore.*"}),
        ("Seattle.*", "Seattle", {"city": "Seattle.*"}),
    ]:
        dfpub = public.select_region(school_district, title=title)
        if private_rule:
            if 'city' in private_rule:
                dfpriv = private.select_city(private_rule['city'])
            elif 'district' in private_rule:
                dfpriv = private.select_zips(pub_district2priv_zips(dfpub, district=private_rule['district']),
                                             title=title)
            else:
                raise Exception(f"Bad select private school select rule: {str(private_rule)}")
        else:
            dfpriv = private.select_zips(pub_district2priv_zips(dfpub), title=title)

        makePlots(dfpub, dfpriv, schoolDetail=True)

    # King county (public only)
    for county in ["King"]:
        makePlots(public.select_county(county), private=None, schoolDetail=False)

    # State wide
    makePlots(public.addPath("State"), private.addPath("State"))




if __name__ == "__main__":
    display = noOp #Jupyter
    # pltShow = noOp

    run()