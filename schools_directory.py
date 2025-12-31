import dataclasses
import functools
import os
import re

import pandas as pd

from util import fixCaps

publicDataRoot = os.path.join(os.path.split(__file__)[0], "OSPI_publicschools/")
privateDataRoot = os.path.join(os.path.split(__file__)[0], "WA_privateschools/")

ZIP = "Zipcode"
SCHOOL_NAME = "SchoolName"
SCHOOL_CODE = "SchoolCode"
DISTRICT_NAME ="DistrictName"

@dataclasses.dataclass
class SchoolsDirectory:
    _df: pd.DataFrame | None = None

    @functools.cached_property
    def by_schoolname_city(self):
        df = self.private.set_index(["School Name", "City"])
        return df

    def schoolname_city_to_zip(self, school_name, city) -> str:
        zips = self.by_schoolname_city.loc[school_name, city][ZIP]
        assert len(zips) == 1
        return zips.iloc.values

    # @functools.cached_property
    # def combined_school_name_city_zip(self):
    #     pub = self.public[["SchoolName", "City", ZIP]]
    #     priv = self.private[["SchoolName", "City", ZIP]]
    #     return pd.concat(pub, priv)

    def select_district(self, str_or_regex):
        if isinstance(str_or_regex, str):
            str_or_regex = re.compile(str_or_regex)
        return self.__class__(self.df[self.df["DistrictName"].str.contains(str_or_regex)])

    def select_zip(self, zip):
        if isinstance(zip, str):
            zip = int(str)

        if isinstance(zip, int):
            return self.__class__(self.df[self.df[ZIP] == zip])
        else:
            zip = set(zip)
            return self.__class__(self.df[self.df[ZIP].isin(zip)])

    @functools.cached_property
    def by_zip(self):
        return self.df.set_index(ZIP)

    @functools.cached_property
    def zip2city(self):
        return self.by_zip["City"].unique()

    @functools.cached_property
    def zip2district(self):
        return self.by_zip["DistrictName"].unique()

def zip_mappr(x):
    if isinstance(x, int): return x
    return int(x.split('-')[0])

@dataclasses.dataclass
class PublicSchoolsDirectory(SchoolsDirectory):

    @functools.cached_property
    def df(self):
        """
       'ESDCode', 'ESDName', 'LEACode', 'LEAName', 'SchoolCode', 'SchoolName',
       'LowestGrade', 'HighestGrade', 'AddressLine1', 'AddressLine2', 'City',
       'State', 'Zip', 'PrincipalName', 'Email', 'Phone', 'OrgCategoryList',
       'AYPCode', 'GradeCategory', 'TransitionToKindergarten', 'Sector'
        :return:
        """
        if self._df is not None:
            return self._df
        df = pd.read_csv(f'{publicDataRoot}/Washington_School_Directory_20251019.csv')
        df = df.rename(columns={"ZipCode": ZIP})
        df[ZIP] = df[ZIP].apply(zip_mappr)
        df[SCHOOL_CODE] = df[SCHOOL_CODE].astype(int)
        df['City'] = fixCaps(df['City'])
        df[SCHOOL_NAME] = fixCaps(df[SCHOOL_NAME])
        df["Sector"] = "public"
        df.rename(columns={"LEAName": "DistrictName"}, inplace=True)
        return df

@dataclasses.dataclass
class PrivateSchoolsDirectory(SchoolsDirectory):

    @functools.cached_property
    def df(self):
        """
        'Approval Date', 'SchoolName', 'Street Address', 'City', 'State', 'Zip',
       'County', 'School District', 'Profit', 'Online Option',
       'Lasst Updated 11/7/2025', 'Sector'
        :return: pd.DataFrame
        """
        if self._df is not None:
            return self._df

        # TODO. Add more years.
        df = pd.read_excel(f'{privateDataRoot}/2025-26 Approved Private Schools.xlsx')
        # Make consistent with public data
        df.rename(columns={"ZIP": ZIP}, inplace=True)
        df[ZIP] = df[ZIP].apply(zip_mappr)
        df.rename(columns={"School Name": "SchoolName"}, inplace=True)
        df.rename(columns={"District Name": "DistrictName"}, inplace=True)
        df.rename(columns={"School District": "DistrictName"}, inplace=True)
        df["Sector"] = "private"
        df['City'] = fixCaps(df['City'])
        df['County'] = fixCaps(df['County'])
        df[SCHOOL_NAME] = fixCaps(df[SCHOOL_NAME])
        return df

public = PublicSchoolsDirectory()
private = PrivateSchoolsDirectory()

if __name__=='__main__':
    private.df[["SchoolName", "City", ZIP]]
    public.df[["SchoolName", "City", ZIP]]
    # private.df["DistrictName"]
    # public.df["DistrictName"]
    # private.df.columns

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None,
                           'display.max_colwidth', None):
        # print(public.select_district("Northshore School District"))
        # print(private.select_district(".*Northshore.*").df)
        # # print(public.select_district(".*Northshore.*").df.tail(20))
        # print(len(public.select_district("Northshore School District").df))
        # print(len(public.select_district(".*Northshore.*").df))
        # print(len(private.select_district(".*Northshore.*").df))

        # print(public.select_district(".*Issaquah.*").df)
        zips = public.select_district(".*Issaquah.*").df[ZIP]
        # print(private.select_district(".*Issaquah.*").df)
        print(private.select_zip(zips).df)



    zips = public.select_district(".*Bellevue.*").df[ZIP].unique().tolist()
    print("Bellevue public zips", sorted(set(int(i) for i in zips)))
    zips = private.select_district(".*Bellevue.*").df[ZIP].unique().tolist()
    print("Bellevue private zips", sorted(set(int(i) for i in zips)))

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None,
                           'display.max_colwidth', None):
        print(private.select_district(".*Renton.*"))


    with_zip = public.select_zip(zips)
    with_zip.df["DistrictName"].unique()

    public.select_zip(11901)["DistrictName"].unique()
    private.select_zip(11901)["DistrictName"].unique()
    # 11901

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None,
                           'display.max_colwidth', None):
        private.df[private.df["DistrictName"].str.contains(re.compile(".*Bellevue.*"))]
        public.df[public.df["DistrictName"].str.contains(re.compile(".*Bellevue School District.*"))].sample

    pass
