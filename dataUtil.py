import dataclasses
import os
import time
from functools import cached_property
from typing import Callable, Dict

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import data
from schools_directory import ZIP, SCHOOL_NAME, DISTRICT_NAME

SHOULD_SHOW = False
matplotlib.rcParams["axes.formatter.limits"] = (-99, 99)

def noOp(*args, **kw):
    pass

display = noOp

def timed(func=None):
    def timed(func):
        name = func.__name__

        def timed(*args, **kw):
            t = time.time()
            result = func(*args, **kw)
            print("**** {:<10.2f}seconds : {}".format(time.time() - t, name))
            return result

        return timed

    return timed(func) if func else timed


def clean(f):
    return f.replace(" ", "_").replace(",", "_").replace("/", "_").replace("__", "_").replace("-", "_")


nameYears = [
    ('P-12', ["P", "K"] + [str(s) for s in range(1, 13)]),
    ('K-12', ["K"] + [str(s) for s in range(1, 13)]),
    ('K-5', ["K"] + [str(s) for s in range(1, 6)]),
    ('6-8', [str(s) for s in range(6, 9)]),
    ('9-12', [str(s) for s in range(9, 13)]),
    ('K-3', ["K"] + [str(s) for s in range(1, 4)]),
    ('P-3', ["P", "K"] + [str(s) for s in range(1, 4)]),
    ('P-5', ["P", "K"] + [str(s) for s in range(1, 6)]),
    ('1-3', [str(s) for s in range(1, 4)]),
]

name2Years = dict(nameYears)
name2Years["Elementary"] = ["K"] + [str(s) for s in range(1, 6)]
name2Years["Middle"] = [str(s) for s in range(6, 9)]
name2Years['High'] = [str(s) for s in range(9, 13)]

nameYears.extend(
    (g, [g]) for g in ["P", "K"] + [str(s) for s in range(1, 13)]
)


class Updateable:
    """
    An immutable dataclass with an update method. Update returns a new object with the attribute replaced.
    """

    def update(self, **changes):
        # changes
        fields = [f.name for f in dataclasses.fields(self)]
        for name in changes:
            if name not in fields:
                raise Exception(f"{name} in not in {self.__class__.__name__}.{fields}")
        return dataclasses.replace(self, **changes)


def df2SVGFile(df, filename):
    def render_mpl_table(data, col_width=1.5, row_height=0.4, font_size=10, edges='horizontal',
                         header_color='#fff', row_colors=['w', '#eee'], edge_color='w',
                         bbox=[0, 0, 1, 1], header_columns=0,
                         ax=None, **kwargs):
        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')
        mpl_table = ax.table(cellText=data.values,
                             bbox=bbox,
                             colLabels=data.columns,
                             # edges=edges,
                             **kwargs)
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        for k, cell in mpl_table._cells.items():
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='black')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        return ax.get_figure(), ax

    fig, ax = render_mpl_table(df, header_columns=0, col_width=2.0)
    plt.plot([0, 1], [1 - 1 / (df.shape[0] + 1), 1 - 1 / (df.shape[0] + 1)], c='black', lw=0.5,
             marker='.',
             )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    fixAxis()

    saveSVGFig(plt, filename)


def deltaPercent(old, new):
    """
    x+x*p = x'
    x(1+p) = x'
    1+p = x'/x
    p = (x'/x) - 1
    """
    return 100 * ((new / old) - 1)


def addDeltas(df, totalName="Total", addVsBaseYear=None):
    dOld = df[totalName].values[:-1]
    dNew = df[totalName].values[1:]
    delta = (dNew - dOld)
    dTot = np.concatenate([np.zeros(1, dtype=delta.dtype), delta])
    dPercent = np.concatenate([np.zeros(1, dtype=delta.dtype), deltaPercent(dOld, dNew)])
    df = df.assign(**{'ΔTotal': dTot.astype(str),
                      'Δ%': [f"{x:+3.1f}%" for x in dPercent]})
    df.iloc[0, df.columns.get_loc("ΔTotal")] = ""
    df.iloc[0, df.columns.get_loc("Δ%")] = ""

    if addVsBaseYear:
        tots = df["Total"]
        _baseTotal = int(tots[addVsBaseYear])
        df[f"Δ{addVsBaseYear}"] = tots - _baseTotal
        df[f"Δ% since {addVsBaseYear}"] = ((100 * tots / _baseTotal) - 100).map(lambda s: f"{s:+3.1f}" + "%")

    return df


def saveSVGFig(plt, filename):
    print('Saving', filename)
    os.makedirs(os.path.split(filename)[0], exist_ok=True)
    plt.savefig(filename, format='svg')
    # os.system(f"xdg-open {filename}")


def pltShow():
    if SHOULD_SHOW:
        plt.show()


def fixAxis():
    # try:
    #     # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    #     plt.gca().xaxis.get_major_locator().set_params(integer=True)
    # except Exception:
    #     # try:
    #     #     ax.xaxis.get_major_locator().set_params(integer=True)
    #     # except Exception:
    #     pass

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # loc='lower right')
    plt.tight_layout()


@dataclasses.dataclass
class DataSet(Updateable):
    df: pd.DataFrame
    # selection: str = "Washington"
    path: tuple[str] = ()

    def addPath(self, path):
        return self.update(path = self.path+(path,) )

    @cached_property
    def totalsByYear(self):
        df = self.df[["Year", "Total"]]
        s = df.groupby("Year")["Total"].sum()
        s.name = ", ".join(self.path)
        return s

    @staticmethod
    def plotStack(*series, xlabel=None, ylabel=None, title=None, savedir=None):
        df = pd.DataFrame({s.name: s for s in series})

        f = plt.figure(figsize=(10, 6))
        # Create the stacked bar chart
        ax = df.plot(kind='bar', stacked=True, ax=f.gca())
        # ax.ticklabel_format(style='plain')
        # ax.ticklabel_format(scilimits=(-5, 8))
        # ax.ticklabel_format(useOffset=False, style='plain')
        # Customize the plot (optional)
        if xlabel: plt.xlabel(xlabel)
        if ylabel: plt.ylabel(ylabel)
        if title: plt.title(title)
        # fixAxis()
        # if legend:
        #     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=legend)
        # else:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        #         plt.legend(title=legend)  # Add a legend
        pltShow()
        common = series[0].name

        def common_prefix(str1, str2):
            prefix = ""
            str1 = str1.split("/")
            str2 = str2.split("/")
            for i in range(min(len(str1), len(str2))):
                print(i, repr(str1[i]), repr(str2[i]))
                if str1[i] == str2[i]:
                    if prefix:
                        prefix = f"{prefix}/{str1[i]}"
                    else:
                        prefix = str1[i]
                else:
                    break
            return prefix

        from functools import reduce

        common = reduce(common_prefix, [s.name for s in series])

        # ax = df.plot(kind='bar', stacked=True, figsize=(8, 6), ax=f.gca())
        # fixAxis()
        filename = ("_vs_".join(clean(s.name.replace(common, "")) for s in series)) + ".svg"
        if savedir:
            filename = os.path.join(savedir, filename)
        saveSVGFig(plt, filename)
        matplotlib.pyplot.close()

        # pltShow()
        # return plt

    def select_county(self, county):
        df = self.df[self.df["County"].astype(str).str.contains(county)]
        names = df["County"].unique()
        print("Counties:", ", ".join(list(names)))

        selection = names[0] if len(names) == 1 else county
        return self.update(
            df=df,
            # selection=selection,
            path=self.path + (selection,)
        )

    def select_school(self, school):
        return self.update(
            df=self.df[self.df[SCHOOL_NAME] == school],
            # selection=school,
            path=self.path + (school,)
        )

    @cached_property
    def schools(self):
        return list(self.df[SCHOOL_NAME].unique())

    def select_since(self, year, addPath=True):
        return self.update(
            df=self.df[self.df["Year"] >= year],
            # selection=f">={year}",
            path=self.path + (f">={year}",) if addPath else self.path
        )

    # @cached_property
    def select_years(self, years, name=None):
        assert not isinstance(years, str)
        if name is None:
            name = ",".join(list(years))
        years = {int(s) for s in years}
        return self.update(
            df=self.df[self.df["Year"].isin(years)],
            # selection=name,
            path=self.path + (name,)
        )

    def select_grades(self, grades, name=None):
        if isinstance(grades, str):
            name = name or grades
            grades = name2Years[name]
        assert not isinstance(grades, str)
        if name is None:
            name = ",".join(list(grades))
        grades = {str(s) for s in grades}
        return self.update(
            df=self.df[self.df["Grade"].isin(grades)],
            # selection=name,
            path=self.path + (name,)
        )

    # @cached_property
    # def regions(self):
    #     return df["Region"].unique()

    # @cached_property
    def select_region(self, region, title=None):
        # Trim data to a particular school district
        if isinstance(region, str) and "*" in region:
            region = re.compile(region)

        if not isinstance(region, str):
            assert isinstance(region, re.Pattern)
            df = self.df[self.df[DISTRICT_NAME].astype(str).str.match(region)]
        else:
            assert isinstance(region, str)
            df = self.df[self.df[DISTRICT_NAME].astype(str).str.lower().str.contains(region.lower())]

        names = df[DISTRICT_NAME].unique()
        print("Regions:", list(names))

        if len(df)==0:
            raise Exception(f"Empty region: {region}")
        selection = title or (names[0] if len(names) == 1 else region)
        # print(df.summary())
        # print(sorted(df["Grade"].unique()))
        # print(sorted(df[df["Year"]==2024]["Grade"].unique()))
        # print(df[(df["Year"]==2023) & (df["Grade"]=='1')]["Total"].sum())
        # print(df[(df["Year"]==2023) ]["Total"].sum())
        # self.df[self.df["ESDName"].apply(lambda s:not isinstance(s, str))]
        # sorted(self.df["ESDName"].unique())

        return self.update(
            df=df,
            # selection=selection,
            path=self.path + (selection,)
        )

    def select_city(self, city, title=None):
        # Trim data to a particular school district
        if isinstance(city, str) and "*" in city:
            city = re.compile(city)

        if not isinstance(city, str):
            df = self.df[self.df['City'].astype(str).str.match(city)]
        else:
            df = self.df[self.df['City'].astype(str).str.lower().str.contains(city.lower())]

        names = df['City'].unique()
        print("Select city:", list(names))

        if len(df)==0:
            raise Exception(f"Empty city: {city}")
        selection = title or (names[0] if len(names) == 1 else city)
        return self.update(
            df=df,
            # selection=selection,
            path=self.path + (selection,)
        )

    def select_zips(self, zips, title=None):
        # Trim data to a particular school district
        df = self.df[self.df[ZIP].isin(zips)]
        if len(df)==0:
            raise Exception(f"Empty zips: {zips}")
        print(f"Search {title} for zips: {sorted((int(s) for s in set(zips)))}")
        # print(f"Using schools:")
        d = df.set_index(SCHOOL_NAME)
        for s in sorted((df[SCHOOL_NAME].unique())):
            d_ = d[d.index==s]
            d_ = d_[["City", ZIP]].drop_duplicates()
            # if len(d_)==1:
            #     city,zip = d_.iloc[0]
            #     print(f"{s} {city} {zip}")
            # elif len(d)>=2:
            if 1:
                print(f"{s}")
                for i, (city,zip) in d_.iterrows():
                    print(f"  - {city} {zip}")
        print(f"Schools matched in cities: {sorted(set(df['City']))}")
        print(f"Schools matched with zips: {sorted(set(df[ZIP]))}")
        return self.update(
            df=df,
            path=self.path + (title,)
        )


    @timed
    def reportRegions(self):
        print("reportRegions", ", ".join(self.path))
        dAllRegionOfInterest = self.df
        print(f'Regions considered:')
        display(dAllRegionOfInterest[DISTRICT_NAME].unique())

        print(f'\nSchools reporting: {dAllRegionOfInterest[SCHOOL_NAME].unique().size}.')
        display(list(dAllRegionOfInterest[SCHOOL_NAME].unique()))
        zeroReportedAnyYear = dAllRegionOfInterest.groupby(SCHOOL_NAME)["Total"].sum()
        zeroReportedAnyYear = zeroReportedAnyYear[zeroReportedAnyYear == 0]

        print(f'\nSchools reporting 0 enrollment (any year): {zeroReportedAnyYear.size}')
        display(zeroReportedAnyYear)

    @timed
    def reportYearlyChangesSinceBaselineYear(self, baselineYear=2014):
        print("reportYearlyChangesSinceBaselineYear", ", ".join(self.path))
        df = self.df
        df = df[df["Year"] >= baselineYear]
        firstYear = df["Year"].min()
        firstYearTotal = df[df["Year"] == firstYear]["Total"].sum()
        print(firstYear, firstYearTotal)
        df = df.groupby("Year")["Total"].sum().to_frame()

        df = addDeltas(df)
        # df["Change"] = df["Total"] - firstYearTotal
        # df["% Change"] = 100 * (df["Change"] / df["Total"])
        display(df)

    @timed
    def plotSchoolPercentGrowth(self, yearStart=2014, nameYears=nameYears):
        import math
        print("plotSchoolPercentGrowth", yearStart, ", ".join(self.path))

        def genGrowthPercent(df):
            schools = df[SCHOOL_NAME].unique()
            for school in schools:
                d = df[df[SCHOOL_NAME] == school]
                d = d[~(d["Total"].isna())]
                if len(d) == 0: continue

                firstYear = d["Year"].min()
                if math.isnan(firstYear): continue
                firstYearEnrollment = int(d[d["Year"] == firstYear].groupby("Year")["Total"].sum())
                d = 100 * (d.groupby("Year")["Total"].sum() - firstYearEnrollment) / firstYearEnrollment
                sortkey = d.values[-1]
                yield (-sortkey, school), (d.index.values, d.values)

        print(f"YEAR > {yearStart} -------------------------")
        for title, selectGrades in nameYears:
            # Plot Enrollment Growth
            f = self._plot(genGrowthPercent, baselineYear=yearStart, selectGrades=selectGrades, title=title,
                           xlabel='Year (start)',
                           ylabel='Enrollment Growth (%)')
            saveSVGFig(f, self.filename(post="_enrollmentGrowthPercent.svg"))
        matplotlib.pyplot.close()

    @timed
    def plotSchoolTotalGrowth(self, yearStart=2014, nameYears=nameYears):
        # Absolute enrollment change
        import math
        print("plotSchoolTotalGrowth", yearStart, ", ".join(self.path))

        def genGrowth(df):
            schools = df[SCHOOL_NAME].unique()
            for school in schools:
                d = df[df[SCHOOL_NAME] == school]
                d = d[~(d["Total"].isna())]
                if len(d) == 0: continue
                firstYear = d["Year"].min()
                if math.isnan(firstYear): continue
                firstYearEnrollment = int(d[d["Year"] == firstYear].groupby("Year")["Total"].sum())
                d = d.groupby("Year")["Total"].sum() - firstYearEnrollment
                sortkey = d.values[-1]
                yield (-sortkey, school), (d.index.values, d.values)

        # for year in [2014, 2019]:
        print(f"YEAR > {yearStart} -------------------------")
        for title, selectGrades in nameYears:
            f = self._plot(genGrowth, baselineYear=yearStart, selectGrades=selectGrades, title=title,
                           xlabel='Year (start)',
                           ylabel='Enrollment Growth')
            saveSVGFig(f, self.filename(post="_enrollmentGrowth.svg"))
        matplotlib.pyplot.close()

    @timed
    def plotSchoolEnrollment(self, yearStart=2014, nameYears=nameYears):
        # Total enrollment over time per school
        print("plotSchoolEnrollment", yearStart, ", ".join(self.path))

        def genTotals(df):
            schools = df[SCHOOL_NAME].unique()
            for school in schools:
                d = df[df[SCHOOL_NAME] == school]
                d = d[~(d["Total"].isna())]
                # if school=="Highland Middle School":  1 student in grade 5 :-/
                #     print("SCHOOL", school)
                #     display(d)
                if d.empty: continue
                d = d.groupby("Year")["Total"].sum()
                sortkey = d.values[-1]
                yield (-sortkey, school), (d.index, d.values)

        # for year in [2014, 2019]:
        print(f"YEAR > {yearStart} -------------------------")
        for title, selectGrades in nameYears:
            f = self._plot(genTotals, baselineYear=yearStart, selectGrades=selectGrades,
                           title=title, xlabel='Year (start)', ylabel='Enrollment Growth')
            saveSVGFig(f, self.filename(post="_enrollment.svg"))
        matplotlib.pyplot.close()

    @timed
    def plotSchools(self, addDir=""):
        df = self.df

        def lastYearTotal(s):
            _df = df[df[SCHOOL_NAME] == s]
            yearMax = _df["Year"].max()
            return -(_df[_df["Year"] == yearMax]["Total"].sum())

        def plotSchool(df, schooName):
            data = df[df[SCHOOL_NAME] == schooName].groupby("Year")["Total"].sum()
            data = data[data > 0]
            if len(data) > 0:
                ax.plot(data.index.astype(int), data.values,
                        label=schooName,
                        marker='.', )
            # fixAxis()

        f, ax = plt.subplots(1, figsize=(12, 10))
        ax.xaxis.get_major_locator().set_params(integer=True)
        for schooName in sorted(self.schools, key=lastYearTotal):
            plotSchool(df, schooName)

        # if title is None:
        #     title = ",".join(sorted(self.df["Grade"].unique()))
        # plt.title(title)
        plt.title(", ".join(self.path))
        plt.ylabel("Enrollment")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # loc='lower right')
        plt.tight_layout()

        # defaultFigSize(f)
        pltShow()

        saveSVGFig(f, self.filename(post=".schools_enrollment_trend.svg", addDir=addDir))

    def _plot(self, genFunc, baselineYear, selectGrades, title, xlabel, ylabel):
        df = self.df
        f = plt.figure(figsize=(8, 6))
        defaultFigSize(f)

        df = df[df["Year"] >= baselineYear]
        df = df[df["Grade"].isin(selectGrades)]
        for i, ((_sum, school), (x, y)) in enumerate(sorted(genFunc(df), key=lambda x: x[0][0])):
            linestyle, color = school2style(school)
            plt.plot(x, y, linestyle=linestyle, color=color, label=school,
                     marker='.')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pltShow()
        return f

    def filename(self, pre="", post="", addDir=""):
        fname = clean(f"{pre}{self.path[-1]}")+post
        dirs = ['plots'] + [clean(f) for f in self.path[:-1]]
        if addDir:
            dirs.append(addDir)

        if dirs:
            dirs = "/".join(dirs)
            os.makedirs(dirs, exist_ok=True)
            return "/".join([dirs, fname]).replace("//","/")
        else:
            return fname

    def plotGradeTierEnrollment(self, addDir="", post=""):
        f, ax = plt.subplots(1, figsize=(8, 6))

        for grade in ["Elementary", "Middle", "High"]:
            data = self.select_grades(grade).df
            data = data[["Year", "Total"]]
            totals = data.groupby(["Year"])["Total"].sum().to_frame()
            ax.plot(totals.index, totals.values, label=grade, marker='.', )

        plt.ylim(ymin=0)
        plt.title(", ".join(self.path))

        plt.ylabel("Enrollment")
        # plt.ylabel("Enrollment")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # loc='lower right')
        plt.tight_layout()
        # do_plot()

        pltShow()
        # path = clean("_".join(self.path))
        saveSVGFig(f, self.filename(addDir=addDir, post=post )+f"/enrollment.gradeTier.svg")

    def plotTotalEnrollment(self, addDir="", post=""):
        f, ax = plt.subplots(1, figsize=(8, 6))

        data = self.df
        data = data[["Year", "Total"]]
        totals = data.groupby(["Year"])["Total"].sum().to_frame()
        ax.plot(totals.index, totals.values, marker='.', )

        plt.ylim(ymin=0)
        plt.title(", ".join(self.path))

        plt.ylabel("Enrollment")
        # plt.ylabel("Enrollment")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # loc='lower right')
        plt.tight_layout()
        # do_plot()

        pltShow()
        # path = clean("_".join(self.path))
        saveSVGFig(f, self.filename(addDir=addDir, post=post )+f"/enrollment.total.svg")



    @timed
    def plotCohortProgression(self, addDir="", post="", addVsBaseYear=None, outputTable=True):

        if len(self.df)==0:
            print("plotCohortProgression empty:", self.path)
            return

        # if dfName is None:
        #     dfName = ",".join(self.path)
        # filenameBase = "/".join(t.replace(', ', '_').replace(' ', '_') in self.path)
        # filename = "/".join(self.path[-1])


        # for df in
        # print(dfName, ">" * 20)
        if outputTable:
            self.saveGradeYearTableWithTotals(
                self.df, #self.asGradeYearTotal,
                year_sort=self.year_sort,
                filename=self.filename(addDir=addDir, post=post) + f".cohorts_table.svg",
                addVsBaseYear=addVsBaseYear)

        # params = (
        #     # 'display.height', 100000,
        #     'display.width', 100000,
        #     'display.max_rows', 100000,
        #     'display.max_columns', 100000)
        # with pd.option_context(*params):
        self._plotGradeProgression(self.asGradeYearTotal,
                                   yearSort=self.year_sort,
                                   path=self.path,
                                   filename=self.filename(addDir=addDir, post=post )+f".cohorts.svg")  # , ylabel=f"{dfName} Grade Enrollment Progression")

        print(self.path, "<" * 20)
        matplotlib.pyplot.close()

    @staticmethod
    def _plotGradeProgression(df, yearSort, path, filename):
        title = ", ".join(path)

        # yearSort = self.year_sort
        grades = ['K'] + [str(i) for i in range(1, 13)]

        def gen(df):

            _df = df[[g for g in grades if g in df.columns]]
            yeari = 0

            if _df.shape[0] == 0: return
            if _df.shape[1] == 1: return

            gradeStart = yearSort.index2year[min([yearSort.yearSortMap[i] for i in _df.columns])]

            gradeStartI = grades.index(gradeStart)

            while True:
                year_ = []
                total_ = []
                try:
                    year1 = _df.index[yeari]
                except IndexError:
                    break

                i = 0
                while True:
                    try:
                        year = year1 + i
                        g = grades[i + gradeStartI]
                        tot = _df.loc[year][g]
                        total_.append(tot)
                        year_.append(year)
                    except IndexError:
                        break
                    except KeyError:
                        break
                    i += 1
                s = pd.Series(index=year_, data=total_, name=str(_df.index[yeari]))
                s = s[s != 0]
                yield f"Grade {gradeStart} of {year1}", s
                yeari += 1

        _grade2PlotParam = dict(nameYears)
        f, ax = plt.subplots(1, figsize=(8, 6))
        # defaultFigSize(f, width=10)

        try:
            datas = list(gen(df))
            ymax = max([d.max() for year1, d in datas if not math.isnan(d.max())])
        except ValueError:
            print(f"plotCohortProgression: Empty data: {path}")
            return

        for name, data in datas:
            #         print(name)
            ax.plot(data.index, data.values, label=name, marker='.', )
            # ax.set_ylim(ymin=0, ymax=1.15 * ymax)
            # ax.set_xlim(xmax=2028.5)
        # fixAxis()

        # ax.legend(loc='lower right')
        #     display(s.to_frame().plot())
        # plt.ylabel(ylabel)
        # plt.ylabel("Cohort Enrollment")
        # plt.title(title)

        plt.ylim(ymin=0)
        plt.title(title)
        plt.ylabel("Cohort Enrollment")
        # plt.ylabel("Enrollment")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # loc='lower right')
        plt.tight_layout()
        # do_plot()

        pltShow()
        # path = clean("_".join(self.path))
        saveSVGFig(f, filename)

    @staticmethod
    def saveGradeYearTableWithTotals(df, filename, year_sort, addVsBaseYear=None):
        # df = self.df
        # df = df[["Grade", "Year", "Total"]].copy()
        # totals = df.groupby(["Year", "Grade"])["Total"].sum().to_frame()

        params = (
            # 'display.height', 100000,
            'display.width', 100000,
            'display.max_rows', 100000,
            'display.max_columns', 100000)
        with pd.option_context(*params):
            # df = totals.reset_index().sort_values(["Grade", "Year"]).set_index("Year").pivot(
            #     columns=["Grade"]).fillna(0).astype(int).sort_values(by="Grade", axis=1, key=self.year_sort.sort)
            # df = df["Total"]
            # plotGradeProgression(df)  # , ylabel=f"{dfName} Grade Enrollment Progression")
            #       df["Total"] = df.sum(axis=1)
            _df = df
            df = _df


            df = df[["Grade", "Year", "Total"]].copy()
            totals = df.groupby(["Year", "Grade"])["Total"].sum().to_frame()
            params = (
                # 'display.height', 100000,
                'display.width', 100000,
                'display.max_rows', 100000,
                'display.max_columns', 100000)
            with pd.option_context(*params):
                df = totals.reset_index().sort_values(["Grade", "Year"]).set_index("Year").pivot(
                    columns=["Grade"]).fillna(0).astype(int).sort_values(by="Grade", axis=1, key=year_sort.sort)
                df = df["Total"]
                # plotGradeProgression(df)  # , ylabel=f"{dfName} Grade Enrollment Progression")
                #       df["Total"] = df.sum(axis=1)
                d = df.assign(Total=df.sum(axis=1))
                d = addDeltas(d, addVsBaseYear=addVsBaseYear)


            # d = addDeltas(d)


                cols = list(d.columns)
                #       d = d.assign(Year=d.index.values) #["Year"+cols]

                display(d)

                df2SVGFile(d.reset_index()[["Year"] + cols],
                           # self.filename(post="_cohorts_table.svg", addDir="schoolCohorts")
                           filename
                           )

    @cached_property
    def asGradeYearTotal(self):
            df = self.df[["Grade", "Year", "Total"]].copy()
            totals = df.groupby(["Year", "Grade"])["Total"].sum().to_frame()
            df = totals.reset_index().sort_values(["Grade", "Year"]).set_index("Year").pivot(
                columns=["Grade"]).fillna(0).astype(int).sort_values(by="Grade", axis=1, key=self.year_sort.sort)
            df = df["Total"]
            return df

    @cached_property
    def year_sort(self):
        yearSortMap = {str(i): i for i in range(1, 13)} | {i: i for i in range(1, 13)} | {"P": -1, "K": 0}
        index2year = {v: str(k) for k, v in yearSortMap.items()}
        def yearSort(g):
            # print("**", g.name)
            # if g.name!='Grade': return g
            with pd.option_context("future.no_silent_downcasting", True):
                r = g.replace(yearSortMap).infer_objects(copy=False)
                return r

        @dataclasses.dataclass
        class YearSort:
            sort:Callable
            index2year:Dict
            yearSortMap:Dict
        return YearSort(yearSort, index2year, yearSortMap)

    @timed
    def plotGrades(self, grades=None, title=None, addDir="", outputTable=False):
        df = self.df

        def lastYearTotal(s):
            _df = df[df["Grade"] == s]
            yearMax = _df["Year"].max()
            return -(_df[_df["Year"] == yearMax]["Total"].sum())

        foundGrades = []
        def plotGrade(df, gradeName):
            data = df[df["Grade"] == gradeName].groupby("Year")["Total"].sum()
            if len(data)>0:
                foundGrades.append(gradeName)
            ax.plot(data.index.astype(int), data.values,
                    label=gradeName,
                    marker='.', )

        if grades is None:
            grades = ["P", "K"] + [str(s) for s in range(1, 13)]
        elif isinstance(grades, str):
            grades = name2Years[grades]

        f, ax = plt.subplots(1, figsize=(8, 6))
        print(f"g {grades}")
        for gradeName in sorted(grades, key=lastYearTotal):
            print(gradeName)
            plotGrade(df, gradeName)
        ax.set_ylim(ymin=0)
        # fixAxis()
        # ax.xaxis.get_major_locator().set_params(integer=True)
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # loc='lower right')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # loc='lower right')
        # plt.tight_layout()

        if title is None:
            title = ", ".join(grades)

        title = ", ".join(self.path + (title,))
        plt.title(title)
        plt.ylabel("Enrollment")
        # defaultFigSize(f)
        pltShow()

        filename = "/".join([self.filename()+f'/{addDir}grade_group_enrollment_trends', clean(f"{title}_gradeGroup.svg")])
        saveSVGFig(f, filename)
        matplotlib.pyplot.close()

        if outputTable:
            df = self.df
            df = df[df["Grade"].isin(set(foundGrades))]
            self.saveGradeYearTableWithTotals(df,
                                              year_sort=self.year_sort,
                                              filename=filename[:-4]+".table.svg", addVsBaseYear=2019)


def defaultFigSize(f, width=8):
    f.set_figwidth(width)
    f.set_figheight(6)


def publicData() -> DataSet:
    return DataSet(df=data.publicData(),
                   # selection="Washington Public",
                   path=("Washington Public",))


def privateData() -> DataSet:
    return DataSet(df=data.privateData(),
                   # selection="Washington Private",
                   path=("Washington Private",))


import matplotlib.colors as _colors
import itertools, random
import matplotlib.ticker as mticker

styles = list(itertools.product(['solid', 'dashed', 'dashdot', 'dotted'], _colors.TABLEAU_COLORS.values()))
random.Random(0).shuffle(styles)
_school2style = {}


def school2style(school):
    s = _school2style.get(school)
    if s is None:
        s = styles.pop()
        _school2style[school] = s
    return s


# [school2style(s) for s in sorted(dAllRegionOfInterest[SCHOOL_NAME].unique())]


