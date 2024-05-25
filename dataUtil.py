import dataclasses
import os
import re
import time
from functools import cached_property

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import data

matplotlib.rcParams["axes.formatter.limits"] = (-99, 99)


def timed(func=None):
    def timed(func):
        name = func.__name__

        def timed(*args, **kw):
            t = time.time()
            result = func(*args, **kw)
            print("**** {:<10.4}s : {}".format(time.time() - t, name))
            return result

        return timed

    return timed(func) if func else timed


def clean(f):
    return f.replace(" ", "_").replace(",", "_").replace("/", "_").replace("__", "_")


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


def addDeltas(df, totalName="Total"):
    dOld = df[totalName].values[:-1]
    dNew = df[totalName].values[1:]
    delta = (dNew - dOld)
    dTot = np.concatenate([np.zeros(1, dtype=delta.dtype), delta])
    dPercent = np.concatenate([np.zeros(1, dtype=delta.dtype), deltaPercent(dOld, dNew)])
    df = df.assign(**{'ΔTotal': dTot.astype(str),
                      'Δ%': [f"{x:3.1f}%" for x in dPercent]})
    df.iloc[0, df.columns.get_loc("ΔTotal")] = ""
    df.iloc[0, df.columns.get_loc("Δ%")] = ""
    return df


def saveSVGFig(plt, filename):
    print('Saving', filename)
    os.makedirs(os.path.split(filename)[0], exist_ok=True)
    plt.savefig(filename, format='svg')
    # os.system(f"xdg-open {filename}")


def pltShow():
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

    @cached_property
    def totalsByYear(self):
        df = self.df[["Year", "Total"]]
        s = df.groupby("Year")["Total"].sum()
        s.name = ", ".join(self.path)
        return s

    @staticmethod
    def stack(*series, xlabel=None, ylabel=None, title=None, savedir=None):
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

    def county(self, county):
        df = self.df[self.df["County"].astype(str).str.contains(county)]
        names = df["County"].unique()
        print("Counties:", ", ".join(list(names)))

        selection = names[0] if len(names) == 1 else county
        return self.update(
            df=df,
            # selection=selection,
            path=self.path + (selection,)
        )

    def school(self, school):
        return self.update(
            df=self.df[self.df["School Name"] == school],
            # selection=school,
            path=self.path + (school,)
        )

    @cached_property
    def schools(self):
        return list(self.df["School Name"].unique())

    def since(self, year, addPath=True):
        return self.update(
            df=self.df[self.df["Year"] >= year],
            # selection=f">={year}",
            path=self.path + (f">={year}",) if addPath else self.path
        )

    # @cached_property
    def years(self, years, name=None):
        assert not isinstance(years, str)
        if name is None:
            name = ",".join(list(years))
        years = {int(s) for s in years}
        return self.update(
            df=self.df[self.df["Year"].isin(years)],
            # selection=name,
            path=self.path + (name,)
        )

    def grades(self, grades, name=None):
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
    def region(self, region, title=None):
        # Trim data to a particular school district
        if not isinstance(region, str):
            df = self.df[self.df["Region"].astype(str).str.lower().str.match(region)]
        else:
            df = self.df[self.df["Region"].astype(str).str.lower().str.contains(region.lower())]
        names = df["Region"].unique()
        print("Regions:", ", ".join(list(names)))

        selection = title or (names[0] if len(names) == 1 else region)
        return self.update(
            df=df,
            # selection=selection,
            path=self.path + (selection,)
        )

    @timed
    def reportRegions(self):
        print("reportRegions", ", ".join(self.path))
        dAllRegionOfInterest = self.df
        print(f'Regions considered:')
        display(dAllRegionOfInterest["Region"].unique())

        print(f'\nSchools reporting: {dAllRegionOfInterest["School Name"].unique().size}.')
        display(list(dAllRegionOfInterest["School Name"].unique()))
        zeroReportedAnyYear = dAllRegionOfInterest.groupby("School Name")["Total"].sum()
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
            schools = df["School Name"].unique()
            for school in schools:
                d = df[df["School Name"] == school]
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
            schools = df["School Name"].unique()
            for school in schools:
                d = df[df["School Name"] == school]
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
            schools = df["School Name"].unique()
            for school in schools:
                d = df[df["School Name"] == school]
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
        fname = clean(f"{pre}{self.path[-1]}{post}")
        dirs = ['plots'] + [clean(f) for f in self.path[:-1]]
        if addDir:
            dirs.append(addDir)

        if dirs:
            dirs = "/".join(dirs)
            os.makedirs(dirs, exist_ok=True)
            return "/".join([dirs, fname])
        else:
            return fname

    @timed
    def plotCohortProgression(self):
        def plotGradeProgression(df):
            grades = ['K'] + [str(i) for i in range(1, 13)]

            def gen(df):

                _df = df[[g for g in grades if g in df.columns]]
                yeari = 0

                if _df.shape[0] == 0: return
                if _df.shape[1] == 1: return

                gradeStart = index2year[min([yearSortMap[i] for i in _df.columns])]

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
                print(f"plotCohortProgression: Empty data: {self.path}")
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

            pltShow()
            saveSVGFig(f, self.filename(post="_cohorts.svg", addDir="schoolCohorts"))

        # if dfName is None:
        #     dfName = ",".join(self.path)
        title = ", ".join(self.path)
        # filenameBase = "/".join(t.replace(', ', '_').replace(' ', '_') in self.path)
        # filename = "/".join(self.path[-1])

        yearSortMap = {str(i): i for i in range(1, 13)} | {i: i for i in range(1, 13)} | {"P": -1, "K": 0}
        index2year = {v: str(k) for k, v in yearSortMap.items()}

        def yearSort(g):
            # print("**", g.name)
            # if g.name!='Grade': return g
            with pd.option_context("future.no_silent_downcasting", True):
                r = g.replace(yearSortMap).infer_objects(copy=False)
                return r

        # for df in
        # print(dfName, ">" * 20)
        df = self.df
        df = df[["Grade", "Year", "Total"]].copy()
        totals = df.groupby(["Year", "Grade"])["Total"].sum().to_frame()

        params = (
            # 'display.height', 100000,
            'display.width', 100000,
            'display.max_rows', 100000,
            'display.max_columns', 100000)
        with pd.option_context(*params):
            df = totals.reset_index().sort_values(["Grade", "Year"]).set_index("Year").pivot(
                columns=["Grade"]).fillna(0).astype(int).sort_values(by="Grade", axis=1, key=yearSort)
            df = df["Total"]
            plotGradeProgression(df)  # , ylabel=f"{dfName} Grade Enrollment Progression")
            #       df["Total"] = df.sum(axis=1)
            d = df.assign(Total=df.sum(axis=1))
            d = addDeltas(d)
            cols = list(d.columns)
            #       d = d.assign(Year=d.index.values) #["Year"+cols]

            display(d)
            df2SVGFile(d.reset_index()[["Year"] + cols],
                       self.filename(post="_cohorts_table.svg", addDir="schoolCohorts"))
        print(self.path, "<" * 20)
        matplotlib.pyplot.close()

    @timed
    def plotSchools(self):
        df = self.df

        def lastYearTotal(s):
            _df = df[df["School Name"] == s]
            yearMax = _df["Year"].max()
            return -(_df[_df["Year"] == yearMax]["Total"].sum())

        def plotSchool(df, schooName):
            data = df[df["School Name"] == schooName].groupby("Year")["Total"].sum()
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

        saveSVGFig(f, self.filename(post="_school.svg"))

    @timed
    def plotGrades(self, grades=None, title=None):
        df = self.df

        def lastYearTotal(s):
            _df = df[df["Grade"] == s]
            yearMax = _df["Year"].max()
            return -(_df[_df["Year"] == yearMax]["Total"].sum())

        def plotGrade(df, gradeName):
            data = df[df["Grade"] == gradeName].groupby("Year")["Total"].sum()
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

        saveSVGFig(f, "/".join([self.filename(addDir='gradeEnrollment'), clean(f"{title}_gradeGroup.svg")]))
        matplotlib.pyplot.close()


def defaultFigSize(f, width=8):
    f.set_figwidth(width)
    f.set_figheight(6)


def publicData():
    return DataSet(df=data.publicData(),
                   # selection="Washington Public",
                   path=("Washington Public",))


def privateData():
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


# [school2style(s) for s in sorted(dAllRegionOfInterest["School Name"].unique())]


if __name__ == "__main__":
    import openpyxl


    def makePlots(public, private, schoolDetail=False):
        # Cohort for all
        for dfBase in [
            public,
            private,
        ]:
            dfBase.plotCohortProgression()

        public.reportYearlyChangesSinceBaselineYear(2019)
        private.reportYearlyChangesSinceBaselineYear(2019)

        for ds in [
            public,
            private,
        ]:
            for name, gradeYears in gradeSets:
                ds.plotGrades(name, title=name)

        if 1:  # A vs B stack plots
            a = public.since(2019, False)
            b = private.since(2019, False)
            DataSet.stack(
                a.totalsByYear,
                b.totalsByYear,
                ylabel="Enrollment",
                savedir=a.filename()
            )

            for grade in ["Elementary", "Middle", "High", "K"]:
                a = public.since(2019, False).grades(grade)
                b = private.since(2019, False).grades(grade)
                DataSet.stack(
                    a.totalsByYear,
                    b.totalsByYear,
                    ylabel="Enrollment",
                    savedir=a.filename()
                )

        if schoolDetail:
            # Plot schools
            for ds in [public, private]:
                for name, gradeYears in gradeSets:
                    ds.grades(name, name=name).plotSchools()

            # Cohort for school
            for dfBase in [public.school(s) for s in public.schools] \
                          + [private.school(s) for s in private.schools]:
                dfBase.plotCohortProgression()


    gradeSets = [
        ("Elementary", ['K'] + [str(s) for s in range(1, 6)]),
        ("Middle", [str(s) for s in range(6, 9)]),
        ("High", [str(s) for s in range(9, 13)]),
        ("K-12", ['K'] + [str(s) for s in range(1, 13)]),
        ("P-12", ['P', 'K'] + [str(s) for s in range(1, 13)]),
        ("P-5", ['P', 'K'] + [str(s) for s in range(1, 6)]),
        ("K-5", ['K'] + [str(s) for s in range(1, 6)]),
    ]


    def run():

        public = publicData()
        private = privateData()

        makePlots(public.region("Lake Washington School District",
                                title="LWSD"),
                  private.region(re.compile("(kirkland)|(redmond)|(sammamish)"),
                                 title="Kirkland/Redmond/Sammamish"
                                 ), schoolDetail=True)

        makePlots(public.region("Seattle"), private.region("Seattle"), schoolDetail=True)

        makePlots(public.region("Bellevue"), private.region("Bellevue"), schoolDetail=True)
        # State wide
        makePlots(public, private)

        # King county (public only)
        for dfBase in [public.county("King")]:
            dfBase.plotCohortProgression()

        pass


    def noOp(*args, **kw):
        pass


    display = noOp
    pltShow = noOp

    run()
