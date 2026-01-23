# Washington School Enrollment Data

This project processes data from the OSPI Data Portal and SBE Data Portal to summarize changes in Washington school
enrollment from state data (OPSI for public schoole and Washington State Board of Education for private schools).
It should be noted that the private school data reported to SBE is voluntary reporting and likely under-estimates private
school enrollment, and that reporting may be inconsistent from year to year with both *over* and under-estimation 
of public school growth being possible consequences.

Generated are available in the `plots` directories:

* [Public school enrollment](plots/Washington_Public) by district
* [Private school enrollment](plots/Washington_Private) by district by similar zip/city


Individual directories contains state-wide and county-wide (e.g. King county) views.

Several kinds of plots are available:

* Cohort progression plots. These track enrollment over time for a grade as it ages and progress. Of particular interest to
  future enrollment are trends following a grade from Kindergarten. We see:
  - recession in Kindergarten enrollment appears to have bottomed and is turning around.
  - grades generally grow year-on-year. Because few births go on to initially enter the public system after Kindergarten, we should infer that
    growth in the size of a grade as it ages is due one of the following causes: inward migration (foreign or domestic) 
    into the state and its cities, and/or movement between public and homeschool and private.

![Public Cohort Progression](plots/Washington_Public/State/cohorts.cohorts.svg)
*State public school cohort trends*

![Private Cohort Progression](plots/Washington_Private/State/cohorts.cohorts.svg)
*State private school cohort trends*


* Total enrollment plots:

![Cohort](plots/Washington_Public/State/cohorts.cohorts.svg)
*State public school cohort trends*

![Cohort](plots/Washington_Private/State/cohorts.cohorts.svg)
*State private school cohort trends*




**Caution on interpretation of private school trends** 

It is unclear whether some years had more private school reporting than others, which might significantly affect the interpretation of 
private school trends. However, independent estimation of the number of children in a city can be obtained from county demographers.
At least for Bellevue, we saw roughly a net-zero growth in the city's population of children via county demographers in the same period as 
a (roughly) net-zero public/private transfer of students during and following COVID.





Although Bellevue public school enrollment is briefly in decline, it is instructive to know where the loss is coming from.

The Bellevue school district has argued:

* declining birth rates
* families selling homes and exiting the district, to be replace by (for some reason) assumed to be childless families
* high costs of living (ignoring that families in all income brackets have children)
* new housing unattractive to parents (but are unable to say how it knows whether these new families are really childless, or just not enrolled in the the district).

The administration has ignored:

* parent dissatisfaction in the public school district, particular over school closures
* the boom in personal finances during COVID that made private school more affordable to homeowners able to refinance to low monthly payments and take equity out of their home.
* Bellevue's population is composed heavily of migrant workers, and worker migration is largely unmodeled.

We can see clearly from the local private school data that public school losses were principally to local private schools.

Since elementary school are first on the chopping block - let's look at that:

P-3 enrollment, for the school year starting in 2019/20 - 2021/22:

* Bellevue public schools lost 1354, (-19.8%)
* Bellevue private schools gained 1282 (+77.6%)

**For every 100 lost from the public district, private gained 95.**

K-5 (2019/20 - 2021/22):

* Bellevue public lost: 1143 (-12.7%)
* Bellevue private gained: 947 (+58%)

**83 were gained by private for every 100 lost to private schools.**

Brandon Adams has performed a similar analysis and found similar transfer to private (85 per 100).
https://mostlywashington.substack.com/p/how-do-births-and-housing-prices

Feedback is welcome. (Perhaps submit an issue)
I've published the analysis in workbook form so that data processing steps can be verified (and corrected), if necessary.

The analysis is for Bellevue, but its easily modified for difference districts by changing the regionSubstring (a regular expression may also be used for complex district selection).


* Public school data: https://ospi.k12.wa.us/data-reporting/data-portal
* Public schools directory: https://eds.ospi.k12.wa.us/DirectoryEDS.aspx?_gl=1*sgzoun*_ga*MzEwMTc3NDYxLjE3NTczNTA2ODI.*_ga_SQS5QZLGMR*czE3NjA5MTA5NjMkbzgkZzEkdDE3NjA5MTA5NzMkajUwJGwwJGgw*_ga_ZKSN9461S2*czE3NjA5MTA5NjMkbzgkZzEkdDE3NjA5MTA5NzMkajUwJGwwJGgw
* Private school enrollment data: https://sbe.wa.gov/our-work/private-schools#Private%20School%20Enrollment
* Private schools directory: https://sbe.wa.gov/our-work/private-schools#ApprovedSchoolsList