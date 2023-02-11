# Washington School Enrollment Data

This project contains Python Jupyter workbooks as well as Microsoft PowerBI Desktop files summarizing changes in Washington school enrollment from state data.

Click the above links to:

* https://github.com/stuz5000/WASchoolEnrollment/blob/main/enrollmentPublic.ipynb
* https://github.com/stuz5000/WASchoolEnrollment/blob/main/enrollmentPrivate.ipynb

which summarize Bellevue specifically. Additionally, in the powerBI folder you can find a PBIX that will refresh data directly from the OSPI Data Portal for all schools in Washington State. This data is pre-filtered to purely Bellevue School District 405 data and can be visualized by the end user.

Although Bellevue public school enrollment is briefly in decline, it is instructive to know where the loss is coming from.

The Bellevue school district has argued:

* declining birth rates
* families selling homes and exiting the district, to be replace by (for some reason) assumed to be childless families
* high costs of living (ignoring that families in all income brackets have children)
* new housing unattractive to parents (but are unable to say how it knows whether these new families are really childless, or just not enrolled in the the district).

The administration has ignored:

* parent dissatisfaction in the private school district
* the boom in personal finances during COVID that made private school more affordable to homeowners able to refinance to low monthly payments and take equity out of their home.

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
