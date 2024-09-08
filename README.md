# Renewable Energy's Economic Impact in the U.S. 
**Introduction & Purpose**

In recent years, the U.S. has seen a surge in the deployment of renewable energy sources, driven both by environmental concerns and economic incentives. While various news outlets (i.e. Bloomberg, Wall Street Journal) and databases (e.g. Census Bureau, Energy Information Administration) provide data on the current state of renewable energy and its economic implications, a deeper analysis is needed to understand the economic impact of different renewable energy sources across U.S. states and sectors. Our project aim is to bridge the gap, converting raw data into the prime data for analysis, examining the relationship between renewable energy production, consumption, and economic indicators such as GDP. 

These analyses can help policymakers, environmental researchers, data scientists, and large organizations like the Environmental Protection Agency (EPA) and the United Nations (U.N.) develop targeted strategies for optimizing renewable energy deployment, reducing reliance on traditional energy sources, and mitigating climate change. For example, by identifying trends in the adoption of Alternative Fuel Stations (AFS), agencies like the EPA and U.N. can allocate resources more effectively, focusing more on gathering and distributing funds to create more AFSs that accelerate the transition to renewable energy. Thus, our project's results might offer key insights for data-driven decision-making on improving the environment and the economy.

**Techniques used**

- Loading and Merging Data
- Correlation analysis
- Visualizations (scatter plots, regression plots, boxplots)
- Geospatial heatmaps/Geopandas
- Bar charts
- Residual plot
- Confusion matrix with a KNN classifier

**Insights**

Analyzed Sectors: Electric Power, Industrial, Residential, Commercial (1949-2023)
Sector with the highest total renewable energy consumption: Industrial sector at 111415.74 Trillion btu, not the Electric Power sector at 91597.39 Trillion btu. 
We think that the industrial sector has consumed the most energy between 1949 and 2023 because it takes up vast amounts of energy to support its production and manufacturing industries.                               

Each sector’s most consumed renewable energy source:
- Industrial Sector consumes biomass energy the most at 110,480.94 Trillion btu,
- Electric Power Sector consumes conventional hydroelectric power at 58,757.67 Trillion btu
- Residential Sector consumes wood energy at 36,997.09 Trillion btu
- Commercial Sector consumes biomass energy at 4,732.62 Trillion btu.
It was not surprising to see that the residential sector consumes mostly wood energy because people use wood consistently to power and heat their homes.
Biomass energy appears to be becoming the next most popular renewable energy source in industrial and commercial sectors, moving away from fossil fuels and other harmful energy sources.

While the correlations between year and electric power, industrial, and residential sectors were strong at 0.914, 0.943, and 0.93 respectively, the residential sector had a really weak correlation to year at 0.074. Its regression plots showed likewise with the residential sector having a sinusoidal curve, suggesting no relationship, while all other sectors had an increasing linear relationship. This suggests that all other sectors are focusing on increasing their total renewable energy consumption over the years and leaving climate-damaging energy sources behind (oil, fossil fuels etc.) while residential sectors have years where they increase and decrease their renewable energy consumption levels (switching between renewable and non-renewable energy consumption).

From our analysis of renewable energy consumption and production over time, we found that the top 5 states with the greatest increases in consumption since 1970 were Texas, California, Iowa, Florida, and Georgia, while the top 5 states with the greatest increases in production were Iowa, Texas, Nebraska, California, and Illinois. These trends can be attributed to various factors, including government policies, natural resource availability, and population size. 

Additionally, our analysis found strong positive correlations between a state's renewable energy consumption and production with its real GDP, with correlation coefficients of 0.8333 and 0.5530, respectively. To gain further insights, we identified the top 5 states with the highest renewable energy consumption and production relative to their GDP. Surprisingly, the same states—Washington D.C., Delaware, New Jersey, Maryland, and Massachusetts—topped the list for both metrics. This may be because states with high renewable energy production often have correspondingly high consumption, as they tend to use much of the energy they produce.

Furthermore, the Pearson correlation coefficient between renewable energy percentage and state location was found to be r = −0.2434. This indicates a weak negative correlation, suggesting that inland states generally produce slightly less renewable energy than coastal states. However, this correlation is not strong enough to draw definitive conclusions about the impact of geographic location on energy production.

Our geospatial map showed that Vermont (Inland) leads in renewable energy generation percentage, contributing 49% of its total energy from renewable sources. Following Vermont are South Dakota (Coastal): 38%, Maine (Coastal): 37%, Washington (Coastal): 35%, Idaho (Inland): 33%. This finding challenged our initial hypothesis that coastal states produce more renewable energy, suggesting that factors like natural resources, climate, policies, and infrastructures may have greater and more complex impact on energy production than location alone.

The linear regression analysis of Vermont’s wind energy production (2001 to 2023) showed a relatively high RMSE of 72.36, indicating that the model struggles to precisely predict wind energy generation. The residual plot further emphasized this variability, highlighting discrepancies between the actual and predicted values. Also, our boxplots showed that inland states tend to have higher median and mean energy generation, along with greater variability (wider IQR), suggesting more diverse energy output likely due to different climates and other influencing factors. 

Despite different optimal k-values in K-Nearest Neighbors (KNN) performance, the confusion matrix results were identical, likely due to the model's high accuracy score (0.9231) and the clear and distinct nature of the dataset's labels. This clarity in the data led to minimal misclassification, with only one state, categorized as "High hydroelectric," being incorrectly predicted as "High solar." The result indicates that the KNN classifier effectively distinguished between states' dominant renewable energy sources, making it relatively reliable for our prediction.

**Code structure**

A series of python files which contain imports, functions, and comments. Could be easily run through Anaconda Spyder. 
This project also includes csv files for datasets we used, and pdf files for our comprehensive final report and presentation.

**Sources**

Datasets used in our report: 

https://www.eia.gov/state/seds/seds-data-complete.php#Keystatisticsrankings

https://www.eia.gov/electricity/data/browser/#/topic/0?agg=2,0,1&fuel=05bc&geo=00fvvvvvvvvvo&sec=g&linechart=ELEC.GEN.HYC-CT-99.M&columnchart=ELEC.GEN.HYC-CT-99.M&map=ELEC.GEN.HYC-CT-99.M&freq=M&start=200101&end=202405&ctype=linechart&ltype=pin&columnendpoints=0&columnvalues=0&rtype=s&pin=&rse=0&maptype=0
https://coast.noaa.gov/states/

https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip

https://data.nrel.gov/system/files/150/cost_of_charging_2019_v1.xlsx

https://www.eia.gov/totalenergy/data/browser/xls.php?tbl=T10.02C&freq=m

https://www.eia.gov/totalenergy/data/browser/xls.php?tbl=T10.02B&freq=m

https://www.eia.gov/totalenergy/data/browser/xls.php?tbl=T10.02A&freq=m

https://www.eia.gov/totalenergy/data/browser/xls.php?tbl=T10.02A&freq=m

https://catalog.data.gov/dataset/alternative-fueling-station-locations-422f2

https://catalog.data.gov/dataset/electric-vehicle-population-data

Datasets for future work:
https://www.ncei.noaa.gov/ 
- contains more comprehensive datasets for region/climate specific data for analysis of renewable energy consumption and emissions across various U.S. sectors
    
**Contributions** 
- Harshini Dinesh (4381hdinesh@gmail.com) 
  - Cleaned up data on energy consumption by sector and data on accessibility of alternative fuel stations (AFS)
  - Analyzed data to find out if initial hypotheses were true
     - Hypothesis: Given the growth of the EV industry, the electric power sector has consumed more renewable energy than the industrial, commercial, and 
       residential sectors. --> False; the industrial sector has consumed more energy than all other sectors.
     - Hypothesis: Given the growth of the EV industry, there are more alternative fuel stations available to the public for electric cars than for stations 
       providing other fuels. --> True
  - Created visualizations (bar charts, lineplots) for data to show how different sectors consume different energy types,
    and to show how many AFS are accessible privately versus publicly. 
  - Conducted correlation and regression analysis (regression plots to understand how different sectors consumed energy over time (at an    
    increasing/decreasing/inconsistent rate) 
  - Evaluated how to enhance the current code to improve predictions for future energy consumption by sector and future AFS accessibility.
- Hang Hang(hang.h@husky.neu.edu)
    - Cleaned up and analyzed the data on the net generation for all sectors (renewable energy generation), the state's geographical (inland vs. coastal) locations, and the cost of charging electric vehicles
    - Used Geopandas to plot a geospatial heatmap for visualizing energy production across states
       - Hypothesis: Coastal states have a higher proportion of renewable energy production compared to landlocked states due to greater access to diverse renewable resources.
       - Results: False. Top states contain both inland and coastal areas.
    - Analyzed how state geographical locations correlate with energy production
       - Weak negative correlation at r = -0.2434 challenges the initial hypothesis noted above.
    - Developed a box plot to illustrate the distribution of renewable energy generation data across inland and coastal areas
    - Plotted a bar chart for comparing the baseline residential and workplace Levelized Cost of Charging (LCOC) for Battery Electric Vehicles (BEVs)
    - Performed a linear regression analysis on Vermont's wind energy generation over the years
    - Produced a residual plot to assess the linear regression model’s fit and accuracy
    - Conducted a KNN Classifier analysis to classify states based on their renewable energy generation profile such as "High hydroelectric", and "High Wind", and generated a confusion matrix to evaluate the model's performance
- Ryan Jiang (jiang.ry@husky.neu.edu)
  - Cleaned up data on state renewable energy consumption, production, price, and GDP from 1970-2020
  - Calculated basic statistics and correlation, creating line charts, scatter plots, and bar plots to see which states/regions had the greatest changes in 
    energy prices, consumption, and production over time. This also proved our hypothesis true: there is a positive correlation between a state's GDP and its 
    energy consumption.
  - Analyzed the change in renewable energy consumption and production for each state from 1970 onwards, identifying the states with the most significant changes.
  - Determined the top states with the highest ratios of energy consumption and production relative to real GDP.
- Hersh Joshi (jhersh003@gmail.com)
  - Analyzed the impact of population, climate, geographical location on energy consumption.
  - Compared EV charging costs with traditional fuel costs 
