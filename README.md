# ACTIVATE: PREDICTIVE MAINTENANCE

## General Motivation
PredictiveMaintenance
Problems in the predictive maintenance domain range from high
operational risk due to unexpected failures and limited insight into the
root cause of problems in complex business environments. Most of these
problems can be categorized as falling under the following business
questions:

-   **What is the probability that the equipment will fail soon?**
-   What is the remaining useful life (RUL) of the equipment?
-   What are the causes of failures and what maintenance actions should
    be performed to fix these issues?

By utilizing predictive maintenance to answer these questions,
businesses can:

-   Reduce operational risk and increase rate of return on assets by
    spotting failures before they occur.
-   Reduce unnecessary time-based maintenance operations and control
    cost of maintenance
-   Improve overall brand image, eliminate bad publicity and result in
    lost sales from customer attrition.
-   Lower inventory costs by reducing inventory levels by predicting the
    reorder point.
-   Discover patterns connected to various maintenance problems.

*In this ACTIVATE case, the prediction problem is to develop a model
that predicts the likelihood that an engine will fail in each time
window.*
![Chart, line chart Description automatically
generated](./images//media/E2Earch.png)

#### Data Description

Main data sources for this case are sensor readings that collect
numerical measurements while the machinery is running. Sensor data
included sensor readings per time cycle completed. This sensor data will
be merged with other sources to determine remaining useful life and
asset information. Descriptions of terms in the data sources are shown
below:

1.  Cycles: -units of time (e.g., hours, days, etc.).

2.  S1-S21: Sensor time series measurements available at each cycle.

3.  Settings 1-3: Record of equipment settings.

4.  RUL: The remaining useful life in units of time (e.g., hours or
    cycles).

# Methodology Review
![Chart, line chart Description automatically
generated](./images//media/E2E.png)

## Merging Data Sources

Before getting into any type of feature engineering or labeling process,
we need to first prepare our data in the form required to create
features from the source data. **The goal is to generate a record for
each time unit for each asset with its features and labels to be fed
into the machine learning algorithm.**

The final table before labeling, and feature generation can be generated
by left joining machine conditions table with failure records on Asset
ID and time fields. This table can then be joined with maintenance
records on Asset ID and Time fields and finally with machine and
operator features on Asset ID. The first left join will leave null
values for failure column when machine is in normal operation, these can
be imputed by an indicator value for normal operation. This failure
column will be used to create labels for the predictive model.

## Feature Engineering

Before applying any machine learning algorithm, the first step is
feature generation. The idea of feature generation is to conceptually
describe and abstract a machine's health condition at a given time using
historical data that was collected up to that point in time. In the next
section, we provide an overview of the type of techniques that can be
used for predictive maintenance and how the labelling is done for.

### Lag Features

As mentioned earlier, in predictive maintenance, historical data usually
comes with timestamps indicating the time of collection for each piece
of data. There are many ways of creating features from the data that
comes with timestamps. In this section, we discuss some of these methods
used for predictive maintenance. However, we are not limited by these
methods alone as feature engineering is one of the most creative areas
of predictive modelling so there can be many other ways to create
features. Here, we provide some general techniques.

*Rolling Aggregates*

For each record of an asset, we pick a rolling window of size "W" which
is the number of units of time that we would like to compute historical
aggregates for. We then compute rolling aggregate features using the W
periods before the date of that record. Some examples rolling aggregates
can be rolling counts, means, standard deviations, outliers based on
standard deviations, CUSUM measures, minimum and maximum values for the
window. Another interesting technique is to capture trend changes,
spikes and level changes using algorithms that detect anomalies in data
using anomaly detection algorithms.

For demonstration, see Figure 1 where we represent sensor values
recorded for an asset for each unit of time with the blue lines and mark
the rolling average feature calculation for W=3 for the records at t1
and t2 which are indicated by orange and green groupings respectively.

![Chart, line chart Description automatically
generated](./images//media/image1.png)

Figure 1. Rolling aggregate features

Additionally, by picking a W that is very large (ex. years), it is
possible to look at the whole history of an asset such as counting all
maintenance records, failures etc. up until the time of the record. This
method was used for counting circuit breaker failures for the last three
years. Also, for train failures, all maintenance events were counted to
create a feature to capture the long-term maintenance effects.

#### Tumbling aggregates

For each labelled record of an asset, we pick a window of size "W-~k~"
where k is the number or windows of size "W" that we want to create lag
features for. "k" can be picked as a large number to capture long term
degradation patterns or a small number to capture short term effects. We
then use k tumbling windows W-k , W-(k-1) , ..., W-2 , W-1 to
create aggregate features for the periods before the record date and
time (see Figure 2). These are also rolling windows at the record level
for a time unit which is not captured in Figure 2 but the idea is the
same as in Figure 1 where t~2~ is also used to demonstrate the rolling
effect.

![Chart, line chart Description automatically
generated](./images//media/image2.png)

Figure 2. Tumbling Aggregate Features

As an example, for wind turbines, W=1 and k=3 months were used to create
lag features for each of the last 3 months using top and bottom
outliers.

### Static Features

These are technical specifications of the equipment such as manufacture
date, model number, location, etc. While lag features are mostly numeric
in nature, static features usually become categorical variables in the
models.

During feature generation, some other important steps such as handling
missing values and normalization should be performed. There are numerous
methods of missing value imputation and data normalization which will
not be discussed here. However, it is beneficial to try different
methods to see if an increase in prediction performance is possible.

The final feature table after feature engineering steps discussed in the
earlier section should resemble the following example data schema when
time unit is a day:

| Asset ID | Time      | Feature Columns     | Label      |
|----------|-----------|---------------------|------------|
| 1        | Day 1     |                     |            |
| 1        | Day 2     |                     |            |
| ...      | ...       |                     |            |
| 2        | Day 1     |                     |            |
| 2        | Day 2     |                     |            |
| ...      | ...       |                     |            |

# Modelling Techniques

Predictive Maintenance is a very rich domain often employing business
questions which may be approached from many different angles from the
predictive modeling perspective. In the next section, we will provide
main techniques that are used to model different business questions that
can be answered with predictive maintenance solutions.

## Binary Classification for Predictive Maintenance

Binary Classification for predictive maintenance is used to predict the
probability that an equipment will fail within a future time. The time
is determined by and based on business rules and the data at hand. Some
common time periods are minimum lead time required to purchase spare
parts to replace likely to damage components or time required to deploy
maintenance resources to perform maintenance routines to fix the problem
that is likely to occur within that time. We call this future horizon
period "X".

To use binary classification, we need to identify two types of examples
which we call positive and negative. Each example is a record that
belongs to a time unit for an asset conceptually describing and
abstracting its operating conditions up to that time unit through
feature engineering using historical and other data sources described
earlier. In the context of binary classification for predictive
maintenance, positive type denotes failures (label 1) and negative type
denotes normal operations (label = 0) where labels are of type
categorical. The goal is to find a model that will identify each new
example as likely to fail or operate normally within the next X units of
time.

### Label Construction

To create a predictive model to answer the question "What is the
probability that the asset will fail in the next X units of time?",
labeling is done by taking X records prior to the failure of an asset
and labeling them as "about to fail" (label = 1) while labeling all
other records as "normal" (label =0). In this method, labels are
categorical variables (see Figure 3).

![Diagram Description automatically generated with low
confidence](./images//media/image3.png)

Figure 3. Labelling for binary classification

For example, flight delays and cancellations, X is picked as one day to
predict delays in the next 24 hours. All flights that are within 24
hours before failures were labeled as 1s.
