Code for Analytics Genpact competititon

**Steps to reporduce solution for 3rd place**:
* Change file locations in config file
* run `bash run_all.sh`



**Dependencies**:

    python >= 3.6
    numpy >= 1.14.2
    pandas >= 0.23.4
    scipy >= 1.1.0
    sklearn >= 0.20.0
    lightgbm >= 2.1.2
    keras == 2.2.0
    tensorflow == 1.8.0
    

**Approach**:

#### Validation strategy
I treated it as multiple time series problem. So, creating different validation sets in time. After some testing with LB and fue to time limitation I decided to split train/val as 130/16 weeks.  

#### Feature engineering
All the time series were really noisy, so I had to very careful while feature engineering to avoid overfitting.
   * Except `city_code`, all features from main and additional files were used. 
   * Expanding/Rolling Mean of orders for (center_id, meal_id) combination with windows 5 and 120. I just chose windows randomnly, may be tuning windows might result in better score. On validation and test sets, values corresponding to first week were mapped to all weeks.
   * Expanding/Rolling Mean of orders for (center_id, meal_id) combination with windows 5 and 120. I just used window 120 feature in final model as adding window 5 median seemed to overfit on validation.
   * Similar mean feature as above for (center_id, category) seemed to improve validation score a bit.
   * Expanding/Rolling Mean of checkout price for (center_id, meal_id) and (city_code, meal_id) combinations with windows 5 and 120. On validation and test sets, values corresponding to first week were mapped to all weeks.
   * `Week of year` : week % 52 as a feature to factor in seasonality
   * `Week start` : first week where number of orders are available to factor in age of center
   * `Week gap` : no of weeks since information about num orders for that center, meal were given. The hypothesis was closing of restaurant might have some impact on weeks after reopening. It didnt to improve model much, but I still decided to keep it as it is such a cool feature :)
   * `discount` : Difference between base price and checkout price relative to base price.
   * `price_diff`: Change in base price week over week 
   * `disc_rat`: Ratio of number of orders to base price

#### Models
   * Decided to use lightgbm, extra trees regressor and simple neural network on above set of features
   * Parameter tuning for all 3 models was very crucial as well
   * Finally, i just took weighted average of all 3 models as final submission

#### Key takeaway:
   * Being a time series problem with lot of stochastic noise, careful addition of features was really important
 
#### 5 things to focus
   * Validation strategy - split train and validation sets in time
   * Careful feature engineering - while generating features at any each week, only information from previous weeks needed to be used.
   * Extra features - Features such as **week of year** and **discount** turned to be really important
   * Its okay to throw important features - although along with mean, median over different windows were powerful fatures individually, but when combined together due to high corelation they lead to overfitting. Dropping trhem resulted in improvement of mdoel.
   * Add more algorithms - Adding extra trees and simple neural network improved score
   
Thanks for reading :)
 
 

