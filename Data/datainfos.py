#Reference: 
"Saad Haroon. (n.d.). Hotel Booking Dataset [Data set]. Kaggle. 
Retrieved November 22, 2024, from https://www.kaggle.com/datasets/saadharoon27/hotel-booking-dataset"


First few rows of the dataset:
          hotel  is_canceled  lead_time  ...                        email  phone-number       credit_card
0  Resort Hotel            0        342  ...  Ernest.Barnes31@outlook.com  669-792-1661  ************4322
1  Resort Hotel            0        737  ...       Andrea_Baker94@aol.com  858-637-6955  ************9157
2  Resort Hotel            0          7  ...   Rebecca_Parker@comcast.net  652-885-2745  ************3734
3  Resort Hotel            0         13  ...            Laura_M@gmail.com  364-656-8427  ************5677
4  Resort Hotel            0         14  ...           LHines@verizon.com  713-226-5883  ************5498

[5 rows x 36 columns]

Dataset information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 119390 entries, 0 to 119389
Data columns (total 36 columns):
 #   Column                          Non-Null Count   Dtype
---  ------                          --------------   -----
 0   hotel                           119390 non-null  object
 1   is_canceled                     119390 non-null  int64
 2   lead_time                       119390 non-null  int64
 3   arrival_date_year               119390 non-null  int64
 4   arrival_date_month              119390 non-null  object
 5   arrival_date_week_number        119390 non-null  int64
 6   arrival_date_day_of_month       119390 non-null  int64
 7   stays_in_weekend_nights         119390 non-null  int64
 8   stays_in_week_nights            119390 non-null  int64
 9   adults                          119390 non-null  int64
 10  children                        119386 non-null  float64
 11  babies                          119390 non-null  int64
 12  meal                            119390 non-null  object
 13  country                         118902 non-null  object
 14  market_segment                  119390 non-null  object
 15  distribution_channel            119390 non-null  object
 16  is_repeated_guest               119390 non-null  int64
 17  previous_cancellations          119390 non-null  int64
 18  previous_bookings_not_canceled  119390 non-null  int64
 19  reserved_room_type              119390 non-null  object
 20  assigned_room_type              119390 non-null  object
 21  booking_changes                 119390 non-null  int64
 22  deposit_type                    119390 non-null  object
 23  agent                           103050 non-null  float64
 24  company                         6797 non-null    float64
 25  days_in_waiting_list            119390 non-null  int64
 26  customer_type                   119390 non-null  object
 27  adr                             119390 non-null  float64
 28  required_car_parking_spaces     119390 non-null  int64
 29  total_of_special_requests       119390 non-null  int64
 30  reservation_status              119390 non-null  object
 31  reservation_status_date         119390 non-null  object
 32  name                            119390 non-null  object
 33  email                           119390 non-null  object
 34  phone-number                    119390 non-null  object
 35  credit_card                     119390 non-null  object
dtypes: float64(4), int64(16), object(16)
memory usage: 32.8+ MB
None

Summary statistics:
         is_canceled      lead_time  ...  required_car_parking_spaces  total_of_special_requests
count  119390.000000  119390.000000  ...                119390.000000              119390.000000
mean        0.370416     104.011416  ...                     0.062518                   0.571363
std         0.482918     106.863097  ...                     0.245291                   0.792798
min         0.000000       0.000000  ...                     0.000000                   0.000000
25%         0.000000      18.000000  ...                     0.000000                   0.000000
50%         0.000000      69.000000  ...                     0.000000                   0.000000
75%         1.000000     160.000000  ...                     0.000000                   1.000000
max         1.000000     737.000000  ...                     8.000000                   5.000000

[8 rows x 20 columns]

Column names:
Index(['hotel', 'is_canceled', 'lead_time', 'arrival_date_year',
       'arrival_date_month', 'arrival_date_week_number',
       'arrival_date_day_of_month', 'stays_in_weekend_nights',
       'stays_in_week_nights', 'adults', 'children', 'babies', 'meal',
       'country', 'market_segment', 'distribution_channel',
       'is_repeated_guest', 'previous_cancellations',
       'previous_bookings_not_canceled', 'reserved_room_type',
       'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
       'company', 'days_in_waiting_list', 'customer_type', 'adr',
       'required_car_parking_spaces', 'total_of_special_requests',
       'reservation_status', 'reservation_status_date', 'name', 'email',
       'phone-number', 'credit_card'],
      dtype='object')
      dtype='object')