import requests


if __name__ == '__main__':
    url = "http://localhost:8888/predict/"
    booking = {
        'no_of_adults': 2,
        'no_of_children': 0,
        'no_of_weekend_nights': 0,
        'no_of_week_nights': 1,
        'type_of_meal_plan': 'meal_plan_1',
        'required_car_parking_space': 0,
        'room_type_reserved': 'room_type_4',
        'lead_time': 17,
        'arrival_month': 9,
        'arrival_date': 16,
        'market_segment_type': 'online',
        'repeated_guest': 0,
        'no_of_previous_cancellations': 0,
        'no_of_previous_bookings_not_canceled': 0,
        'avg_price_per_room': 183.0,
        'no_of_special_requests': 1
    }
    result = requests.post(url, json=booking).json()
    print(result)
